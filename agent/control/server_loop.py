"""The control for part 6: the same tasks through stock llama-server, driven by a Python loop.

    python agent/control/server_loop.py --model <gguf> --agent <specedge-agent> --tasks <jsonl> \
        --system agent/tasks/system_prompt.txt --out <results file> --mode raw --cache 0

Runs on the rented machine. Starts an unmodified llama-server (pinned commit) and, for each
task, loops: prompt, reply, run the tool calls, feed the results back, until the model
answers in plain text or the turn limit is reached. The tools and the checker are the
runtime's own, reached through `specedge-agent toolhost`, so both sides act on the same world
and are judged by the same code.

  --mode chat  the server's chat endpoint does everything a stock user would let it do: it
               builds the prompt from the messages and the tool list and parses the tool
               calls itself. Arguments go to the tools as the server returns them. At the
               pinned commit it answers HTTP 500 when the model writes several calls in one
               reply; that ends the task as failed.
  --mode raw   the server builds the prompt (/apply-template) and returns the raw text
               (/completion); the calls are read by the runtime's parser (toolhost). This
               isolates speed: the model sees the same prompts as in the runtime.

  --cache 0|1  the server's own prompt cache (cache_prompt) off or on.

Timings are the server's own (prompt_ms, predicted_ms) plus wall-clock per task.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "/workspace/llama.cpp/build/bin/llama-server"


class Server:
    def __init__(self, args: argparse.Namespace, log_path: Path) -> None:
        self.port = args.port
        command = [SERVER, "-m", args.model, "--port", str(args.port), "--host", "127.0.0.1", "-c", str(args.ctx),
                   "-np", "1", "--seed", "0", "-ngl", str(args.ngl), "-t", str(args.threads), "--jinja"]
        self.command = command
        self.log = open(log_path, "w")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
        for _ in range(900):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:  # noqa: BLE001 - still loading
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited while loading") from None
                time.sleep(1)
        raise RuntimeError("llama-server did not become ready")

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"HTTP {err.code}: {err.read().decode(errors='replace')[:300]}") from None

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=60)
        self.log.close()


class ToolHost:
    def __init__(self, args: argparse.Namespace) -> None:
        self.process = subprocess.Popen([args.agent, "toolhost", "--tools", args.tools, "--contacts", args.contacts],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")

    def ask(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def stop(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=10)


def run_task(task: dict, server: Server, host: ToolHost, tools: list[dict], system: str, args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    host.ask({"op": "reset", "initial": task["initial"]})
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": task["user"]}]
    turns, final_answer, ended, error = [], "", "turn_limit", ""
    sampling = {"temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": bool(args.cache)}
    template = {"chat_template_kwargs": {"tools_in_user_message": False}} if args.tools_in == "system" else {}
    first_turn_state_ok = -1

    def state_ok() -> bool:
        verdict = host.ask({"op": "check", "expect": task["expect"], "final_answer": ""})
        return all(verdict[key] for key in ("events_ok", "expenses_ok", "messages_ok", "tools_ok"))

    try:
        for turn in range(args.max_turns):
            record = {"turn": turn, "calls": []}
            if args.mode == "raw":
                built = time.perf_counter()
                prompt = server.post("/apply-template", {"messages": messages, "tools": tools, **template})["prompt"]
                record["build_ms"] = (time.perf_counter() - built) * 1000
                reply = server.post("/completion", {"prompt": prompt, "n_predict": args.max_tokens, **sampling})
                text, timings = reply["content"], reply["timings"]
                parsed = host.ask({"op": "parse", "text": text})
                calls = [{"name": c["name"], "arguments": c["arguments"], "id": None} for c in parsed["calls"]]
                broken, answer = parsed["broken"], parsed["text"]
                record.update(text=text, broken_call=broken, documented_format=parsed["documented_format"],
                              values_converted=parsed["converted"], hit_token_limit=bool(reply.get("truncated")) or
                              reply.get("stop_type") == "limit")
                messages.append({"role": "assistant", "content": text})
            else:
                reply = server.post("/v1/chat/completions", {"messages": messages, "tools": tools,
                                                             "max_tokens": args.max_tokens, **sampling, **template})
                message, timings = reply["choices"][0]["message"], reply["timings"]
                calls, broken, answer = [], False, (message.get("content") or "").strip()
                for call in message.get("tool_calls") or []:
                    arguments = call["function"].get("arguments")
                    try:
                        arguments = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
                    except ValueError:
                        broken = True
                        continue
                    calls.append({"name": call["function"]["name"], "arguments": arguments, "id": call.get("id")})
                record.update(text=message.get("content"), tool_calls=message.get("tool_calls"), broken_call=broken,
                              hit_token_limit=reply["choices"][0].get("finish_reason") == "length", build_ms=0.0)
                messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})
            record.update(prompt_tokens=timings.get("prompt_n", 0) + timings.get("cache_n", 0),
                          reused_tokens=timings.get("cache_n", 0), processed_tokens=timings.get("prompt_n", 0),
                          prompt_ms=timings.get("prompt_ms", 0.0), written_tokens=timings.get("predicted_n", 0),
                          generation_ms=timings.get("predicted_ms", 0.0))
            turns.append(record)

            if not calls and not broken:
                final_answer, ended = answer, "answered"
                record["state_ok"] = state_ok()
                break
            if not calls:
                messages.append({"role": "tool", "content": json.dumps({"error": "the tool call was not valid JSON"})})
            for call in calls:
                result = host.ask({"op": "call", "name": call["name"], "arguments": call["arguments"]})
                record["calls"].append({"name": call["name"], "arguments": call["arguments"], "result": result["result"]})
                tool_message = {"role": "tool", "content": result["text"]}
                if call["id"]:
                    tool_message["tool_call_id"] = call["id"]
                messages.append(tool_message)
            record["state_ok"] = state_ok()
            if first_turn_state_ok < 0 and record["state_ok"]:
                first_turn_state_ok = turn
    except Exception as problem:  # noqa: BLE001 - a server refusal is a result of the control, not a crash
        ended, error = "error", f"{type(problem).__name__}: {problem}"

    verdict = host.ask({"op": "check", "expect": task["expect"], "final_answer": final_answer})
    if ended != "answered":
        verdict["correct"] = verdict["correct_strict"] = False
    total = lambda key: sum(t.get(key, 0) or 0 for t in turns)  # noqa: E731
    return {"id": task["id"], "family": task["family"], "correct": verdict["correct"],
            "correct_strict": verdict.get("correct_strict"), "verdict": verdict, "ended": ended,
            "first_turn_state_ok": first_turn_state_ok, "error": error, "turns_used": len(turns), "calls_made": sum(len(t["calls"]) for t in turns),
            "wall_ms": (time.perf_counter() - start) * 1000, "prompt_ms": total("prompt_ms"),
            "generation_ms": total("generation_ms"), "build_ms": total("build_ms"), "prompt_tokens": total("prompt_tokens"),
            "processed_tokens": total("processed_tokens"), "written_tokens": total("written_tokens"),
            "final_answer": final_answer, "final_state": host.ask({"op": "state"}), "turns": turns}


def first_line_with(path: str, key: str) -> str:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent", required=True, help="the specedge-agent binary (for toolhost)")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tools", default=str(here / "tools" / "schemas"))
    parser.add_argument("--contacts", default=str(here / "tools" / "contacts.json"))
    parser.add_argument("--system", default=str(here / "tasks" / "system_prompt.txt"))
    parser.add_argument("--mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--cache", type=int, default=0)
    parser.add_argument("--tools-in", choices=["user", "system"], default="user",
                        help="where the template puts the tool list (its tools_in_user_message variable)")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--port", type=int, default=8171)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--name", default="control")
    parser.add_argument("--timing", type=int, default=0, help="1 only for a booked, quiet machine")
    parser.add_argument("--note", default="")
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--tasks-sha256", default="")
    parser.add_argument("--gpu", default="")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        tasks = [task for task in tasks if task["id"] in set(args.ids.split(","))]
    if args.limit:
        tasks = tasks[:args.limit]
    system = Path(args.system).read_text(encoding="utf-8") if args.system else ""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    started, begin = time.time(), time.perf_counter()
    host = ToolHost(args)
    tools = host.ask({"op": "tools"})["tools"]
    server = Server(args, out.with_suffix(".server.log"))
    rows = []
    try:
        server.post("/completion", {"prompt": "Hello", "n_predict": 4, "cache_prompt": False})      # warm up
        for index, task in enumerate(tasks):
            row = run_task(task, server, host, tools, system, args)
            rows.append(row)
            print(f"[{index + 1}/{len(tasks)}] {row['id']} {'correct' if row['correct'] else 'wrong'}  {row['ended']}  "
                  f"turns {row['turns_used']}  {row['wall_ms'] / 1000:.1f} s  {row['error'][:120]}", flush=True)
    finally:
        server.stop()
        host.stop()

    n = max(len(rows), 1)
    seconds = lambda key: [row[key] / 1000 for row in rows]  # noqa: E731
    families: dict[str, dict] = {}
    for row in rows:
        family = families.setdefault(row["family"], {"correct": 0, "tasks": 0})
        family["correct"] += bool(row["correct"])
        family["tasks"] += 1
    commit = Path("/workspace/llama_commit.txt")
    record = {
        "run_id": args.name, "part": 6, "program": "agent/control/server_loop.py + stock llama-server",
        "timing_valid": bool(args.timing), "note": args.note,
        "config": {"mode": args.mode, "cache": bool(args.cache), "forced_json": False, "max_turns": args.max_turns,
                   "max_tokens": args.max_tokens, "decoding": "greedy", "tool_results": "stock", "tools_in": args.tools_in, "n_ctx": args.ctx,
                   "threads": args.threads, "n_gpu_layers": args.ngl, "server_command": server.command},
        "seed": 0,
        "model": {"path": args.model, "file": Path(args.model).name, "bytes": Path(args.model).stat().st_size,
                  "sha256": args.model_sha256},
        "llama_cpp_commit": commit.read_text().split()[0] if commit.exists() else "unknown",
        "tasks_file": args.tasks, "tasks_sha256": args.tasks_sha256, "tools": len(tools),
        "hardware": {"cpu": first_line_with("/proc/cpuinfo", "model name"), "cpus_visible": os.cpu_count(),
                     "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else "",
                     "memory": first_line_with("/proc/meminfo", "MemTotal"), "gpu": args.gpu},
        "started_unix": started, "wall_seconds": time.perf_counter() - begin,
        "summary": {"tasks": len(rows), "correct": sum(bool(row["correct"]) for row in rows),
                    "correct_strict": sum(bool(row["correct_strict"]) for row in rows), "by_family": families,
                    "reached_expected_state_at_some_turn": sum(
                        row["first_turn_state_ok"] >= 0 or all(row["verdict"][k] for k in
                                                               ("events_ok", "expenses_ok", "messages_ok", "tools_ok"))
                        for row in rows),
                    "ended_with_an_answer": sum(row["ended"] == "answered" for row in rows),
                    "errors": sum(row["ended"] == "error" for row in rows),
                    "seconds_per_task": {"wall_mean": sum(seconds("wall_ms")) / n,
                                         "wall_median": statistics.median(seconds("wall_ms")) if rows else 0.0,
                                         "prompt_mean": sum(seconds("prompt_ms")) / n,
                                         "generation_mean": sum(seconds("generation_ms")) / n,
                                         "build_prompt_mean": sum(seconds("build_ms")) / n},
                    "seconds_total": {"wall": sum(seconds("wall_ms")), "prompt": sum(seconds("prompt_ms")),
                                      "generation": sum(seconds("generation_ms"))}},
        "tasks": rows}
    out.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{record['summary']['correct']} of {len(rows)} correct; mean {record['summary']['seconds_per_task']['wall_mean']:.2f} s "
          f"per task; wrote {out}")


if __name__ == "__main__":
    main()
