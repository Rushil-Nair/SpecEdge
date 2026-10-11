"""Part 3: accuracy, speed and memory of one model at each quantization level.

    python quant/run.py --name mine --what accuracy
    python quant/run.py --name mine --what speed --device gpu
    python quant/run.py --name mine --what speed --device cpu

Runs on the rented machine after quant/make.sh has made <name>-<level>.gguf for the five
levels. Every combination writes one file in results/part3/, and a finished combination is
skipped, so the script can be stopped and started again.

  accuracy   the whole frozen tool-call set on the GPU, once per level, with forced JSON off
             and on (bench/forced_json.py). Greedy decoding, so one pass is the result.
             File: <name>_<level>_<json|free>_accuracy.json, same layout as bench/toolcall_eval.py.
  speed      a fixed 60-task subset of the frozen set (20 call, 20 should-ask, 20 no-tool,
             chosen with a fixed seed; on the CPU 15 of those 60, 5 of each type), three
             passes, on the GPU or on the CPU (7 threads, as in round 1), forced JSON off and on. Reports generation tokens/s, prompt
             tokens/s, time to the first token, the server's peak memory and, on the GPU,
             GPU memory in use. File: <name>_<level>_<json|free>_speed_<gpu|cpu>.json

F16 is the control row of the table. The 1B-as-draft column of the original plan is not run:
an earlier check measured that drafting does not help the small model, so the
small model runs without it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import first_hour as fh  # noqa: E402
import forced_json  # noqa: E402
import toolcall_eval as te  # noqa: E402

LEVELS = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]
TASKS = ROOT / "data" / "toolset" / "v1" / "toolset_v1.jsonl"
GGUF = Path("/workspace/models/gguf")
OUT = ROOT / "results" / "part3"
CPU_THREADS, MAX_TOKENS, SPEED_SEED, SPEED_PER_TYPE, CPU_SPEED_PER_TYPE = 7, 384, 3, 20, 5


def load_tasks() -> list[dict]:
    te_frozen = json.loads(TASKS.with_name("FROZEN.json").read_text(encoding="utf-8"))
    import hashlib
    if hashlib.sha256(TASKS.read_bytes()).hexdigest() != te_frozen["sha256"]:
        raise SystemExit("the task file does not match its frozen checksum")
    return [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines()]


def speed_subset(tasks: list[dict], device: str = "gpu") -> list[dict]:
    """60 tasks on the GPU. On the CPU the first 5 of each type's 20 (15 tasks, all among the
    GPU's 60): F16 on 7 threads reads a prompt at tens of tokens a second, and 60 tasks times
    three passes times ten files per model would hold the machine for most of a day."""
    rng = random.Random(SPEED_SEED)
    keep = SPEED_PER_TYPE if device == "gpu" else CPU_SPEED_PER_TYPE
    return [task for kind in ("call", "ask", "no_tool")
            for task in rng.sample([t for t in tasks if t["type"] == kind], SPEED_PER_TYPE)[:keep]]


def peak_rss_gb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1e6, 2)
    except OSError:
        pass
    return None


def gpu_used_mib() -> int | None:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True)
    return int(result.stdout.split()[0]) if result.returncode == 0 and result.stdout.strip() else None


def ask(task: dict, forced: bool) -> dict:
    prompt = fh.post("/apply-template", {"messages": task["messages"], "tools": te.sendable(task["tools"])})["prompt"]
    body = {"prompt": prompt, "n_predict": MAX_TOKENS, "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False}
    if forced:
        body["grammar"] = forced_json.grammar_for(task["tools"])
    return fh.post("/completion", body)


def start(model: Path, cpu: bool, log: Path):
    arguments = argparse.Namespace(context=8192, slots=1, cpu=cpu, threads=CPU_THREADS if cpu else None, head=None,
                                   draft_max=0, draft_p_min=0.0, target=str(model), spec_type="draft-eagle3",
                                   server_bin=None, extra="--jinja")
    return fh.start_server(arguments, log)


def common(name: str, level: str, forced: bool, model: Path) -> dict:
    return {"model": name, "level": level, "file": model.name, "file_bytes": model.stat().st_size,
            "forced_json": forced, "tasks_file": str(TASKS.relative_to(ROOT)), "decoding": "greedy",
            "max_tokens": MAX_TOKENS, "llama_cpp_commit": Path("/workspace/llama_commit.txt").read_text().split()[0],
            "hardware": {"cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                      if line.startswith("model name")), ""), "cpus_visible": os.cpu_count(),
                         "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()}}


def accuracy(name: str, level: str, forced: bool, tasks: list[dict]) -> None:
    path = OUT / f"{name}_{level}_{'json' if forced else 'free'}_accuracy.json"
    if path.exists():
        return
    model = GGUF / f"{name}-{level}.gguf"
    server = start(model, False, path.with_suffix(".server.log"))
    rows, begin = [], time.time()
    try:
        for task in tasks:
            text, error, ended = None, None, {}
            try:
                reply = ask(task, forced)
                text, ended = reply["content"], te.limit_fields(reply, MAX_TOKENS)
            except Exception as err:  # noqa: BLE001 - a refused grammar or request is a result
                error = f"{type(err).__name__}: {str(err)[:200]}"
            rows.append(te.with_limit_rule({"id": task["id"], "source": task["source"], "type": task["type"],
                                            "tags": task.get("tags", []), "expect": task["expect"], "content": text,
                                            "error": error, **ended, **te.score_text(task, text, error)}))
    finally:
        server.terminate()
        server.wait(timeout=60)
    record = {**common(name, level, forced, model), "what": "accuracy", "device": "gpu",
              "wall_seconds": round(time.time() - begin, 1), "finished_unix": time.time(),
              "summary": te.summarise(rows), "rows": rows}
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    values = record["summary"]
    print(f"{name} {level} forced={forced}: all {values['all']['fully_correct']}  "
          f"call {values['call tasks']['fully_correct']}  ask {values['no-call: ask for information']['fully_correct']}  "
          f"no tool {values['no-call: no tool fits']['fully_correct']}  failed {values['all']['request_failed']}", flush=True)


def same_as_accuracy_run(name: str, level: str, forced: bool, rows: list[dict]) -> str | None:
    """Sanity check on the model file: how many of the timed replies equal the reply the same
    file gave to the same task in the accuracy run (GPU). A wrong or damaged file shows here."""
    path = OUT / f"{name}_{level}_{'json' if forced else 'free'}_accuracy.json"
    if not path.exists():
        return None
    before = {row["id"]: row["content"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    return f"{sum(before.get(row['id']) == row['content'] for row in rows)} of {len(rows)}"


def speed(name: str, level: str, forced: bool, device: str, tasks: list[dict]) -> None:
    path = OUT / f"{name}_{level}_{'json' if forced else 'free'}_speed_{device}.json"
    if path.exists():
        return
    model = GGUF / f"{name}-{level}.gguf"
    idle_gpu = gpu_used_mib()
    server = start(model, device == "cpu", path.with_suffix(".server.log"))
    passes = []
    try:
        ask(tasks[0], forced)                                        # warm-up, not counted
        for _ in range(3):
            rows = []
            for task in tasks:
                reply = ask(task, forced)
                timings = reply["timings"]
                rows.append({"id": task["id"], "prompt_n": timings["prompt_n"], "prompt_ms": timings["prompt_ms"],
                             "predicted_n": timings["predicted_n"], "predicted_ms": timings["predicted_ms"],
                             "content": reply["content"]})
            passes.append({
                "generation_tok_s": 1000 * sum(r["predicted_n"] for r in rows) / sum(r["predicted_ms"] for r in rows),
                "prompt_tok_s": 1000 * sum(r["prompt_n"] for r in rows) / sum(r["prompt_ms"] for r in rows),
                # time until the first token is written: reading the prompt, then one generation step
                "first_token_ms_median": statistics.median(r["prompt_ms"] + r["predicted_ms"] / max(r["predicted_n"], 1) for r in rows),
                "rows": rows})
        sanity = same_as_accuracy_run(name, level, forced, passes[0]["rows"])
        memory = {"server_peak_rss_gb": peak_rss_gb(server.pid),
                  "gpu_mib_in_use": None if device == "cpu" else gpu_used_mib(), "gpu_mib_before_start": idle_gpu}
    finally:
        server.terminate()
        server.wait(timeout=60)

    def spread(key: str) -> dict:
        values = [p[key] for p in passes]
        return {"median": statistics.median(values), "min": min(values), "max": max(values)}

    record = {**common(name, level, forced, model), "what": "speed", "device": device,
              "threads": CPU_THREADS if device == "cpu" else None, "tasks": len(tasks), "repeats": 3,
              "generation_tok_s": spread("generation_tok_s"), "prompt_tok_s": spread("prompt_tok_s"),
              "first_token_ms": spread("first_token_ms_median"), "memory": memory,
              "replies_same_as_accuracy_run": sanity,
              "finished_unix": time.time(), "passes": passes}
    path.write_text(json.dumps(record), encoding="utf-8")
    g, f = record["generation_tok_s"], record["first_token_ms"]
    print(f"{name} {level} forced={forced} {device}: {g['median']:.1f} tok/s ({g['min']:.1f}-{g['max']:.1f}), "
          f"first token {f['median']:.0f} ms, peak memory {memory['server_peak_rss_gb']} GB, GPU {memory['gpu_mib_in_use']} MiB, "
          f"replies same as the accuracy run {sanity}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--what", choices=["accuracy", "speed"], required=True)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--levels", nargs="+", default=LEVELS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    for level in args.levels:
        for forced in (False, True):
            if args.what == "accuracy":
                accuracy(args.name, level, forced, tasks)
            else:
                speed(args.name, level, forced, args.device, speed_subset(tasks, args.device))
    print(f"quant_run DONE {args.name} {args.what} {args.device}", flush=True)


if __name__ == "__main__":
    main()
