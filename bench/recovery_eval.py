"""Live check on the two held-out generator topics (tool hire, parking). Runs on the rented machine.

    python bench/recovery_eval.py --name mine_multistep --model /workspace/models/gguf/mine_multistep-F16.gguf --tools-in user
    python bench/recovery_eval.py --selftest          (no model: replays the scripted answers; every task must pass)

120 multi-step tasks written by training/recovery_world.py on two topics whose tools never
appear in any training file. Unlike the frozen set this is a loop: the model's calls are run
against the simulated world, the results are fed back, and it goes on until it answers in
text (at most 10 model turns). Greedy, up to 384 tokens per turn, GPU, not timed.

The prompt is rendered with the base model's chat template through transformers, the same
call that renders the training file, and sent to llama-server as raw text. When a reply holds
several calls, each is run in order and written into the conversation as its own call and
result, because Llama 3.2's template takes one call per assistant turn.

A task is correct when
  - the conversation ended with a text answer, no reply ran to the token limit, and
  - the world ends in the state the task asks for: the same records and the same notes, and
  - for a task that cannot be completed as asked (unknown or ambiguous name), the model did
    look the name up before answering.
What is not checked: the wording of the final answer (for example whether it admits that
one of two parts failed). Also reported: tasks with at least one tool error, tasks where the
model made a lookup call after its first error, and tasks that ended on the turn right
after the first error.

File: results/part2/recovery_eval_<name>_<layout>.json
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "bench"))

import recovery_world as rw  # noqa: E402

SEED, TASKS, MAX_TURNS, MAX_TOKENS = 4041, 120, 10, 384


def run_task(task: dict, reply_to) -> dict:
    """reply_to(messages, tools) -> (text, at_limit). Returns the transcript and the verdict."""
    world = copy.deepcopy(task["world"])
    names = world.t["tools"]
    lookups = {names["find_item"][0], names["find_party"][0], names["list"][0]}
    messages, tools = [task["messages"][0]], task["tools"]
    transcript, errors, first_error_turn, lookup_after_error, party_lookup, at_limit, ended, turns = [], 0, None, False, False, False, False, 0
    for turn in range(MAX_TURNS):
        text, limit = reply_to(messages, tools)
        turns, at_limit = turn + 1, at_limit or limit
        calls = text_calls(text)
        transcript.append({"model": text})
        if not calls:
            ended = True
            break
        for call in calls:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            result, failed = world.run(name, arguments)
            if first_error_turn is not None and turn > first_error_turn and name in lookups:
                lookup_after_error = True
            party_lookup = party_lookup or name == names["find_party"][0]
            if failed:
                errors += 1
                if first_error_turn is None:
                    first_error_turn = turn
            messages = messages + [{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]},
                                   {"role": "tool", "content": result}]
            transcript.append({"tool": result})
    state_ok = world.state() == task["state_after"]
    correct = ended and not at_limit and state_ok and (party_lookup or not task["ends_with_question"])
    return {"topic": task["topic"], "pattern": task["pattern"], "seed": task["seed"], "user": task["messages"][0]["content"],
            "correct": correct, "state_ok": state_ok, "ended_with_text": ended, "at_limit": at_limit, "model_turns": turns,
            "tool_errors": errors, "lookup_after_first_error": lookup_after_error if errors else None,
            "ended_right_after_first_error": (ended and turns == first_error_turn + 2) if errors else None,
            "transcript": transcript}


def text_calls(text: str) -> list[dict]:
    import toolcall_eval as te
    return te.calls_in_text(text)[0]


def summarise(rows: list[dict]) -> dict:
    def part(subset: list[dict]) -> dict:
        with_error = [r for r in subset if r["tool_errors"]]
        return {"tasks": len(subset), "correct": sum(r["correct"] for r in subset),
                "percent": round(100 * sum(r["correct"] for r in subset) / len(subset), 1),
                "tasks_with_a_tool_error": len(with_error),
                "of_those_lookup_after_the_error": sum(bool(r["lookup_after_first_error"]) for r in with_error),
                "of_those_ended_right_after_the_error": sum(bool(r["ended_right_after_first_error"]) for r in with_error),
                "at_limit": sum(r["at_limit"] for r in subset), "no_text_answer": sum(not r["ended_with_text"] for r in subset)}
    out = {"all": part(rows)}
    for key in ("topic", "pattern"):
        for value in sorted({r[key] for r in rows}):
            out[f"{key}: {value}"] = part([r for r in rows if r[key] == value])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name")
    parser.add_argument("--model")
    parser.add_argument("--tools-in", choices=["user", "system"], default="user")
    parser.add_argument("--base", default="/workspace/models/llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--limit", type=int, default=TASKS)
    args = parser.parse_args()
    tasks = rw.conversations(rw.HELDOUT_TOPICS, TASKS, SEED)[: args.limit]

    if args.selftest:
        failed = 0
        for task in tasks:
            scripted = iter([json.dumps({"name": m["tool_calls"][0]["function"]["name"], "parameters": m["tool_calls"][0]["function"]["arguments"]})
                             if m.get("tool_calls") else m["content"]
                             for i, m in enumerate(task["messages"]) if m["role"] == "assistant" and i not in task["context_only"]])
            row = run_task(task, lambda messages, tools: (next(scripted), False))
            failed += not row["correct"]
        print(f"selftest: {len(tasks) - failed} of {len(tasks)} scripted conversations pass the check",
              collections.Counter(t["pattern"] for t in tasks))
        raise SystemExit(1 if failed else 0)

    import first_hour as fh
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    in_user = args.tools_in == "user"

    def render(messages, tools) -> str:
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, tools_in_user_message=in_user)
        # llama-server adds the start-of-text token itself; its own template output does not carry it
        return text.removeprefix("<|begin_of_text|>")

    out = ROOT / "results" / "part2" / f"recovery_eval_{args.name}_{args.tools_in}.json"
    server_args = argparse.Namespace(context=8192, slots=1, cpu=False, threads=None, head=None, draft_max=0, draft_p_min=0.0,
                                     target=args.model, spec_type="draft-eagle3", server_bin=None, extra="--jinja")
    server = fh.start_server(server_args, out.with_suffix(".server.log"))
    rows, begin = [], time.time()
    try:
        first = tasks[0]
        served = fh.post("/apply-template", {"messages": first["messages"][:1], "tools": first["tools"],
                                             "chat_template_kwargs": {"enable_thinking": False, "tools_in_user_message": in_user}})["prompt"]
        same_prompt = served == render(first["messages"][:1], first["tools"])

        def reply_to(messages, tools):
            reply = fh.post("/completion", {"prompt": render(messages, tools), "n_predict": MAX_TOKENS, "temperature": 0.0, "top_k": 1,
                                            "seed": 0, "cache_prompt": False})
            return reply["content"], reply.get("stop_type") == "limit" or (reply.get("tokens_predicted") or 0) >= MAX_TOKENS

        for task in tasks:
            rows.append(run_task(task, reply_to))
    finally:
        server.terminate()
        server.wait(timeout=60)
    record = {"name": args.name, "model": Path(args.model).name, "tools_in": args.tools_in, "tasks": len(rows), "tasks_seed": SEED,
              "topics": rw.HELDOUT_TOPICS, "max_model_turns": MAX_TURNS, "max_tokens": MAX_TOKENS, "decoding": "greedy",
              "first_prompt_same_as_the_server_template": same_prompt,
              "llama_cpp_commit": Path("/workspace/llama_commit.txt").read_text().split()[0],
              "wall_seconds": round(time.time() - begin, 1), "finished_unix": time.time(), "summary": summarise(rows), "rows": rows}
    out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    print(f"{args.name} {args.tools_in}: first prompt same as the server's template: {same_prompt}")
    for key, value in record["summary"].items():
        print(f"{key:28s} {value['correct']:3d} of {value['tasks']:3d}  errors in {value['tasks_with_a_tool_error']:3d}, "
              f"lookup after {value['of_those_lookup_after_the_error']:3d}, stopped after {value['of_those_ended_right_after_the_error']:3d}")
    print(f"recovery_eval DONE {args.name} {args.tools_in}", flush=True)


if __name__ == "__main__":
    main()
