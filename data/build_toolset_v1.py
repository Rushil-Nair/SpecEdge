"""Build the candidate for the full tool-call test set (part 1). Runs on the laptop.

    python data/build_toolset_v1.py

Reads the two source files already downloaded on the laptop and writes
data/toolset/v1/toolset_v1.jsonl. Tasks listed in data/toolset/v1/dropped_unstated_values.json
(expected values the question never gives, found with data/flag_unstated_values.py and read
by hand) are left out.

VERSION 1 IS FROZEN (2026-10-08). Do not re-run this script to change it:
its SHA-256 is in data/toolset/v1/FROZEN.json and bench/toolcall_eval.py refuses a file that
does not match. A change means a new version folder and a re-run of every earlier row.

Composition (single-turn only; by design multi-turn behaviour is measured by the
100 scripted agent tasks of part 6, not by this set):

  xLAM function-calling 60k     call tasks: the right answer is one or more calls
  When2Call test_mcq            three variants of the same BFCL request, kept together so
                                that the only thing that differs is whether a call is right:
      tool_call                 the tool is there and the request is complete
      request_for_info          the request leaves out something the tool requires
      cannot_answer             the tool that could do it has been removed

Every task carries tags used for reporting: "decimal" (an expected argument is a non-integer
number), "multi_call" (more than one expected call). When2Call's expected calls come from
BFCL and list the acceptable values for each argument; they are stored under
"alternatives" and the scorer accepts any one of them ("" means the argument may be left out).

The xLAM tasks' tool names are written to data/toolset/xlam_tool_names.json. Training data
must not contain any example that lists one of them.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

from schema_utils import json_type_names, to_json_schema

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "hf_cache"
OUT = ROOT / "data" / "toolset" / "v1"
SEED = 2026
XLAM_CALL_TASKS = 330                    # before dropping; about one in fifteen is dropped
ASK, NO_TOOL, W2C_CALL = 130, 90, 70     # When2Call variants, from 130 source requests


def tags_for(calls: list[dict]) -> list[str]:
    values = [v for call in calls for v in call.get("arguments", {}).values()]
    tags = []
    if any(isinstance(v, float) and v != int(v) for v in values):
        tags.append("decimal")
    if len(calls) > 1:
        tags.append("multi_call")
    return tags


def xlam_tasks(rng: random.Random) -> list[dict]:
    source = glob.glob(str(CACHE / "**" / "xlam_function_calling_60k.json"), recursive=True)[0]
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    tasks = []
    for row in rng.sample(rows, XLAM_CALL_TASKS):
        tools = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": to_json_schema(t.get("parameters") or {}, keep_defaults=True)}} for t in json.loads(row["tools"])]
        calls = json.loads(row["answers"])
        tasks.append({"id": f"xlam-{row['id']}", "source": "xlam-function-calling-60k", "type": "call",
                      "tags": tags_for(calls), "messages": [{"role": "user", "content": row["query"]}],
                      "tools": tools, "expect": {"kind": "calls", "calls": calls}})
    return tasks


def when2call_tools(row: dict) -> list[dict]:
    tools = []
    for raw in row["tools"]:
        spec = json.loads(raw)
        parameters = json_type_names(dict(spec.get("parameters") or {}))
        parameters["type"] = "object"
        tools.append({"type": "function", "function": {
            "name": spec["name"], "description": spec.get("description", ""), "parameters": parameters}})
    return tools


def when2call_tasks(rng: random.Random) -> list[dict]:
    source = glob.glob(str(CACHE / "**" / "when2call_test_mcq.jsonl"), recursive=True)[0]
    by_request: dict[str, dict[str, dict]] = {}
    for line in open(source, encoding="utf-8"):
        row = json.loads(line)
        by_request.setdefault(row["source_id"], {})[row["correct_answer"]] = row
    complete = sorted(k for k, v in by_request.items() if len(v) == 3)
    chosen = rng.sample(complete, ASK)
    tasks = []
    for position, request in enumerate(chosen):
        variants = by_request[request]
        wanted = ["request_for_info"] + (["cannot_answer"] if position < NO_TOOL else []) + \
                 (["tool_call"] if position < W2C_CALL else [])
        for kind in wanted:
            row = variants[kind]
            task = {"id": f"when2call-{row['uuid']}", "source": f"When2Call test_mcq ({row['source']})",
                    "request": request, "messages": [{"role": "user", "content": row["question"]}],
                    "tools": when2call_tools(row)}
            if kind == "tool_call":
                call = json.loads(row["answers"]["tool_call"])
                task.update(type="call", tags=["bfcl_alternatives"],
                            expect={"kind": "calls", "calls": [{"name": call["name"], "alternatives": call["arguments"]}]})
            else:
                task.update(type="ask" if kind == "request_for_info" else "no_tool", tags=[],
                            expect={"kind": "no_call", "category": kind})
            tasks.append(task)
    return tasks


def main() -> None:
    rng = random.Random(SEED)
    tasks = xlam_tasks(rng) + when2call_tasks(rng)
    dropped_path = OUT / "dropped_unstated_values.json"
    dropped = json.loads(dropped_path.read_text(encoding="utf-8"))["dropped"] if dropped_path.exists() else {}
    kept = [t for t in tasks if t["id"] not in dropped]
    random.Random(SEED).shuffle(kept)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "toolset_v1.jsonl", "w", encoding="utf-8", newline="\n") as out:
        for task in kept:
            out.write(json.dumps(task, ensure_ascii=False) + "\n")
    names = sorted({tool["function"]["name"] for t in kept if t["source"].startswith("xlam") for tool in t["tools"]})
    (OUT / "xlam_tool_names.json").write_text(json.dumps({"xlam_tool_names": names}, indent=0), encoding="utf-8")
    counts = {kind: sum(t["type"] == kind for t in kept) for kind in ("call", "ask", "no_tool")}
    print(f"{len(tasks)} built, {len(tasks) - len(kept)} dropped, {len(kept)} kept: {counts}; "
          f"decimal {sum('decimal' in t['tags'] for t in kept)}, multi-call {sum('multi_call' in t['tags'] for t in kept)}, "
          f"xLAM tool names {len(names)}")


if __name__ == "__main__":
    main()
