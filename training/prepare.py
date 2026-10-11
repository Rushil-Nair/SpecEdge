"""Build the fine-tuning file for part 2, and check it against the frozen test set.

    python training/prepare.py --out /workspace/data/train_v1

Runs on the rented machine, on the data downloaded there with the owner's Hugging Face login.

Sources
  xLAM function-calling 60k     the answer is one or more calls
  When2Call train_sft           the answer is a question back to the user or a refusal
                                (rows whose answer is a call are used as call examples too)

Format: the prompt is the base model's own chat template with the tool list, exactly what the
test harness sends (bench/toolcall_eval.py renders the same template through llama-server).
A call is written the way Llama 3.2 documents and the base model already writes it:
    {"name": "f", "parameters": {...}}            several calls joined by "; "
with numbers, lists and booleans as real JSON values, never quoted strings. An earlier check
found the base model loses decimals when it quotes numbers; the training answers do not.

No test data in training. A training example is left out when
  - it lists a tool whose name is used by any task of the frozen test set (xLAM and When2Call
    tools both), or
  - its question is the same as a test question after lower-casing and removing punctuation.
The numbers are written to <out>/overlap_check.json, and the check is then run again on the
finished file: the script fails if anything overlaps.

Layouts. Llama 3.2's template can put the tool list in the first user message (its default)
or in the system message (tools_in_user_message = false). The agent-task runs found the stock model
keeps calling tools in the first layout and stops in the second. With --layouts both, each
example is rendered in one of the two, chosen at random with its own seed, so which examples
are chosen does not change. The first run used the default layout only.

Output: <out>/train.jsonl ({"prompt", "completion", "kind", "source", "layout"}), <out>/stats.json.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import hashlib
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "toolset" / "v1" / "toolset_v1.jsonl"
SEED = 2026
XLAM_EXAMPLES, WHEN2CALL_EXAMPLES = 14000, 6000
JSON_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
              "number": "number", "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array",
              "dict": "object", "object": "object", "tuple": "array", "set": "array"}


def json_type(raw) -> str:
    return JSON_TYPES.get(str(raw).split(",")[0].strip().lower().split("[")[0], "string")


def xlam_tool(tool: dict) -> dict:
    properties, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        raw = str(spec.get("type", "str"))
        properties[name] = {"type": json_type(raw), "description": str(spec.get("description", ""))}
        if spec.get("default") not in (None, ""):
            properties[name]["default"] = spec["default"]
        if "optional" not in raw.lower() and spec.get("default") in (None, ""):
            required.append(name)
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def when2call_tool(raw: str) -> dict:
    spec = json.loads(raw)
    parameters = spec.get("parameters") or {}
    properties = {}
    for name, prop in (parameters.get("properties") or {}).items():
        properties[name] = {**prop, "type": json_type(prop.get("type", "str"))} if isinstance(prop, dict) else {"type": "string"}
        if properties[name].get("default") in (None, ""):
            properties[name].pop("default", None)
    return {"type": "function", "function": {"name": spec["name"], "description": spec.get("description", ""),
                                             "parameters": {"type": "object", "properties": properties,
                                                            "required": parameters.get("required", [])}}}


def question_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def calls_text(calls: list[dict]) -> str:
    return "; ".join(json.dumps({"name": c["name"], "parameters": c.get("arguments", c.get("parameters", {}))},
                                ensure_ascii=False) for c in calls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="/workspace/data")
    parser.add_argument("--base", default="/workspace/models/llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--layouts", choices=["user", "both"], default="user")
    args = parser.parse_args()
    layout_rng = random.Random(SEED + 1)        # separate, so the choice of examples stays the same
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    test = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    test_tools = {tool["function"]["name"] for task in test for tool in task["tools"]}
    test_questions = {question_key(task["messages"][0]["content"]) for task in test}

    def overlaps(tool_names, question: str) -> str | None:
        if set(tool_names) & test_tools:
            return "tool"
        if question_key(question) in test_questions:
            return "question"
        return None

    check = {"test_file": str(TEST.relative_to(ROOT)), "test_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(),
             "test_tasks": len(test), "test_tool_names": len(test_tools), "sources": {}}

    xlam_path = glob.glob(f"{args.data}/xlam/**/xlam_function_calling_60k.json", recursive=True)[0]
    xlam_rows = json.loads(Path(xlam_path).read_text(encoding="utf-8"))
    counts, xlam = collections.Counter(), []
    for row in xlam_rows:
        tools = json.loads(row["tools"])
        reason = overlaps([t["name"] for t in tools], row["query"])
        counts[reason or "kept"] += 1
        if not reason:
            xlam.append({"messages": [{"role": "user", "content": row["query"]}], "tools": [xlam_tool(t) for t in tools],
                         "completion": calls_text(json.loads(row["answers"])), "kind": "call",
                         "source": f"xlam-{row['id']}"})
    check["sources"]["xlam-function-calling-60k"] = {"rows": len(xlam_rows), "left_out_shared_tool": counts["tool"],
                                                     "left_out_same_question": counts["question"], "usable": counts["kept"]}

    w2c_path = glob.glob(f"{args.data}/when2call/**/when2call_train_sft.jsonl", recursive=True)[0]
    counts, when2call = collections.Counter(), []
    for index, line in enumerate(open(w2c_path, encoding="utf-8")):
        row = json.loads(line)
        tools = [when2call_tool(raw) for raw in row["tools"]]
        user, answer = row["messages"][0], row["messages"][-1]
        reason = overlaps([t["function"]["name"] for t in tools], user["content"])
        if len(row["messages"]) != 2 or user["role"] != "user" or answer["role"] != "assistant":
            reason = "unexpected_shape"
        counts[reason or "kept"] += 1
        if reason:
            continue
        if answer.get("tool_calls"):
            calls = [c.get("function", c) for c in answer["tool_calls"]]
            calls = [{"name": c["name"], "arguments": json.loads(c["arguments"]) if isinstance(c.get("arguments"), str)
                      else c.get("arguments", {})} for c in calls]
            completion, kind = calls_text(calls), "call"
        else:
            completion, kind = answer["content"].strip(), "no_call"
        when2call.append({"messages": [user], "tools": tools, "completion": completion, "kind": kind,
                          "source": f"when2call-train_sft-{index}"})
    check["sources"]["When2Call train_sft"] = {
        "rows": sum(counts.values()), "left_out_shared_tool": counts["tool"], "left_out_same_question": counts["question"],
        "left_out_unexpected_shape": counts["unexpected_shape"], "usable": counts["kept"],
        "usable_by_kind": dict(collections.Counter(r["kind"] for r in when2call))}

    chosen = rng.sample(xlam, min(XLAM_EXAMPLES, len(xlam))) + rng.sample(when2call, min(WHEN2CALL_EXAMPLES, len(when2call)))
    rng.shuffle(chosen)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    start = datetime.date(2024, 1, 1)
    with open(out / "train.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for example in chosen:
            # The template prints today's date; llama-server fills in the real one at test time.
            # A different date per example keeps the model from tying its behaviour to one date.
            date = (start + datetime.timedelta(days=rng.randrange(1100))).strftime("%d %b %Y")
            layout = layout_rng.choice(["user", "system"]) if args.layouts == "both" else "user"
            example["layout"] = layout
            prompt = tokenizer.apply_chat_template(example["messages"], tools=example["tools"], tokenize=False,
                                                   add_generation_prompt=True, date_string=date,
                                                   tools_in_user_message=(layout == "user"))
            handle.write(json.dumps({"prompt": prompt, "completion": example["completion"] + "<|eot_id|>",
                                     "kind": example["kind"], "source": example["source"], "layout": layout},
                                    ensure_ascii=False) + "\n")

    # Second pass on the finished file: nothing in it may share a tool or a question with the test set.
    leaked = 0
    by_source = {e["source"]: e for e in chosen}
    for line in open(out / "train.jsonl", encoding="utf-8"):
        example = by_source[json.loads(line)["source"]]
        leaked += bool(overlaps([t["function"]["name"] for t in example["tools"]], example["messages"][0]["content"]))
    check["examples_written"] = len(chosen)
    check["examples_overlapping_the_test_set"] = leaked
    (out / "overlap_check.json").write_text(json.dumps(check, indent=1))
    stats = {"seed": SEED, "examples": len(chosen), "by_layout": dict(collections.Counter(e["layout"] for e in chosen)),
             "by_kind": dict(collections.Counter(e["kind"] for e in chosen)),
             "by_source": dict(collections.Counter(e["source"].split("-")[0] for e in chosen)),
             "calls_per_call_example": dict(collections.Counter(e["completion"].count("; {\"name\"") + 1
                                                               for e in chosen if e["kind"] == "call"))}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(check, indent=1))
    print(json.dumps(stats, indent=1))
    if leaked:
        raise SystemExit("training file overlaps the frozen test set")


if __name__ == "__main__":
    main()
