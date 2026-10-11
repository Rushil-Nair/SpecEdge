"""Add multi-step tool conversations to the fine-tuning file (second run of part 2).

    python training/prepare_multistep.py --base-train /workspace/data/train_v1/train.jsonl --out /workspace/data/train_v2

Why: the agent-task runs showed that the stock 3B keeps calling tools after an agent task is finished
(0 of 10 multi-step tasks). The first training file is single-step: every example ends at the
first reply. This adds conversations in which a tool result comes back and the assistant then
either makes the next call or, mostly, answers in plain text and stops.

Sources (both Apache 2.0, checked on the Hub on 2026-10-08)
  Team-ACE/ToolACE                       every conversation that contains a tool result
  glaiveai/glaive-function-calling-v2    3,000 conversations that contain a tool result, fixed seed

Conversion. Each conversation is rewritten into Llama 3.2's own chat format through the base
model's chat template: the tool list in the first user turn, an assistant call as
{"name": ..., "parameters": ...}, the tool's result as an "ipython" turn. One training example
is written for every assistant turn: the prompt is the conversation up to that turn, the
completion is that turn. Llama 3.2's template allows one call per assistant turn, so a
conversation is cut just before the first turn that makes several calls at once or that
cannot be parsed.

The same overlap check as the first file: a conversation is left out when it lists a
tool whose name any frozen test task uses, or when one of its user messages is a test
question. The output is the first file's 20,000 examples, unchanged, plus the new ones.
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import hashlib
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

from prepare import TEST, ROOT, json_type, question_key

SEED, GLAIVE_CONVERSATIONS = 2027, 3000


def schema(parameters: dict | None) -> dict:
    """A tool's parameters as JSON schema with real type names."""
    parameters = parameters or {}
    properties = {}
    for name, prop in (parameters.get("properties") or {}).items():
        properties[name] = {**prop, "type": json_type(prop.get("type", "string"))} if isinstance(prop, dict) else {"type": "string"}
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    return {"type": "object", "properties": properties, "required": required}


def tool_of(spec: dict) -> dict:
    return {"type": "function", "function": {"name": spec["name"], "description": spec.get("description", ""),
                                             "parameters": schema(spec.get("parameters"))}}


def call_message(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def toolace_calls(text: str) -> list[tuple[str, dict]] | None:
    """Parse ToolACE's  [Name One(a="x", b=2), Other()]  into (name, arguments) pairs."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    body, calls, depth, start, quote = text[1:-1], [], 0, 0, None
    pieces = []
    for index, char in enumerate(body):                       # split on commas outside brackets and strings
        if quote:
            if char == quote and body[index - 1] != "\\":
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(body[start:index])
            start = index + 1
    pieces.append(body[start:])
    for piece in pieces:
        match = re.match(r"^\s*(.+?)\((.*)\)\s*$", piece, re.S)
        if not match:
            return None
        try:
            node = ast.parse(f"f({match[2]})", mode="eval").body
            if node.args:
                return None
            arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
        except (SyntaxError, ValueError, TypeError):
            return None
        calls.append((match[1].strip(), arguments))
    return calls or None


def toolace_conversations(path: str):
    for index, row in enumerate(json.loads(Path(path).read_text(encoding="utf-8"))):
        if not any(m["from"] == "tool" for m in row["conversations"]):
            continue
        start = row["system"].find("[{")
        try:
            specs, _ = json.JSONDecoder().raw_decode(row["system"][start:]) if start >= 0 else (None, 0)
            tools = [tool_of(spec) for spec in specs]
        except (ValueError, TypeError, KeyError):
            yield f"toolace-{index}", None, None
            continue
        messages, pending = [], None
        for message in row["conversations"]:
            role, value = message["from"], message["value"]
            if role == "user":
                messages.append({"role": "user", "content": value})
            elif role == "assistant":
                calls = toolace_calls(value) if value.lstrip().startswith("[") else None
                if value.lstrip().startswith("[") and (calls is None or len(calls) != 1):
                    break                                      # several calls at once, or not parseable: stop here
                if calls:
                    messages.append(call_message(*calls[0]))
                    pending = calls[0][0]
                else:
                    messages.append({"role": "assistant", "content": value})
            elif role == "tool":
                if pending is None:
                    break
                try:
                    results = json.loads(value)
                    content = json.dumps(results[0].get("results", results[0]), ensure_ascii=False) if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict) else value
                except ValueError:
                    content = value
                messages.append({"role": "tool", "content": content})
                pending = None
        yield f"toolace-{index}", tools, messages


def glaive_conversations(path: str):
    decoder = json.JSONDecoder()
    for index, row in enumerate(json.loads(Path(path).read_text(encoding="utf-8"))):
        if "FUNCTION RESPONSE:" not in row["chat"]:
            continue
        tools, position, text = [], 0, row["system"]
        try:
            while (position := text.find("{", position)) >= 0:
                spec, position = decoder.raw_decode(text, position)
                tools.append(tool_of(spec))
        except (ValueError, KeyError, TypeError):
            yield f"glaive-{index}", None, None
            continue
        parts = re.split(r"\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)", row["chat"])
        messages, pending = [], False
        for marker, body in zip(parts[1::2], parts[2::2]):
            body = body.replace("<|endoftext|>", "").strip()
            if marker == "USER:":
                messages.append({"role": "user", "content": body})
            elif marker == "ASSISTANT:":
                if "<functioncall>" in body:
                    match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\s*\}\s*$', body, re.S)
                    try:
                        arguments = json.loads(match[2])
                        if not isinstance(arguments, dict):
                            raise ValueError
                    except (ValueError, TypeError):
                        break
                    messages.append(call_message(match[1], arguments))
                    pending = True
                else:
                    messages.append({"role": "assistant", "content": body})
            else:
                if not pending:
                    break
                messages.append({"role": "tool", "content": body})
                pending = False
        yield f"glaive-{index}", tools, messages


def trimmed(messages: list[dict]) -> list[dict]:
    """Cut to the last assistant turn, and drop a call that never got its result."""
    while messages and (messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls")):
        messages = messages[:-1]
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidates", default="/workspace/data/candidates")
    parser.add_argument("--base", default="/workspace/models/llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--layouts", choices=["user", "both"], default="user",
                        help="both: each added example gets the tool list in the user or the system message at random")
    args = parser.parse_args()
    layout_rng = random.Random(SEED + 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    test = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    test_tools = {tool["function"]["name"] for task in test for tool in task["tools"]}
    test_questions = {question_key(task["messages"][0]["content"]) for task in test}

    def overlap(tools: list[dict], messages: list[dict]) -> str | None:
        if {t["function"]["name"] for t in tools} & test_tools:
            return "tool"
        if any(m["role"] == "user" and question_key(m["content"]) in test_questions for m in messages):
            return "question"
        return None

    check = {"test_file": str(TEST.relative_to(ROOT)), "test_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(), "sources": {}}
    kept: dict[str, list] = {}
    for name, conversations in (("ToolACE", toolace_conversations(f"{args.candidates}/ToolACE/data.json")),
                                ("glaive-function-calling-v2", glaive_conversations(f"{args.candidates}/glaive/glaive-function-calling-v2.json"))):
        counts, usable = collections.Counter(), []
        for source, tools, messages in conversations:
            if tools is None:
                counts["tool_list_not_parseable"] += 1
                continue
            messages = trimmed(messages)
            if not any(m["role"] == "tool" for m in messages):
                counts["no_tool_result_left_after_cutting"] += 1
                continue
            reason = overlap(tools, messages)
            counts[f"left_out_shared_{reason}" if reason else "usable"] += 1
            if not reason:
                usable.append((source, tools, messages))
        if name.startswith("glaive"):
            usable = rng.sample(usable, min(GLAIVE_CONVERSATIONS, len(usable)))
        kept[name] = usable
        check["sources"][name] = {"conversations_with_a_tool_result": sum(counts.values()), **counts, "conversations_used": len(usable)}

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    start, examples, kinds, failed = datetime.date(2024, 1, 1), [], collections.Counter(), 0
    layouts = collections.Counter()
    for name, usable in kept.items():
        for source, tools, messages in usable:
            date = (start + datetime.timedelta(days=rng.randrange(1100))).strftime("%d %b %Y")
            for index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                after_result = index > 0 and messages[index - 1]["role"] == "tool"
                if message.get("tool_calls"):
                    function = message["tool_calls"][0]["function"]
                    completion = json.dumps({"name": function["name"], "parameters": function["arguments"]}, ensure_ascii=False)
                    kind = "call_after_tool_result" if after_result else "call"
                else:
                    completion = message["content"]
                    kind = "text_after_tool_result" if after_result else "text"
                layout = layout_rng.choice(["user", "system"]) if args.layouts == "both" else "user"
                try:
                    prompt = tokenizer.apply_chat_template(messages[:index], tools=tools, tokenize=False,
                                                           add_generation_prompt=True, date_string=date,
                                                           tools_in_user_message=(layout == "user"))
                except Exception:  # noqa: BLE001 - the template refuses a few shapes; count them
                    failed += 1
                    continue
                kinds[(name, kind)] += 1
                layouts[layout] += 1
                examples.append({"prompt": prompt, "completion": completion + "<|eot_id|>", "kind": kind,
                                 "source": f"{source}-turn{index}", "layout": layout})

    first = [line for line in Path(args.base_train).read_text(encoding="utf-8").splitlines() if line]
    lines = first + [json.dumps(e, ensure_ascii=False) for e in examples]
    random.Random(SEED).shuffle(lines)
    (out / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # Second pass on what was kept: nothing may share a tool or a question with the test set.
    leaked = sum(bool(overlap(tools, messages)) for usable in kept.values() for _, tools, messages in usable)
    check.update(first_file=args.base_train, first_file_examples=len(first), examples_added=len(examples),
                 examples_total=len(lines), template_refused=failed,
                 added_conversations_overlapping_the_test_set=leaked)
    (out / "overlap_check.json").write_text(json.dumps(check, indent=1))
    first_layouts = collections.Counter(json.loads(line).get("layout", "user") for line in first)
    stats = {"seed": SEED, "examples_total": len(lines), "examples_from_first_file": len(first),
             "layout_of_added_examples": dict(layouts), "layout_of_first_file_examples": dict(first_layouts),
             "added_by_source_and_kind": {f"{n} / {k}": v for (n, k), v in sorted(kinds.items())}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(check, indent=1))
    print(json.dumps(stats, indent=1))
    sample = next(e for e in examples if e["kind"] == "text_after_tool_result")
    print("SAMPLE PROMPT TAIL:", repr(sample["prompt"][-700:]))
    print("SAMPLE COMPLETION:", repr(sample["completion"][:300]))
    if leaked:
        raise SystemExit("added conversations overlap the frozen test set")


if __name__ == "__main__":
    main()
