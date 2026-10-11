"""Score a model on a tool-call task set. Runs on the rented machine.

    python bench/toolcall_eval.py --name base_3b --model <gguf> --tasks data/toolset/v1/toolset_v1.jsonl

Starts llama-server with the model, sends each task with its tool list (greedy decoding),
and scores what comes back. Two ways of reading the reply:

  --mode raw   (default) the prompt is rendered with the model's own chat template, the raw
               generated text is returned, and this script finds the tool calls in it. Every
               JSON object that names a function and has "parameters"/"arguments" counts as a
               call, so several calls in one reply are read whatever separates them. The name
               may be written the way Llama 3.2 documents it ({"name": ...}) or the two other
               ways the 1B uses ({"function": "<name>", ...} and {"function": {"name": ...}});
               official_format records whether a reply used only the documented way.
  --mode chat  llama-server's chat endpoint parses the reply itself. At the pinned commit it
               rejects Llama 3.2's way of writing several calls ("{...}; {...}") with an HTTP
               500, so every multi-call task fails there regardless of what the model wrote.

Both are kept because they answer different questions: raw mode measures the model, chat
mode measures the model through the stock server.

Scores, each a yes/no per task:
  valid            the reply could be read: plain text, or tool calls whose arguments are valid JSON
  right_decision   called a tool when it should, did not when it should not
  right_functions  the set of function names matches the expected set (call tasks only)
  fully_correct    for call tasks: same functions with the same arguments;
                   for no-call tasks: no tool call was made

Argument comparison normalises what should not count as a difference: "5" and 5, 5 and
5.0, "true" and true, a list or object written as a JSON string ("[1, 2]" and [1, 2]),
surrounding whitespace, and the order of calls. A predicted argument that the task
does not expect is tolerated only when it equals the tool's declared default. Anything
else must match exactly. An expected call may instead carry "alternatives" (the BFCL form
used by the When2Call call tasks): a list of acceptable values per argument, where "" means
the argument may be left out. The raw reply is saved for every task so the scoring can be
checked by hand.

    python bench/toolcall_eval.py --rescore results/part2/base_3b_f16_v1.json --tasks <tasks>

re-scores the saved replies of an earlier raw-mode run with the current scorer, without a
model, and writes <name>_rescored.json beside it. Tasks whose request failed stay failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import first_hour as fh


JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}


def sendable(schema):
    """A tool list llama-server will accept. It refuses a schema whose "type" is not a JSON
    schema type (HTTP 400), and two tasks of frozen set v1 say "any". The frozen file is not
    changed; an unknown type name is sent as "string", which is also what training/prepare.py
    does with unknown names. Applied to every task and every model alike."""
    if isinstance(schema, list):
        return [sendable(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {key: ("string" if key == "type" and isinstance(value, str) and value not in JSON_SCHEMA_TYPES and value != "function"
                  else sendable(value)) for key, value in schema.items()}


def normalise(value):
    """Make values comparable across harmless formatting differences."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
            if math.isfinite(number):
                return number
        except ValueError:
            pass
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        if text[:1] in ("[", "{"):               # a list or object written as a string
            try:
                return normalise(json.loads(text))
            except ValueError:
                pass
        return text
    if isinstance(value, list):
        return [normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in sorted(value.items())}
    return value


def defaults_of(tools: list[dict], name: str) -> dict:
    for tool in tools:
        function = tool["function"]
        if function["name"] == name:
            properties = function.get("parameters", {}).get("properties", {}) or {}
            return {k: normalise(v["default"]) for k, v in properties.items() if isinstance(v, dict) and "default" in v}
    return {}


def arguments_match(predicted: dict, expected: dict, defaults: dict) -> bool:
    predicted, expected = normalise(predicted), normalise(expected)
    for key, value in expected.items():
        if predicted.get(key) != value:
            return False
    # extras are fine only when they restate the tool's own default
    return all(key in expected or defaults.get(key) == value for key, value in predicted.items())


def alternatives_match(predicted: dict, alternatives: dict, defaults: dict) -> bool:
    """BFCL-style expectation: each argument lists its acceptable values; "" in the list means
    the argument may be left out. Used for the When2Call call tasks."""
    predicted = normalise(predicted)
    for key, accepted in alternatives.items():
        accepted = accepted if isinstance(accepted, list) else [accepted]
        if key not in predicted:
            if "" in accepted:
                continue
            return False
        if not any(predicted[key] == normalise(value) for value in accepted):
            return False
    return all(key in alternatives or defaults.get(key) == value for key, value in predicted.items())


def call_of(value) -> tuple[dict, bool] | None:
    """Read one JSON object as a tool call, or None. The flag says whether it is in the
    format Llama 3.2 documents: {"name": ..., "parameters": ...}."""
    if not isinstance(value, dict):
        return None
    name, holder, official = value.get("name"), value, True
    if not isinstance(name, str):
        function, official = value.get("function"), False
        if isinstance(function, str):                                   # {"function": "<name>", "parameters": ...}
            name = function
        elif isinstance(function, dict) and isinstance(function.get("name"), str):
            name = function["name"]                                     # {"function": {"name": ..., "parameters": ...}}
            if "parameters" in function or "arguments" in function:
                holder = function
        else:
            return None
    if "parameters" not in holder and "arguments" not in holder:
        return None
    arguments = holder.get("parameters", holder.get("arguments"))
    return {"function": {"name": name, "arguments": arguments if arguments is not None else {}}}, official


def calls_in_text(text: str) -> tuple[list[dict], bool, bool]:
    """Find tool calls written as JSON objects anywhere in the raw text.

    Returns the calls, whether the text was readable, and whether every call was in the
    documented format. The text is unreadable when it starts a JSON object mentioning
    "name" or "function" that never becomes valid JSON (a broken call).
    """
    decoder, calls, position, broken, official = json.JSONDecoder(), [], 0, False, True
    while True:
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            if '"name"' in text[start:start + 200] or '"function"' in text[start:start + 200]:
                broken = True
            position = start + 1
            continue
        position = end
        found = call_of(value)
        if found:
            calls.append(found[0])
            official = official and found[1]
    return calls, not broken, official


def score(task: dict, message: dict) -> dict:
    raw_calls = message.get("tool_calls") or []
    calls, valid = [], True
    for call in raw_calls:
        function = call.get("function", {})
        arguments = function.get("arguments")
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            if not isinstance(arguments, dict):
                raise ValueError
        except (ValueError, TypeError):
            valid, arguments = False, None
        calls.append({"name": function.get("name"), "arguments": arguments})

    expect = task["expect"]
    if expect["kind"] == "no_call":
        return {"valid": valid, "right_decision": not calls, "right_functions": None, "fully_correct": not calls}

    expected_calls = expect["calls"]
    right_decision = bool(calls)
    right_functions = Counter(c["name"] for c in calls) == Counter(c["name"] for c in expected_calls)
    fully = False
    if valid and right_functions:
        remaining = list(calls)
        fully = True
        for wanted in expected_calls:            # match each expected call to one predicted call
            defaults = defaults_of(task["tools"], wanted["name"])
            hit = next((c for c in remaining if c["name"] == wanted["name"] and (
                alternatives_match(c["arguments"], wanted["alternatives"], defaults) if "alternatives" in wanted
                else arguments_match(c["arguments"], wanted.get("arguments", {}), defaults))), None)
            if hit is None:
                fully = False
                break
            remaining.remove(hit)
    return {"valid": valid, "right_decision": right_decision, "right_functions": right_functions, "fully_correct": fully}


def score_text(task: dict, text: str | None, error: str | None) -> dict:
    """Score one raw reply. A failed request or an unreadable reply is wrong for every kind
    of task; without this a failed request would count as "correctly made no call"."""
    found, readable, official = calls_in_text(text) if text is not None else ([], True, True)
    result = score(task, {"content": text, "tool_calls": found})
    if error or not readable:
        result.update(valid=False, right_decision=False, fully_correct=False)
    return {"tool_calls": found, "official_format": official, **result,
            "fully_correct_official_format": result["fully_correct"] and official}


GROUPS = {"all": lambda row: True,
          "call tasks": lambda row: row["expect"]["kind"] == "calls",
          "no-call: ask for information": lambda row: row["expect"].get("category") == "request_for_info",
          "no-call: no tool fits": lambda row: row["expect"].get("category") == "cannot_answer",
          # finer groups, present only when the task set carries sources and tags (toolset v1 on)
          "call tasks from xLAM": lambda row: row["expect"]["kind"] == "calls" and row["source"].startswith("xlam"),
          "call tasks from When2Call": lambda row: row["expect"]["kind"] == "calls" and row["source"].startswith("When2Call"),
          "call tasks with several calls": lambda row: "multi_call" in row.get("tags", []),
          "call tasks with a decimal argument": lambda row: "decimal" in row.get("tags", [])}
KEYS = ("valid", "right_decision", "right_functions", "fully_correct", "fully_correct_official_format",
        "at_limit", "fully_correct_limit_rule")


def limit_fields(reply: dict, max_tokens: int) -> dict:
    """From a llama-server /completion reply: how long the answer was and whether it was cut
    off by the token limit."""
    return {"tokens_out": reply.get("tokens_predicted"), "stop_type": reply.get("stop_type"),
            "at_limit": reply.get("stop_type") == "limit" or (reply.get("tokens_predicted") or 0) >= max_tokens}


def with_limit_rule(row: dict) -> dict:
    """Rule of 2026-10-08: a reply that ran to the token limit is wrong, whatever it
    contains. "No call was made" is otherwise correct for a should-not-call task, so junk and
    answers that never finish were counted as correct refusals. Reported beside the plain
    score; None when the run did not record how the reply ended."""
    row["fully_correct_limit_rule"] = None if row.get("at_limit") is None else bool(row["fully_correct"]) and not row["at_limit"]
    return row


def summarise(rows: list[dict]) -> dict:
    def rate(subset: list[dict], key: str) -> float | None:
        values = [r[key] for r in subset if r.get(key) is not None]
        return round(100 * sum(values) / len(values), 1) if values else None

    summary = {}
    for name, keep in GROUPS.items():
        subset = [r for r in rows if keep(r)]
        if not subset:
            continue
        summary[name] = {"tasks": len(subset), "request_failed": sum(bool(r["error"]) for r in subset),
                         **{k: rate(subset, k) for k in KEYS}}
    return summary


def show(summary: dict) -> None:
    for name, values in summary.items():
        print(f"{name:30s} n={values['tasks']:3d} failed {values['request_failed']}  valid {values['valid']}  "
              f"decision {values['right_decision']}  functions {values['right_functions']}  "
              f"fully correct {values['fully_correct']}  (documented format only {values['fully_correct_official_format']})")


def rescore(path: Path, tasks: dict[str, dict]) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["mode"] != "raw":
        raise SystemExit("only raw-mode files keep the text needed for re-scoring")
    rows = [{**row, "tags": tasks[row["id"]].get("tags", []), **score_text(tasks[row["id"]], row["content"], row["error"])}
            for row in record["rows"]]
    record.update(name=record["name"] + "_rescored", rescored_from=path.name, rescored_unix=time.time(),
                  summary_before=record["summary"], summary=summarise(rows), rows=rows)
    path.with_name(record["name"] + ".json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    show(record["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name")
    parser.add_argument("--model")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--rescore", help="re-score the saved replies in this raw-mode result file; no model is run")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--tools-in", choices=["user", "system"], default="user",
                        help="where Llama 3.2's template puts the tool list: the first user message (its "
                             "default) or the system message (tools_in_user_message = false)")
    parser.add_argument("--out-dir", default=str(fh.ROOT / "results" / "runs"))
    args = parser.parse_args()

    frozen = Path(args.tasks).with_name("FROZEN.json")
    if frozen.exists():          # a frozen set must be the file that was frozen
        expected = json.loads(frozen.read_text(encoding="utf-8"))["sha256"]
        if hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"{args.tasks} does not match the checksum in {frozen}")
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines()]
    if args.rescore:
        rescore(Path(args.rescore), {task["id"]: task for task in tasks})
        return
    if not (args.name and args.model):
        parser.error("--name and --model are required unless --rescore is given")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_args = argparse.Namespace(context=8192, slots=1, cpu=False, threads=None, head=None, draft_max=0,
                                     draft_p_min=0.0, target=args.model, spec_type="draft-eagle3",
                                     server_bin=None, extra="--jinja")
    server = fh.start_server(server_args, out_dir / f"{args.name}.server.log")
    rows = []
    try:
        for task in tasks:
            text, ended = None, {}
            try:
                if args.mode == "chat":
                    reply = fh.post("/v1/chat/completions", {
                        "messages": task["messages"], "tools": sendable(task["tools"]),
                        "temperature": 0.0, "top_k": 1, "seed": 0, "max_tokens": args.max_tokens,
                        "cache_prompt": False})
                    message = reply["choices"][0]["message"]
                else:
                    prompt = fh.post("/apply-template", {"messages": task["messages"], "tools": sendable(task["tools"]),
                                                         "chat_template_kwargs": {
                                                             "enable_thinking": False,
                                                             "tools_in_user_message": args.tools_in == "user"}})["prompt"]
                    reply = fh.post("/completion", {"prompt": prompt, "n_predict": args.max_tokens,
                                                    "temperature": 0.0, "top_k": 1, "seed": 0,
                                                    "cache_prompt": False})
                    text, ended = reply["content"], limit_fields(reply, args.max_tokens)
                    message = {"content": text}
                error = None
            except Exception as err:  # noqa: BLE001 - a server-side parse failure is itself a result
                message, error = {}, f"{type(err).__name__}: {str(err)[:200]}"
            if args.mode == "chat":
                result = {"tool_calls": message.get("tool_calls"), "official_format": None, **score(task, message),
                          "fully_correct_official_format": None}
                if error:
                    # Without this, a failed request would count as "correctly made no call".
                    result.update(valid=False, right_decision=False, fully_correct=False)
            else:
                result = score_text(task, text, error)
            rows.append(with_limit_rule({"id": task["id"], "source": task["source"], "tags": task.get("tags", []),
                                         "expect": task["expect"], "content": message.get("content"), "error": error,
                                         **ended, **result}))
    finally:
        server.terminate()
        server.wait(timeout=60)

    summary = summarise(rows)
    record = {"name": args.name, "mode": args.mode, "tools_in": args.tools_in, "model": Path(args.model).name, "tasks_file": args.tasks,
              "max_tokens": args.max_tokens, "decoding": "greedy",
              "llama_cpp_commit": Path("/workspace/llama_commit.txt").read_text().split()[0],
              "cpus_visible": os.cpu_count(), "finished_unix": time.time(), "summary": summary, "rows": rows}
    (out_dir / f"{args.name}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    show(summary)


if __name__ == "__main__":
    main()
