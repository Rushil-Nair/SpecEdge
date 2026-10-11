"""The scorer rule "a reply that runs to the token limit counts as wrong", applied to every row.

    python bench/limit_rule.py candidates            (laptop)   which saved replies could be at the limit
    python bench/limit_rule.py recheck               (machine)  ask those tasks again and read why the reply ended
    python bench/limit_rule.py report                (laptop)   every row under the old and the new rule

Why: the scorer counts "no call was made" as correct for a should-not-call task, so a reply
that is junk, or simply never finishes, is counted as a correct refusal. The rule of
2026-10-08: a reply that ran to the token limit (384) is wrong, whatever it contains.

Rows from Qualcomm's engine (results/part12) carry `tokens_out`, so the rule is applied
directly. Rows from llama.cpp (results/part2, results/part3) saved only the text. A reply of
384 tokens has at least 384 characters, so only replies that long can be at the limit;
`recheck` asks exactly those again with the same file and request and records the server's
own `stop_type` and token count. Decoding is greedy, and `recheck` also records whether the
reply came back the same as the saved one.

Files: results/limit_rule/candidates.json, recheck.json, report.json, report.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "limit_rule"
TASKS = ROOT / "data" / "toolset" / "v1" / "toolset_v1.jsonl"
GGUF = Path("/workspace/models/gguf")
LIMIT = 384
TYPES = [("all", None), ("call", "call"), ("ask", "ask"), ("no_tool", "no_tool")]


def llama_cpp_files() -> list[Path]:
    return sorted((ROOT / "results" / "part2").glob("*_v1.json")) + sorted((ROOT / "results" / "part3").glob("*_accuracy.json"))


def settings(record: dict) -> dict:
    """How the rows of a llama.cpp result file were asked."""
    if record.get("what") == "accuracy":                          # quant/run.py
        return {"gguf": record["file"], "tools_in": "user", "forced": record["forced_json"]}
    return {"gguf": record["model"], "tools_in": record.get("tools_in", "user"), "forced": False}


def candidates() -> None:
    found = []
    for path in llama_cpp_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        ids = [row["id"] for row in record["rows"] if len(row["content"] or "") >= LIMIT]
        found.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), **settings(record), "rows": len(record["rows"]), "ids": ids})
        print(f"{path.name}: {len(ids)} of {len(record['rows'])} replies have {LIMIT} characters or more")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(json.dumps(found, indent=1), encoding="utf-8")


def recheck() -> None:
    sys.path.insert(0, str(ROOT / "bench"))
    import first_hour as fh
    import forced_json
    import toolcall_eval as te

    tasks = {task["id"]: task for task in map(json.loads, TASKS.read_text(encoding="utf-8").splitlines())}
    wanted = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    done_path = OUT / "recheck.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    for entry in wanted:
        if entry["file"] in done or not entry["ids"]:
            done.setdefault(entry["file"], {})
            continue
        saved = {row["id"]: row["content"] for row in json.loads((ROOT / entry["file"]).read_text(encoding="utf-8"))["rows"]}
        server_args = argparse.Namespace(context=8192, slots=1, cpu=False, threads=None, head=None, draft_max=0,
                                         draft_p_min=0.0, target=str(GGUF / entry["gguf"]), spec_type="draft-eagle3",
                                         server_bin=None, extra="--jinja")
        server = fh.start_server(server_args, OUT / "recheck.server.log")
        rows = {}
        try:
            for task_id in entry["ids"]:
                task = tasks[task_id]
                prompt = fh.post("/apply-template", {"messages": task["messages"], "tools": te.sendable(task["tools"]),
                                                     "chat_template_kwargs": {"enable_thinking": False,
                                                                              "tools_in_user_message": entry["tools_in"] == "user"}})["prompt"]
                body = {"prompt": prompt, "n_predict": LIMIT, "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False}
                if entry["forced"]:
                    body["grammar"] = forced_json.grammar_for(task["tools"])
                reply = fh.post("/completion", body)
                rows[task_id] = {"tokens_out": reply["tokens_predicted"], "stop_type": reply["stop_type"],
                                 "same_reply_as_saved": reply["content"] == saved[task_id]}
        finally:
            server.terminate()
            server.wait(timeout=60)
        done[entry["file"]] = rows
        done_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        at_limit = sum(r["stop_type"] == "limit" for r in rows.values())
        print(f"{entry['file']}: {len(rows)} asked again, {at_limit} ended at the limit, "
              f"{sum(r['same_reply_as_saved'] for r in rows.values())} came back the same", flush=True)
    done_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print("limit_rule recheck DONE", flush=True)


def kind_of(row: dict, tasks: dict) -> str:
    return row.get("type") or tasks[row["id"]]["type"]


def both_rules(rows: list[dict], at_limit: set[str], tasks: dict) -> dict:
    out = {}
    for label, kind in TYPES:
        part = [row for row in rows if kind is None or kind_of(row, tasks) == kind]
        old = sum(bool(row["fully_correct"]) for row in part)
        lost = sum(bool(row["fully_correct"]) and row["id"] in at_limit for row in part)
        out[label] = {"tasks": len(part), "old_correct": old, "new_correct": old - lost,
                      "old_percent": round(100 * old / len(part), 1), "new_percent": round(100 * (old - lost) / len(part), 1),
                      "at_limit": sum(row["id"] in at_limit for row in part)}
    return out


def report() -> None:
    tasks = {task["id"]: task for task in map(json.loads, TASKS.read_text(encoding="utf-8").splitlines())}
    rechecked = json.loads((OUT / "recheck.json").read_text(encoding="utf-8"))
    wanted = {entry["file"]: entry for entry in json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))}
    subset = {task["id"] for task in random.Random(5).sample(list(tasks.values()), 200)}
    table, notes = [], []
    for path in llama_cpp_files():
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        record = json.loads(path.read_text(encoding="utf-8"))
        if all(row.get("at_limit") is not None for row in record["rows"]):
            # written by the current scorer: how each reply ended is in the row itself
            checked, at_limit = {}, {row["id"] for row in record["rows"] if row["at_limit"]}
        else:
            if key not in wanted:
                raise SystemExit(f"{key}: an older file that is not in candidates.json; run candidates and recheck first")
            missing = set(wanted[key]["ids"]) - set(rechecked.get(key, {}))
            if missing:
                raise SystemExit(f"{key}: {len(missing)} long replies were not asked again; run recheck first")
            checked = rechecked.get(key, {})
            at_limit = {task_id for task_id, row in checked.items() if row["stop_type"] == "limit" or row["tokens_out"] >= LIMIT}
        changed = sum(not row["same_reply_as_saved"] for row in checked.values())
        if changed:
            notes.append(f"{key}: {changed} of {len(checked)} re-asked replies differ from the saved text")
        table.append({"file": key, "engine": "llama.cpp", "set": "frozen 591", **both_rules(record["rows"], at_limit, tasks)})
        if path.name in ("base_3b_f16_v1.json", "mine_multistep_f16_v1.json", "mine_recovery_f16_v1.json"):   # part 12's llama.cpp rows
            table.append({"file": key, "engine": "llama.cpp", "set": "200-task subset",
                          **both_rules([row for row in record["rows"] if row["id"] in subset], at_limit, tasks)})
    for path in sorted((ROOT / "results" / "part12").glob("*_sub200.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        at_limit = {row["id"] for row in record["rows"] if (row.get("tokens_out") or 0) >= record["max_tokens"]}
        table.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), "engine": record["engine"], "set": "200-task subset",
                      **both_rules(record["rows"], at_limit, tasks)})
    (OUT / "report.json").write_text(json.dumps({"limit": LIMIT, "rows": table, "notes": notes}, indent=1), encoding="utf-8")

    def cell(entry: dict, label: str) -> str:
        v = entry[label]
        return f"{v['old_percent']}%" if v["old_correct"] == v["new_correct"] else f"{v['old_percent']}% → **{v['new_percent']}%**"

    lines = ["| File | Set | At the limit | Changed | All | Call | Should ask | No tool fits |", "|---|---|---|---|---|---|---|---|"]
    for entry in table:
        lines.append(f"| `{entry['file']}` | {entry['set']} | {entry['all']['at_limit']} | "
                     f"{'**yes**' if entry['all']['old_correct'] != entry['all']['new_correct'] else 'no'} | "
                     + " | ".join(cell(entry, label) for label, _ in TYPES) + " |")
    (OUT / "report.md").write_text("\n".join(lines + [""] + notes) + "\n", encoding="utf-8")
    print("\n".join(lines + notes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["candidates", "recheck", "report"])
    {"candidates": candidates, "recheck": recheck, "report": report}[parser.parse_args().step]()
