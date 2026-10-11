"""Summary tables for the tool-call results, built from the saved replies. No model is run.

    python bench/summary_tables.py          prints the tables and writes results/part2/summary_tables.{json,md}

For every llama.cpp row of parts 2 and 3:
  - the total on all 591 tasks under both scorer rules (old: as scored; new: a reply that ran
    to the token limit is wrong), and on the 572 tasks whose prompt shows a tool list;
  - should-ask tasks: how many made no call (what the scorer counts), and how many of those
    replies contain a question (a "?");
  - no-tool-fits tasks split into the 71 that show tools and the 19 that show none. With no
    tool list llama-server drops all tool text from a Llama prompt, so those 19 are a prompt
    effect and not a skill.
Limit-length replies come from results/limit_rule/recheck.json for older files and from the
row itself (`at_limit`) for files written by the current scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {t["id"]: t for t in map(json.loads, (ROOT / "data/toolset/v1/toolset_v1.jsonl").read_text(encoding="utf-8").splitlines())}
ROWS = [("Base 3B", "user", "part2/base_3b_f16_v1.json"), ("Base 3B", "system", "part2/base_3b_f16_system_v1.json"),
        ("Base 1B", "user", "part2/base_1b_f16_v1.json"),
        ("First fine-tune (`mine`)", "user", "part2/mine_f16_v1.json"), ("First fine-tune (`mine`)", "system", "part2/mine_f16_system_v1.json"),
        ("Second fine-tune (`mine_multistep`)", "user", "part2/mine_multistep_f16_v1.json"),
        ("Second fine-tune (`mine_multistep`)", "system", "part2/mine_multistep_f16_system_v1.json"),
        ("**Final model: third fine-tune (`mine_recovery`)**", "user", "part2/mine_recovery_f16_v1.json"),
        ("**Final model: third fine-tune (`mine_recovery`)**", "system", "part2/mine_recovery_f16_system_v1.json"),
        ("xLAM-2-3b-fc-r (yardstick)", "its own template", "part2/xlam2_3b_f16_v1.json")]


def measure(path: Path, rechecked: dict) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    key = "results/" + str(path.relative_to(ROOT / "results")).replace("\\", "/")
    limit = {i for i, r in rechecked.get(key, {}).items() if r["stop_type"] == "limit" or r["tokens_out"] >= 384}
    rows = record["rows"]
    for row in rows:
        row["_limit"] = bool(row.get("at_limit")) or row["id"] in limit
        row["_type"] = TASKS[row["id"]]["type"]
        row["_tools"] = bool(TASKS[row["id"]]["tools"])

    def count(keep, new=False) -> tuple[int, int]:
        part = [r for r in rows if keep(r)]
        return sum(bool(r["fully_correct"]) and not (new and r["_limit"]) for r in part), len(part)

    ask_ok = [r for r in rows if r["_type"] == "ask" and r["fully_correct"] and not r["_limit"]]
    return {"all_old": count(lambda r: True), "all_new": count(lambda r: True, True),
            "with_tools_new": count(lambda r: r["_tools"], True),
            "call_new": count(lambda r: r["_type"] == "call", True),
            "ask_no_call_old": count(lambda r: r["_type"] == "ask"), "ask_no_call_new": count(lambda r: r["_type"] == "ask", True),
            "ask_with_question": (sum("?" in (r["content"] or "") for r in ask_ok), len([r for r in rows if r["_type"] == "ask"])),
            "no_tool_shown_new": count(lambda r: r["_type"] == "no_tool" and r["_tools"], True),
            "no_tool_none_shown_new": count(lambda r: r["_type"] == "no_tool" and not r["_tools"], True),
            "no_tool_old": count(lambda r: r["_type"] == "no_tool"), "at_limit": sum(r["_limit"] for r in rows)}


def pct(pair: tuple[int, int]) -> str:
    return f"{100 * pair[0] / pair[1]:.1f}% ({pair[0]})"


def main() -> None:
    rechecked = json.loads((ROOT / "results/limit_rule/recheck.json").read_text(encoding="utf-8"))
    out, lines = {}, []
    lines += ["| Model | Tool list in | All 591, old rule | All 591, token-limit rule | 572 tasks that show tools | Call (371) | "
              "Should ask: made no call (130) | of those, reply asks a question | No tool fits, tools shown (71) | No tool fits, no tools shown (19) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for model, layout, name in ROWS:
        path = ROOT / "results" / name
        if not path.exists():
            continue
        m = measure(path, rechecked)
        out[name] = {k: list(v) if isinstance(v, tuple) else v for k, v in m.items()}
        lines.append(f"| {model} | {layout} | {pct(m['all_old'])} | {pct(m['all_new'])} | {pct(m['with_tools_new'])} | {pct(m['call_new'])} | "
                     f"{pct(m['ask_no_call_new'])} | {m['ask_with_question'][0]} | {pct(m['no_tool_shown_new'])} | {pct(m['no_tool_none_shown_new'])} |")
    text = "\n".join(lines)
    (ROOT / "results/part2/summary_tables.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "results/part2/summary_tables.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
