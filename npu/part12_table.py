"""Part 12: one table of every row, under both scorer rules, from the result files.

    python npu/part12_table.py            prints the table and writes results/part12/table.md

Rows on the fixed 200-task subset (seed 5). "Old rule" is the score as written by the scorer;
"token-limit rule" counts a reply that ran to the 384-token limit as wrong. For every
quantized row the paired count against its own full-precision control in the same pipeline is
given: tasks lost and tasks gained, under the token-limit rule. A net difference of a few
tasks with many flips each way is noise; the paired counts show that.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P12 = ROOT / "results" / "part12"
TASKS = [json.loads(line) for line in (ROOT / "data/toolset/v1/toolset_v1.jsonl").read_text(encoding="utf-8").splitlines()]
SUBSET = {task["id"] for task in random.Random(5).sample(TASKS, 200)}
TYPE = {task["id"]: task["type"] for task in TASKS}
ROWS = [
    ("Stock 3B, full precision", "llama.cpp F16", ROOT / "results/part2/base_3b_f16_v1.json", None),
    ("Stock 3B, full precision (control)", "Qualcomm's engine", P12 / "stock_3b_float_sub200.json", None),
    ("Stock 3B, w4a16", "Qualcomm's simulation", P12 / "stock_3b_w4a16_sub200.json", "stock_3b_float_sub200.json"),
    ("Second fine-tune (`mine_multistep`), full precision", "llama.cpp F16", ROOT / "results/part2/mine_multistep_f16_v1.json", None),
    ("Second fine-tune, full precision (control)", "Qualcomm's engine", P12 / "mine_multistep_float_sub200.json", None),
    ("Second fine-tune, w4a16", "Qualcomm's simulation", P12 / "mine_multistep_w4a16_sub200.json", "mine_multistep_float_sub200.json"),
    ("Second fine-tune, w4a16, forced JSON", "Qualcomm's simulation", P12 / "mine_multistep_w4a16_json_sub200.json", "mine_multistep_float_sub200.json"),
    ("Second fine-tune, w4a16, tool-call calibration", "Qualcomm's simulation", P12 / "mine_multistep_w4a16_toolcal_sub200.json", "mine_multistep_float_sub200.json"),
    ("Second fine-tune, w4a16, tool-call calibration, forced JSON", "Qualcomm's simulation", P12 / "mine_multistep_w4a16_toolcal_json_sub200.json", "mine_multistep_float_sub200.json"),
    ("**Final model (`mine_recovery`)**, full precision", "llama.cpp F16", ROOT / "results/part2/mine_recovery_f16_v1.json", None),
    ("**Final model**, full precision (control)", "Qualcomm's engine", P12 / "mine_recovery_float_sub200.json", None),
    ("**Final model**, w4a16, tool-call calibration, forced JSON", "Qualcomm's simulation", P12 / "mine_recovery_w4a16_toolcal_json_sub200.json", "mine_recovery_float_sub200.json"),
]


def verdicts(path: Path) -> dict[str, tuple[bool, bool]]:
    """task id -> (correct under the old rule, correct under the token-limit rule)"""
    record = json.loads(path.read_text(encoding="utf-8"))
    limit_ids: set[str] = set()
    if path.parent != P12:                               # llama.cpp file: limit-length replies from the re-ask
        recheck = json.loads((ROOT / "results/limit_rule/recheck.json").read_text(encoding="utf-8"))
        key = "results/" + str(path.relative_to(ROOT / "results")).replace("\\", "/")
        limit_ids = {i for i, r in recheck.get(key, {}).items() if r["stop_type"] == "limit" or r["tokens_out"] >= 384}
    limit_ids |= {row["id"] for row in record["rows"] if row.get("at_limit")}     # newer files carry it in the row
    out = {}
    for row in record["rows"]:
        if row["id"] not in SUBSET:
            continue
        at_limit = row["id"] in limit_ids or (row.get("tokens_out") or 0) >= record.get("max_tokens", 384)
        out[row["id"]] = (bool(row["fully_correct"]), bool(row["fully_correct"]) and not at_limit)
    return out


def main() -> None:
    lines = ["| Row | Engine | All 200, old rule | All 200, token-limit rule | Call (121) | Should ask: made no call (45) | No tool fits (34) | "
             "Against its control: lost / gained |", "|---|---|---|---|---|---|---|---|"]
    for label, engine, path, control in ROWS:
        if not path.exists():
            lines.append(f"| {label} | {engine} | not run | | | | | |")
            continue
        v = verdicts(path)
        if len(v) != 200:                                    # a row still being scored saves as it goes
            lines.append(f"| {label} | {engine} | running ({len(v)} of 200 tasks) | | | | | |")
            continue

        def share(kind: str | None, rule: int) -> str:
            ids = [i for i in v if kind is None or TYPE[i] == kind]
            got = sum(v[i][rule] for i in ids)
            return f"{100 * got / len(ids):.1f}% ({got})"

        paired = ""
        if control and (P12 / control).exists():
            c = verdicts(P12 / control)
            paired = f"{sum(c[i][1] and not v[i][1] for i in v)} / {sum(v[i][1] and not c[i][1] for i in v)}"
        lines.append(f"| {label} | {engine} | {share(None, 0)} | {share(None, 1)} | {share('call', 1)} | {share('ask', 1)} | "
                     f"{share('no_tool', 1)} | {paired} |")
    text = "\n".join(lines)
    (P12 / "table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
