"""The final model against the xLAM-2-3b yardstick, category by category, as a paired test.

    python bench/xlam_paired.py            prints the table and writes results/part2/xlam_paired.md

Both models answered the same 591 frozen tasks, so each task is a pair. "Only mine" and "only
xLAM" count the tasks one model got and the other did not; the p-value is the exact two-sided
McNemar (binomial) test on those two counts. Token-limit rule applied (neither model has a
reply at the limit, so it changes nothing here).
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {t["id"]: t for t in map(json.loads, (ROOT / "data/toolset/v1/toolset_v1.jsonl").read_text(encoding="utf-8").splitlines())}
GROUPS = [("All 591", lambda t: True), ("The 572 that show tools", lambda t: bool(t["tools"])),
          ("Call (371)", lambda t: t["type"] == "call"), ("Call tasks with several calls (156)", lambda t: "multi_call" in t.get("tags", [])),
          ("Should ask: made no call (130)", lambda t: t["type"] == "ask"), ("No tool fits, all (90)", lambda t: t["type"] == "no_tool"),
          ("No tool fits, tools shown (71)", lambda t: t["type"] == "no_tool" and bool(t["tools"])),
          ("No tool fits, no tools shown (19)", lambda t: t["type"] == "no_tool" and not t["tools"])]


def verdicts(name: str) -> dict[str, bool]:
    rows = json.loads((ROOT / "results/part2" / name).read_text(encoding="utf-8"))["rows"]
    return {r["id"]: bool(r["fully_correct"]) and not r.get("at_limit") for r in rows}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main() -> None:
    mine, xlam = verdicts("mine_recovery_f16_v1.json"), verdicts("xlam2_3b_f16_v1.json")
    lines = ["| Tasks | `mine_recovery` | xLAM-2-3b-fc-r | Only mine / only xLAM | McNemar p (exact, two-sided) | Reading |", "|---|---|---|---|---|---|"]
    for label, keep in GROUPS:
        ids = [i for i, t in TASKS.items() if keep(t)]
        a, b = sum(mine[i] for i in ids), sum(xlam[i] for i in ids)
        only_a, only_b = sum(mine[i] and not xlam[i] for i in ids), sum(xlam[i] and not mine[i] for i in ids)
        p = mcnemar(only_a, only_b)
        reading = "no significant difference" if p >= 0.05 else ("mine better" if only_a > only_b else "xLAM better")
        lines.append(f"| {label} | {100 * a / len(ids):.1f}% ({a}) | {100 * b / len(ids):.1f}% ({b}) | {only_a} / {only_b} | "
                     f"{p:.4f} | {reading} |")
    text = "\n".join(lines)
    (ROOT / "results/part2/xlam_paired.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
