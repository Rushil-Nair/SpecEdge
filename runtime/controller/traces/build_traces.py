"""Turn recorded runs into the traces the policy module is tested against.

    python runtime/controller/traces/build_traces.py

Reads only files already in results/ (no model, no machine):
  results/load_check/{plain,dflash_n16}_u{1,4,16}.json   Qwen3-8B 4-bit, RTX 5090, five kinds of
                                                         text, 200 requests, 3 passes, at 1, 4 and
                                                         16 users. Plain decoding and DFlash block 16.
  results/hard_text/{plain,dflash_n4}.json   one user, Chinese and long reasoning,
                                                         plain and DFlash block 4.

Writes, beside this script:
  load_u1.csv, load_u4.csv, load_u16.csv   one row per request in the order the load run sent
      them: what plain decoding took, what drafting took, and how many drafted tokens were
      accepted. Times are the server's generation time for that request (predicted_ms), median
      of the three passes.
  ANALYSIS.md   where drafting stops paying, by accepted share, from the same files. The
      policy's default thresholds are read off this.

A trace says what each choice cost when it was recorded. Replaying a policy over it gives an
estimate of that policy's generation time on the same requests, not a measurement: under load
the recorded time of one request depends on what the other requests were doing.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "results"
KINDS = ["code", "tool_calls", "chat", "chinese", "reasoning"]


def medians(record: dict, key: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for one in record["passes"]:
        for row in one["rows"]:
            values.setdefault(row["id"], []).append(row[key])
    return {name: statistics.median(v) for name, v in values.items()}


def break_even(points: list[tuple[float, float]], thresholds: list[float]) -> list[str]:
    """points: (accepted share, drafting time / plain time) per request."""
    lines = ["| Accepted share | Requests below | of them slower than plain | Requests at or above | of them slower than plain |",
             "|---|---|---|---|---|"]
    for t in thresholds:
        low = [ratio for share, ratio in points if share < t]
        high = [ratio for share, ratio in points if share >= t]
        lines.append(f"| {t:.2f} | {len(low)} | {sum(r > 1 for r in low)} | {len(high)} | {sum(r > 1 for r in high)} |")
    return lines


def main() -> None:
    analysis = ["# Where drafting stops paying — from recorded runs", "",
                "Built by `build_traces.py` from `results/load_check/` and `results/hard_text/`. "
                "Qwen3-8B 4-bit on the RTX 5090, llama.cpp `f498f864`. Per request: generation time with drafting "
                "divided by generation time with plain decoding (median of three passes), against the share of "
                "drafted tokens that were accepted. Requests shorter than 32 tokens are left out of the per-request "
                "tables.", ""]
    for users in (1, 4, 16):
        plain = json.loads((RESULTS / "load_check" / f"plain_u{users}.json").read_text(encoding="utf-8"))
        draft = json.loads((RESULTS / "load_check" / f"dflash_n16_u{users}.json").read_text(encoding="utf-8"))
        plain_ms, draft_ms = medians(plain, "predicted_ms"), medians(draft, "predicted_ms")
        drafted, accepted, tokens = medians(draft, "draft_n"), medians(draft, "draft_n_accepted"), medians(plain, "tokens")
        order = [row for row in draft["passes"][0]["rows"]]
        lines = ["index,id,kind,users,tokens,plain_ms,draft_ms,draft_n,draft_n_accepted"]
        for index, row in enumerate(order):
            name = row["id"]
            lines.append(f'{index},{name.replace(",", "_")},{row["kind"]},{users},{tokens[name]:.0f},{plain_ms[name]:.3f},'
                         f'{draft_ms[name]:.3f},{drafted[name]:.0f},{accepted[name]:.0f}')
        (HERE / f"load_u{users}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

        kind_of = {row["id"]: row["kind"] for row in order}
        analysis += [f"## DFlash block 16, {users} user{'s' if users > 1 else ''} in flight", "",
                     "| Kind | Accepted share | Drafting time / plain time |", "|---|---|---|"]
        for kind in KINDS:
            names = [n for n in kind_of if kind_of[n] == kind]
            share = sum(accepted[n] for n in names) / max(1.0, sum(drafted[n] for n in names))
            analysis.append(f"| {kind} | {share:.3f} | {sum(draft_ms[n] for n in names) / sum(plain_ms[n] for n in names):.2f} |")
        total = sum(draft_ms.values()) / sum(plain_ms.values())
        analysis += [f"| all 200 requests | | {total:.2f} |", ""]
        points = [(accepted[n] / drafted[n] if drafted[n] else 0.0, draft_ms[n] / plain_ms[n]) for n in kind_of if tokens[n] >= 32]
        analysis += break_even(points, [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40]) + [""]

    # block 4, one user: Chinese and reasoning only
    plain = json.loads((RESULTS / "hard_text" / "plain.json").read_text(encoding="utf-8"))
    short = json.loads((RESULTS / "hard_text" / "dflash_n4.json").read_text(encoding="utf-8"))

    def by_id(record: dict, field) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for run in record["runs"]:
            values.setdefault(run["id"], []).append(field(run))
        return {name: statistics.median(v) for name, v in values.items()}

    p_ms = by_id(plain, lambda r: r["timings"]["predicted_ms"])
    s_ms = by_id(short, lambda r: r["timings"]["predicted_ms"])
    s_n = by_id(short, lambda r: r["timings"]["draft_n"])
    s_a = by_id(short, lambda r: r["timings"]["draft_n_accepted"])
    n_tok = by_id(short, lambda r: r["timings"]["predicted_n"])
    kinds = {run["id"]: run["kind"] for run in short["runs"]}
    analysis += ["## DFlash block 4, 1 user (Chinese and long reasoning only)", "",
                 "| Kind | Accepted share | Drafting time / plain time |", "|---|---|---|"]
    for kind in sorted(set(kinds.values())):
        names = [n for n in kinds if kinds[n] == kind and n in p_ms]
        analysis.append(f"| {kind} | {sum(s_a[n] for n in names) / sum(s_n[n] for n in names):.3f} | "
                        f"{sum(s_ms[n] for n in names) / sum(p_ms[n] for n in names):.2f} |")
    points = [(s_a[n] / s_n[n] if s_n[n] else 0.0, s_ms[n] / p_ms[n]) for n in kinds if n in p_ms and n_tok[n] >= 32]
    analysis += [""] + break_even(points, [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]) + [""]
    analysis += ["## Not in any recorded run", "",
                 "- Block 4 (or any block shorter than 16) with more than one user.",
                 "- Any point between 4 and 16 users.",
                 "- Acceptance per decoding step: the files hold totals per request.", ""]
    (HERE / "ANALYSIS.md").write_text("\n".join(analysis), encoding="utf-8", newline="\n")
    print("\n".join(analysis))


if __name__ == "__main__":
    main()
