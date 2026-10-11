"""Part 12: is tool-call calibration worth running? Decided from two result files.

    python npu/calibration_gate.py <control .json> <w4a16 .json> [--threshold 3.0]

Rule (fixed 2026-10-08, before the run): if the default w4a16 model loses less than 3 points against its
own full-precision control in Qualcomm's engine, calibration is skipped. The score used is
the one where a reply that ran to the token limit counts as wrong (the scorer otherwise
counts junk with no call in it as a correct "no call"); the score without that rule is
printed beside it. Exit status 0 means skip calibration, 3 means run it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def scores(path: Path) -> tuple[float, float, int, int]:
    record = json.loads(path.read_text(encoding="utf-8"))
    rows, limit = record["rows"], record["max_tokens"]
    old = sum(bool(r["fully_correct"]) for r in rows)
    at_limit = [r for r in rows if (r.get("tokens_out") or 0) >= limit]
    new = old - sum(bool(r["fully_correct"]) for r in at_limit)
    return 100 * old / len(rows), 100 * new / len(rows), len(at_limit), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("control", type=Path)
    parser.add_argument("quantized", type=Path)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()
    c_old, c_new, c_limit, c_n = scores(args.control)
    q_old, q_new, q_limit, q_n = scores(args.quantized)
    if c_n != q_n:
        raise SystemExit(f"the two files hold different numbers of tasks ({c_n} and {q_n})")
    print(f"control   {c_old:.1f}% as scored, {c_new:.1f}% with limit-length replies wrong ({c_limit} of {c_n} at the limit)")
    print(f"w4a16     {q_old:.1f}% as scored, {q_new:.1f}% with limit-length replies wrong ({q_limit} of {q_n} at the limit)")
    loss_old, loss_new = c_old - q_old, c_new - q_new
    print(f"loss      {loss_old:.1f} points as scored, {loss_new:.1f} points with limit-length replies wrong")
    skip = loss_new < args.threshold
    print(f"decision  {'SKIP calibration' if skip else 'RUN calibration'} (threshold {args.threshold} points, limit rule applied)")
    raise SystemExit(0 if skip else 3)


if __name__ == "__main__":
    main()
