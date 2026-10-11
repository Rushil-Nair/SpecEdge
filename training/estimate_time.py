"""Before a fine-tune: how long will it take, from the token count of its training file?

    python training/estimate_time.py --new <train.jsonl> --reference <train.jsonl> --reference-minutes 200.7 --limit-minutes 300

Training time follows the number of tokens that pass through the model. The reference is a
file whose run was timed on the same kind of machine with the same recipe (the second
fine-tune: train_v2_clean, 200.7 minutes on an RTX 5090). Examples longer than the trainer's
cut (2,048 tokens) are left out, as the trainer does. Exit status 3 when the estimate is over
the limit: a step of more than about 5 GPU-hours needs a deliberate decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def tokens(path: str, tokenizer, cut: int) -> tuple[int, int, int]:
    total = used = dropped = 0
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    for start in range(0, len(rows), 512):
        batch = rows[start:start + 512]
        lengths = [len(ids) for ids in tokenizer([r["prompt"] + r["completion"] for r in batch], add_special_tokens=False)["input_ids"]]
        for length in lengths:
            if length > cut:
                dropped += 1
            else:
                used += 1
                total += length
    return total, used, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--new", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-minutes", type=float, required=True)
    parser.add_argument("--limit-minutes", type=float, default=300)
    parser.add_argument("--base", default="/workspace/models/llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    new, new_used, new_dropped = tokens(args.new, tokenizer, args.max_tokens)
    old, old_used, old_dropped = tokens(args.reference, tokenizer, args.max_tokens)
    minutes = args.reference_minutes * new / old
    record = {"new_file": args.new, "new_tokens": new, "new_examples_used": new_used, "new_examples_over_the_cut": new_dropped,
              "reference_file": args.reference, "reference_tokens": old, "reference_examples_used": old_used,
              "reference_examples_over_the_cut": old_dropped, "reference_minutes": args.reference_minutes,
              "estimated_minutes": round(minutes, 1), "limit_minutes": args.limit_minutes, "over_the_limit": minutes > args.limit_minutes}
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1))
    raise SystemExit(3 if minutes > args.limit_minutes else 0)


if __name__ == "__main__":
    main()
