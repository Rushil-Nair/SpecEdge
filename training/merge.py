"""Merge a LoRA add-on into the base model, for quantization (part 3). Runs on the rented machine.

    python training/merge.py --base <base model dir> --adapter <run dir>/adapter --out <merged dir>

The merged model stays on the machine and on the laptop only: by design the add-on
and the code are what gets published, not merged weights.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    # The tokenizer is not changed by the add-on. Its files are copied from the base model as
    # they are: transformers 5 would re-save them in a newer layout (tokenizer class
    # "TokenizersBackend", the chat template in a separate file) that llama.cpp's converter and
    # Qualcomm's tools, both on older transformers, cannot read.
    # The same goes for config.json, and it matters more: transformers 5 writes the rotary
    # position settings under a new key ("rope_parameters"). Qualcomm's tools, on transformers
    # 4.45, then read rope_theta as 10000 instead of 500000 and no "llama3" scaling, and run the
    # model with the wrong position encoding. llama.cpp's converter reads the new key correctly.
    # The add-on changes no shape, so the base model's config is the merged model's config.
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy2(Path(args.base) / name, Path(args.out) / name)
    print("MERGE_DONE", args.out)


if __name__ == "__main__":
    main()
