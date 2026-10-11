"""Part 12, recovery attempt: Qualcomm's quantizer with tool-call text as calibration data.

    python npu/quantize_with_calibration.py --calibration <train examples .jsonl> \
        --precision w4a16 --checkpoint <merged model dir> --output-dir <out>

Everything after --calibration is passed to Qualcomm's own quantize command unchanged, so the
recipe is the same as the first run (Sequential MSE, then calibration, 20 iterations each).
The only difference is the text: the recipe names the "Wikitext" dataset, and
qai-hub-models 0.63.0 can load no other. This script replaces what that dataset class
reads, for this process only, with tool-call prompts and answers taken from the fine-tuning
file (never from the test set). The installed package is not edited.

Each example is the prompt followed by its answer. The dataset class joins texts with the
begin-of-text token, so the one at the start of every prompt is removed first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import Dataset

from qai_hub_models.datasets.wikitext.wikitext import WikiText
from qai_hub_models.models.llama_v3_2_3b_instruct.model import (MODEL_ID, SUPPORTED_PRECISIONS, Llama3_2_3B_PreSplit,
                                                                Llama3_2_3B_QuantizablePreSplit)
from qai_hub_models.models.templates.llm.quantize import llm_quantize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibration", required=True)
    args, rest = parser.parse_known_args()
    raw = Path(args.calibration).read_bytes()
    texts = []
    for line in raw.decode("utf-8").splitlines():
        row = json.loads(line)
        texts.append((row["prompt"] + row["completion"]).removeprefix("<|begin_of_text|>"))

    def tool_call_text(self) -> Dataset:
        print(f"CALIBRATION TEXT: {len(texts)} training examples from {args.calibration} "
              f"(sha256 {hashlib.sha256(raw).hexdigest()[:16]}), split asked for: {self.split_str}", flush=True)
        return Dataset.from_dict({"text": texts})

    WikiText.load_raw_dataset = tool_call_text
    sys.argv = [sys.argv[0]] + rest
    llm_quantize(quantized_model_cls=Llama3_2_3B_QuantizablePreSplit, fp_model_cls=Llama3_2_3B_PreSplit,
                 model_id=MODEL_ID, supported_precisions=SUPPORTED_PRECISIONS)


if __name__ == "__main__":
    main()
