"""Score a Qualcomm w4a16 checkpoint on the frozen tool-call set (part 12). Runs on the A100.

    python npu/score_quantized.py --checkpoint /workspace/quant_out --name stock_3b_w4a16

The checkpoint is what Qualcomm's quantize script writes. It is run here in simulation
(AIMET-ONNX on the GPU), which Qualcomm describes as aiming to replicate on-device accuracy;
it is not a run on the phone. Generation follows Qualcomm's own demo code (same model class
and generator), but greedy instead of sampled, so it can be compared with the llama.cpp rows.

The prompt is the Hugging Face chat template with the tool list, which earlier checks showed
to be the same text llama-server sends apart from the date line; the date is set to the day
the full-precision base score was run. Replies are scored by bench/toolcall_eval.py, the same
scorer as every other row. Results are saved after every task, so the run can be stopped and
resumed; --subset N scores a fixed random N-task subset (seed 5) if the full set is too slow.

--float is the control for the quantization loss: the full-precision weights through the same
Qualcomm model class, generator, prompt and stopping rule, so that the only difference from
the w4a16 row is the quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import GenerationConfig, LogitsProcessorList

from qai_hub_models.models.llama_v3_2_3b_instruct.model import Llama3_2_3B_PreSplit, QuantizedSplitModelWrapper
from qai_hub_models.models.templates.llama3.model import END_TOKENS
from qai_hub_models.models.templates.llm.generator_factory import make_generator
from qai_hub_models.models.templates.llm.model import (DEFAULT_CONTEXT_LENGTH, DEFAULT_EXPORT_SEQUENCE_LENGTHS,
                                                       get_tokenizer)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import forced_json_hf  # noqa: E402
import toolcall_eval as te  # noqa: E402

TASKS = ROOT / "data" / "toolset" / "v1" / "toolset_v1.jsonl"
MAX_NEW, DATE, SUBSET_SEED = 384, "08 Oct 2026", 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--forced-json", action="store_true",
                        help="constrain generation to well-formed calls or plain text (bench/forced_json_hf.py)")
    parser.add_argument("--float", action="store_true",
                        help="the control: --checkpoint is a full-precision Hugging Face folder, run through the "
                             "same Qualcomm model class and generator with no quantization")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new tasks (for a timing probe)")
    args = parser.parse_args()
    frozen = json.loads(TASKS.with_name("FROZEN.json").read_text(encoding="utf-8"))
    if hashlib.sha256(TASKS.read_bytes()).hexdigest() != frozen["sha256"]:
        raise SystemExit("the task file does not match its frozen checksum")
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines()]
    if args.subset:
        tasks = random.Random(SUBSET_SEED).sample(tasks, args.subset)
    out = ROOT / "results" / "part12" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"] if out.exists() and out.stat().st_size else []
    done = {row["id"] for row in rows}

    device = torch.device("cuda")
    tokenizer = get_tokenizer(args.checkpoint)
    begin = time.time()
    model_cls = Llama3_2_3B_PreSplit if args.float else QuantizedSplitModelWrapper
    model = model_cls.from_pretrained(host_device=device, checkpoint=args.checkpoint).to(device)
    generator = make_generator(model, sequence_length=sorted(DEFAULT_EXPORT_SEQUENCE_LENGTHS),
                               context_length=DEFAULT_CONTEXT_LENGTH, vision_model=None, model_cls=Llama3_2_3B_PreSplit)
    # Llama 3.2 ends a tool call with <|eom_id|>, which Qualcomm's END_TOKENS leaves out; llama-server
    # stops there, so this run must too or the reply runs on into a made-up next turn.
    ends = [ids[0] for ids in (tokenizer.encode(t, add_special_tokens=False) for t in sorted(END_TOKENS | {"<|eom_id|>"}))
            if len(ids) == 1]
    config = GenerationConfig(max_new_tokens=MAX_NEW, eos_token_id=ends + [tokenizer.eos_token_id],
                              pad_token_id=tokenizer.pad_token_id, do_sample=False)
    generator.generation_config = config        # as Qualcomm's demo does before generating
    print(f"model ready in {time.time() - begin:.0f} s", flush=True)

    def save() -> None:
        record = {"name": args.name, "checkpoint": args.checkpoint, "engine": ("Qualcomm's full-precision model class and generator, no quantization (qai-hub-models 0.63.0)" if args.float
                             else "AIMET-ONNX simulation of w4a16 (qai-hub-models 0.63.0)"),
                  "tasks_file": str(TASKS.relative_to(ROOT)), "subset": args.subset or None, "subset_seed": SUBSET_SEED if args.subset else None,
                  "tasks_planned": len(tasks), "decoding": "greedy", "forced_json": args.forced_json, "max_tokens": MAX_NEW, "date_in_prompt": DATE,
                  "context_length": DEFAULT_CONTEXT_LENGTH, "gpu": torch.cuda.get_device_name(0),
                  "finished_unix": time.time(), "summary": te.summarise(rows), "rows": rows}
        # ASCII-escaped, because a reply can hold half a character; written beside the file and
        # then moved over it, so a failed save cannot leave an empty file behind.
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(out)

    new = 0
    for task in tasks:
        if task["id"] in done:
            continue
        if args.limit and new >= args.limit:
            break
        text, error, start, tokens_out = None, None, time.time(), 0
        try:
            prompt = tokenizer.apply_chat_template(task["messages"], tools=te.sendable(task["tools"]), tokenize=False,
                                                   add_generation_prompt=True, date_string=DATE)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            length = inputs["input_ids"].shape[1]
            if length + MAX_NEW > DEFAULT_CONTEXT_LENGTH:
                raise ValueError(f"prompt of {length} tokens does not fit the {DEFAULT_CONTEXT_LENGTH}-token context")
            extra = {}
            if args.forced_json:
                forcer = forced_json_hf.ForcedJson(tokenizer, te.sendable(task["tools"]), length, config.eos_token_id)
                extra["logits_processor"] = LogitsProcessorList([forcer])
            with torch.no_grad():
                output = generator.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                            generation_config=config, **extra)
            tokens_out = output.shape[1] - length
            text = tokenizer.decode(output[0, length:], skip_special_tokens=True)
        except Exception as err:  # noqa: BLE001 - a task the simulated model cannot run is a result
            error = f"{type(err).__name__}: {str(err)[:200]}"
        rows.append({"id": task["id"], "source": task["source"], "type": task["type"], "tags": task.get("tags", []),
                     "expect": task["expect"], "content": text, "error": error, "seconds": round(time.time() - start, 2),
                     "tokens_out": tokens_out, **te.score_text(task, text, error)})
        new += 1
        save()
        if new <= 3 or new % 25 == 0:
            print(f"{len(rows)}/{len(tasks)}  {rows[-1]['seconds']} s  {tokens_out} tokens  "
                  f"correct so far {sum(r['fully_correct'] for r in rows)}  | {str(text)[:90]!r} {error or ''}", flush=True)
    save()
    te.show(te.summarise(rows))
    print("SCORE_DONE", args.name, len(rows), "tasks", flush=True)


if __name__ == "__main__":
    main()
