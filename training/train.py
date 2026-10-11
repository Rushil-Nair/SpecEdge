"""LoRA fine-tune of Llama 3.2 3B for tool calls (part 2). Runs on the rented machine.

    python training/train.py --config training/configs/lora_r16_full.json

The training file comes from training/prepare.py: one JSON object per line with "prompt" (the
model's own chat template, tools included, ending where the assistant starts) and
"completion" (what the assistant should write, ending with the end-of-turn token). Only the
completion is trained on; prompt tokens are masked out of the loss.

Checkpoints are written every save_steps and the run resumes from the newest one if it is
started again, so it survives a restart of the machine or of the queue. At the end it writes
results/part2/<run_id>.json with the config, seed, model revision, package versions, hardware,
wall-clock time and the loss curve.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]


class Examples(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer, max_tokens: int, limit: int | None, seed: int):
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if limit:                                  # a fixed random subset, for the half-data run
            rows = random.Random(seed).sample(rows, limit)
        self.items, self.too_long = [], 0
        for row in rows:
            prompt = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            completion = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
            if len(prompt) + len(completion) > max_tokens:
                self.too_long += 1
                continue
            self.items.append((prompt, completion))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        prompt, completion = self.items[index]
        return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}


def collate(pad_id: int):
    def run(batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[row, :n] = torch.tensor(item["input_ids"])
            labels[row, :n] = torch.tensor(item["labels"])
            mask[row, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return run


class Curve(TrainerCallback):
    def __init__(self) -> None:
        self.points: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.points.append({"step": state.global_step, "loss": logs["loss"], "lr": logs.get("learning_rate")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    out_dir = Path(cfg["output_dir"])
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.pad_token or "<|finetune_right_pad_id|>"
    print(f"run {cfg['run_id']}: training file {cfg['train_file']}", flush=True)
    data = Examples(cfg["train_file"], tokenizer, cfg["max_tokens"], cfg.get("limit"), cfg["seed"])
    print(f"{len(data)} training examples, {data.too_long} left out as longer than {cfg['max_tokens']} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"], task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # transformers 5 has no warmup_ratio or group_by_length; the warm-up is given in steps.
    total_steps = math.ceil(len(data) / (cfg["batch_size"] * cfg["accumulate"])) * cfg["epochs"]
    arguments = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=cfg["epochs"], per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["accumulate"], learning_rate=cfg["learning_rate"], lr_scheduler_type="cosine",
        warmup_steps=int(cfg["warmup_ratio"] * total_steps), weight_decay=0.0, bf16=True, logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"], save_total_limit=2, seed=cfg["seed"], data_seed=cfg["seed"], report_to=[],
        dataloader_num_workers=2, remove_unused_columns=False)
    curve = Curve()
    trainer = Trainer(model=model, args=arguments, train_dataset=data, data_collator=collate(tokenizer.pad_token_id),
                      callbacks=[curve])
    resume = any(out_dir.glob("checkpoint-*"))
    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    seconds = time.time() - start
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    def version(package: str) -> str:
        return __import__(package).__version__

    record = {
        "run_id": cfg["run_id"], "config": cfg, "resumed_from_checkpoint": resume,
        "examples_used": len(data), "examples_too_long": data.too_long, "trainable_parameters": trainable,
        "steps": result.global_step, "final_train_loss": result.training_loss,
        "wall_seconds_this_start": round(seconds, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "base_model_revision": cfg.get("base_model_revision"),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft")},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or subprocess.run(
            "grep -m1 'model name' /proc/cpuinfo", shell=True, capture_output=True, text=True).stdout.strip(),
            "cpus_visible": os.cpu_count()},
        "finished_unix": time.time(), "loss_curve": curve.points,
    }
    results = ROOT / "results" / "part2"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{cfg['run_id']}.json").write_text(json.dumps(record, indent=1))
    print(f"TRAIN_DONE {cfg['run_id']} steps {result.global_step} loss {result.training_loss:.4f} "
          f"in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
