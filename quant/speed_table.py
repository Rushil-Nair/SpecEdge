"""Part 3: the speed and memory half of the quantization table, from results/part3.

    python quant/speed_table.py            prints the tables and writes results/part3/speed_tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "part3"
LEVELS = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]


def load(name: str, level: str, forced: bool, device: str) -> dict | None:
    path = OUT / f"{name}_{level}_{'json' if forced else 'free'}_speed_{device}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def three(values: dict, digits: int = 1) -> str:
    return f"{values['median']:.{digits}f} ({values['min']:.{digits}f}-{values['max']:.{digits}f})"


def table(name: str, device: str) -> list[str]:
    control = load(name, "F16", False, device)
    if control is None:
        return []
    gpu = device == "gpu"
    lines = [f"**`{name}`, {'GPU (' + control['hardware']['gpu'] + ', ' + str(control['hardware']['cpus_visible']) + ' CPUs visible)' if gpu else 'CPU (' + str(control['threads']) + ' threads, GPU hidden)'}, "
             f"{control['tasks']} tasks, three passes, median (lowest-highest)**", "",
             "| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | "
             + ("GPU memory in use | " if gpu else "Peak server memory | ") + "Replies same as the GPU accuracy run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for level in LEVELS:
        for forced in (False, True):
            record = load(name, level, forced, device)
            if record is None:
                continue
            memory = record["memory"]
            lines.append(
                f"| {level} | {record['file_bytes'] / 1e9:.2f} GB | {'on' if forced else 'off'} | {three(record['generation_tok_s'])} | "
                f"{record['generation_tok_s']['median'] / control['generation_tok_s']['median']:.2f}x | "
                f"{three(record['prompt_tok_s'], 0)} | {three(record['first_token_ms'], 0)} | "
                + (f"{memory['gpu_mib_in_use'] / 1024:.2f} GB" if gpu else f"{memory['server_peak_rss_gb']:.2f} GB")
                + f" | {record.get('replies_same_as_accuracy_run')} |")
    if not gpu:
        lines += ["", "Peak memory is higher at Q4_K_M and Q4_0 than at Q5_K_M because llama.cpp's CPU backend copies "
                      "4-bit tensors into a rearranged layout (the CPU_REPACK buffer: 1,299 MiB at Q4_K_M, 1,512 MiB at Q4_0) "
                      "while the file stays mapped, so those weights are held twice; the other levels are used straight from "
                      "the mapped file (`results/part3/cpu_repack_check/`). \"First token\" is prompt time plus one token, "
                      "calculated, not measured."]
    else:
        lines += ["", "\"First token\" is prompt time plus one token, calculated, not measured. GPU memory is read once at "
                      "the end of the run, not a peak."]
    return lines + [""]


def main() -> None:
    lines: list[str] = []
    for name in ("mine_recovery", "mine_multistep", "mine"):
        for device in ("gpu", "cpu"):
            lines += table(name, device)
    text = "\n".join(lines)
    (OUT / "speed_tables.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
