"""Build the two summary tables of part 6 from the result files.

    python runtime/agent/headline.py        # writes runtime/results/part6/HEADLINE.md

1. Timed runs (booked machine, 3 repeats): the stock llama.cpp server against the runtime,
   each feature on and off, one table per model, layout and device.
2. Tasks correct for every model on both task sets, per machine and layout, with the
   held-out set's two wording halves, under the current and the stricter answer rule.
Nothing is added across task sets or machines.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from report import WORDING
from strict import strict_correct

RESULTS = Path(__file__).resolve().parent.parent / "results" / "part6"
MACHINES = {"1536000 100000": "first 5090", "768000 100000": "second 5090"}
MODELS = {"Llama-3.2-3B-Instruct-Q4_K_M.gguf": "stock 3B", "mine-Q4_K_M.gguf": "mine (first fine-tune)",
          "mine_multistep-Q4_K_M.gguf": "mine_multistep (second)", "mine_recovery-Q4_K_M.gguf": "mine_recovery (third, final)"}
ORDER = [("server", "chat", False, False, "Stock server, chat endpoint"),
         ("server", "raw", False, False, "Stock server, raw text, no cache"),
         ("server", "raw", True, False, "Stock server, raw text, its own prompt cache"),
         ("runtime", None, False, False, "Runtime, nothing on"),
         ("runtime", None, True, False, "Runtime, cache reuse"),
         ("runtime", None, False, True, "Runtime, forced JSON"),
         ("runtime", None, True, True, "Runtime, cache reuse and forced JSON")]


def key_of(record: dict) -> tuple:
    config = record["config"]
    who = "runtime" if record["program"].startswith("specedge-agent") else "server"
    return (who, config.get("mode"), bool(config.get("cache")), bool(config.get("forced_json")))


def where(record: dict) -> tuple:
    config = record["config"]
    machine = MACHINES.get(record.get("hardware", {}).get("cpu_quota", ""), "machine not recognised")
    device = "GPU" if config.get("n_gpu_layers", 0) else f'CPU, {config.get("threads")} threads'
    which = "held-out" if "heldout" in record.get("tasks_file", "") else "development"
    return (MODELS.get(record["model"]["file"], record["model"]["file"]), which, config.get("tools_in", "user"), machine, device)


def med(values: list[float], digits: int = 2) -> str:
    if not values:
        return ""
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})"


def span(values: list[int]) -> str:
    return str(values[0]) if len(set(values)) == 1 else f"{min(values)}-{max(values)}"


def main() -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]
    records = [r for r in records if r.get("part") == 6 and "summary" in r]
    timed, untimed = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for record in records:
        (timed if record.get("timing_valid") else untimed)[where(record)][key_of(record)].append(record)

    out = ["# Part 6 — summary tables", "",
           "Built by `runtime/agent/headline.py` from the result files in this folder. All models are the 4-bit "
           "(Q4_K_M) files; llama.cpp `f498f864`; greedy decoding; at most 10 turns and 256 tokens per turn.", "",
           "## 1. Timed: the stock llama.cpp server against the runtime", "",
           "Machine booked for these runs alone, quiet check before each, 3 repeats. Seconds are per task (mean over "
           "the tasks of a run), shown as the median repeat with the lowest and highest. For the runtime, seconds are "
           "the whole task inside one program; for the stock server they include the HTTP calls of the Python loop "
           "that drives it. Prompt and generation are the engine's own times; the stock server's are its `timings`. "
           "\"Raw text\" rows read the server's text with the runtime's parser, so they differ from the runtime only "
           "in who runs the model.", ""]
    for place in sorted(timed):
        model, which, layout, machine, device = place
        runs0 = next(iter(timed[place].values()))
        tasks = runs0[0]["summary"]["tasks"]
        out += [f"### {model} — {which} set ({tasks} tasks), tool list in the {layout} message — {machine}, {device}", "",
                "| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | "
                "Ended with an answer |", "|---|---|---|---|---|---|---|---|"]
        wall = {}
        for who, mode, cache, forced, name in ORDER:
            runs = timed[place].get((who, mode, cache, forced), [])
            if not runs:
                out.append(f"| {name} | 0 | | | | | | |")
                continue
            seconds = [r["summary"]["seconds_per_task"]["wall_mean"] for r in runs]
            wall[name] = statistics.median(seconds)
            out.append(f"| {name} | {len(runs)} | {med(seconds)} | "
                       f"{med([r['summary']['seconds_per_task']['prompt_mean'] for r in runs])} | "
                       f"{med([r['summary']['seconds_per_task']['generation_mean'] for r in runs])} | "
                       f"{span([r['summary']['correct'] for r in runs])} | "
                       f"{span([sum(strict_correct(t) for t in r['tasks']) for r in runs])} | "
                       f"{span([r['summary'].get('ended_with_an_answer', 0) for r in runs])} |")
        full = wall.get(ORDER[6][4])
        if full:
            parts = []
            for index, label in ((0, "the stock chat endpoint"), (1, "the stock server without cache"), (2, "the stock server with its cache")):
                if wall.get(ORDER[index][4]):
                    parts.append(f"{wall[ORDER[index][4]] / full:.2f}x against {label}")
            out += ["", "Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): "
                    + "; ".join(parts) + ".", ""]
        else:
            out.append("")

    out += ["## 2. Tasks correct, every model, both task sets", "",
            "Runtime with cache reuse, 100 tasks, GPU. Each cell is tasks correct under the current rule, with the "
            "stricter answer rule in brackets. Rows marked timed are the 3 repeats of section 1 (they agreed unless a "
            "range is shown); the others are single runs on a machine that was not booked. The two task sets, the two "
            "machines and the two layouts are separate rows; nothing is added across them.", "",
            "| Model | Task set | Tool list in | Machine | Timed | Forced JSON off | Forced JSON on | "
            "Held-out, same wording as development (of 50): off / on | Held-out, new wording (of 50): off / on |",
            "|---|---|---|---|---|---|---|---|---|"]

    def cell(runs: list[dict], keep=lambda t: True) -> str:
        if not runs:
            return "not run"
        now = [sum(bool(t["correct"]) for t in r["tasks"] if keep(t)) for r in runs]
        hard = [sum(strict_correct(t) for t in r["tasks"] if keep(t)) for r in runs]
        return f"{span(now)} ({span(hard)})"

    for flag, groups in (("no", untimed), ("yes", timed)):
        for place in sorted(groups):
            model, which, layout, machine, device = place
            if device != "GPU":
                continue
            off, on = groups[place].get(("runtime", None, True, False), []), groups[place].get(("runtime", None, True, True), [])
            if not off and not on:
                continue
            halves = ["", ""]
            if which == "held-out":
                halves = [f"{cell(off, lambda t, h=h: WORDING.get(t['id']) == h)} / {cell(on, lambda t, h=h: WORDING.get(t['id']) == h)}"
                          for h in ("as_dev", "new")]
            out.append(f"| {model} | {which} | {layout} | {machine} | {flag} | {cell(off)} | {cell(on)} | {halves[0]} | {halves[1]} |")
    out += ["", "## 3. Held-out set: the stock server against the runtime, every configuration (not timed)", "",
            "One run each on the first 5090's GPU, 100 tasks. Tasks correct under the current rule, stricter rule in "
            "brackets; then the two wording halves, 50 tasks each.", ""]
    for place in sorted(untimed):
        model, which, layout, machine, device = place
        if which != "held-out" or device != "GPU" or not any(k[0] == "server" for k in untimed[place]):
            continue
        out += [f"### {model} — held-out set, tool list in the {layout} message — {machine}, {device}", "",
                "| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |",
                "|---|---|---|---|---|"]
        for who, mode, cache, forced, name in ORDER:
            runs = untimed[place].get((who, mode, cache, forced), [])
            if not runs:
                out.append(f"| {name} | not run | | | |")
                continue
            out.append(f"| {name} | {cell(runs)} | {cell(runs, lambda t: WORDING.get(t['id']) == 'as_dev')} | "
                       f"{cell(runs, lambda t: WORDING.get(t['id']) == 'new')} | "
                       f"{span([r['summary'].get('ended_with_an_answer', 0) for r in runs])} |")
        out.append("")
    (RESULTS / "HEADLINE.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
