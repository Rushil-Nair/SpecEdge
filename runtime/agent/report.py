"""Build the part 6 table from the results files.

    python runtime/agent/report.py [results dir]        # default runtime/results/part6

Every row is one configuration on one model and device. Runs of the same configuration are
repeats. Correctness is shown for every run; seconds are shown only for runs marked
timing_valid (a booked, quiet machine), as the median over repeats with the lowest and
highest repeat. Writes REPORT.md beside the files and prints it.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from strict import strict_correct


def label(record: dict) -> tuple:
    config = record["config"]
    if record["program"].startswith("specedge-agent"):
        who = "runtime"
    else:
        who = "stock server, " + ("chat endpoint" if config.get("mode") == "chat" else "raw text + runtime's parser")
    # the two rented RTX 5090 machines are told apart by their CPU limit; rows from different
    # machines are never counted as repeats of one another
    machine = {"1536000 100000": "first 5090", "768000 100000": "second 5090"}.get(
        record.get("hardware", {}).get("cpu_quota", ""), "machine not recognised")
    device = (f"{machine}, GPU" if config.get("n_gpu_layers", 0) else f'{machine}, CPU, {config.get("threads")} threads')
    # the two task sets are never put in one table
    which = "held-out set" if "heldout" in record.get("tasks_file", "") else "development set"
    return (record["model"]["file"], device, f'{which}, tasks {record.get("tasks_sha256", "")[:8]}', who,
            "on" if config.get("cache") else "off", "on" if config.get("forced_json") else "off",
            config.get("tool_results", "stock"), bool(record.get("timing_valid")), config.get("tools_in", "user"))


def held_out_wording() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "agent" / "tasks" / "heldout_v1.jsonl"
    if not path.exists():
        return {}
    return {t["id"]: t["wording"] for t in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


WORDING = held_out_wording()


def spread(values: list[float], digits: int = 2) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (1 run)"
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})"


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "part6"
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("part") == 6 and "summary" in record:
            groups[label(record)].append(record)

    lines = ["# Part 6 — agent runtime against the stock llama.cpp server", "",
             "Built by `runtime/agent/report.py` from the files in this folder. Seconds are per task, mean over the "
             "tasks of a run, then median over repeats (lowest-highest). Rows without seconds are functional runs on "
             "a machine that was not booked for timing. The development set and the held-out set are never in one table, "
             "and neither are the two machines. In held-out tables the last two columns give the tasks correct in each "
             "wording half under the current rule, with the stricter rule in brackets.", ""]
    sections = sorted({key[:3] for key in groups})
    for model, device, tasks_sha in sections:
        lines += [f"## {model}, {device} ({tasks_sha})", "",
                  "| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | "
                  "Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        held_out = tasks_sha.startswith("held-out")
        if held_out:           # the two wording halves, 50 tasks each: current rule (stricter rule)
            lines[-2] += " Same wording as the development set, of 50 | New wording, of 50 |"
            lines[-1] += "---|---|"
        for key in sorted((k for k in groups if k[:3] == (model, device, tasks_sha)), key=lambda k: (k[8], k[3:8])):
            runs = groups[key]
            correct = [r["summary"]["correct"] for r in runs]
            tasks = runs[0]["summary"]["tasks"]
            shown = f"{correct[0]} of {tasks}" if len(set(correct)) == 1 else f"{min(correct)}-{max(correct)} of {tasks}"
            # re-scored from the saved final answers, so files written before the rule existed are covered
            strict = sorted({sum(strict_correct(t) for t in r["tasks"]) for r in runs})
            strict_shown = f"{strict[0]} of {tasks}" if len(strict) == 1 else f"{strict[0]}-{strict[-1]} of {tasks}"
            seconds = [[r["summary"]["seconds_per_task"][k] for r in runs] if key[7] else []
                       for k in ("wall_mean", "prompt_mean", "generation_mean")]
            def counts(name: str) -> str:
                values = sorted({r["summary"].get(name) for r in runs if r["summary"].get(name) is not None})
                return "" if not values else str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}"

            lines.append(f"| {key[8]} message | {key[3]} | {key[4]} | {key[5]} | {key[6]} | {'yes' if key[7] else 'no'} | "
                         f"{len(runs)} | {shown} | {strict_shown} | {counts('ended_with_an_answer')} | "
                         f"{counts('reached_expected_state_at_some_turn')} | "
                         f"{spread(seconds[0])} | {spread(seconds[1])} | {spread(seconds[2])} |")
            if held_out:
                for half in ("as_dev", "new"):
                    now = sorted({sum(bool(t["correct"]) for t in r["tasks"] if WORDING.get(t["id"]) == half) for r in runs})
                    hard = sorted({sum(strict_correct(t) for t in r["tasks"] if WORDING.get(t["id"]) == half) for r in runs})
                    one = lambda v: str(v[0]) if len(v) == 1 else f"{v[0]}-{v[-1]}"  # noqa: E731
                    lines[-1] += f" {one(now)} ({one(hard)}) |"
        lines.append("")
    text = "\n".join(lines)
    (folder / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
