"""Read the part 6 result files and write two reports beside them. No model, no machine.

    python runtime/agent/analyze_part6.py

FAILURES_mine_multistep.md   every failed task of the second fine-tune, sorted into one type
                             each, with counts per layout and two transcripts per type.
RESCORE.md                   every row scored under the current rule and the stricter rule
                             for the final answer (strict.py), and how many "correct" tasks
                             end by giving a tool result back as the answer.

How a failed task gets its type (the first that applies):
  1. asked the user after a tool error   the last tool turn had an error and the final reply asks
                                         the user for something
  2. gave up after a tool error          same, and the reply says it could not do it, without asking
  3. said it was done despite an error   same, and the reply reports success
  4. wrong tool                          no error at the end; it called a tool the reference
                                         solution does not use
  5. stopped before finishing            no error at the end; it made fewer calls of some tool
                                         than the reference solution needs
  6. wrong argument or wrong choice      no error at the end; the right kinds of calls, wrong content
  7. other
"Guessed an id" is counted beside the types, not as one of them, because it is the cause of
most tool errors: a call used a contact id or event id that no tool had returned.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from strict import answer_is_text, row_label, strict_correct

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "part6"
TASKS = {t["id"]: t for t in (json.loads(line) for line in
                              (HERE.parents[1] / "agent" / "tasks" / "tasks_v1.jsonl").read_text(encoding="utf-8").splitlines())}
TYPES = ["asked the user after a tool error", "gave up after a tool error", "said it was done despite an error",
         "wrong tool", "stopped before finishing", "wrong argument or wrong choice", "other"]
ASKS = re.compile(r"\?|please provide|could you|can you|would you|let me know", re.I)
FAILED_WORDS = re.compile(r"sorry|unable|couldn'?t|could not|cannot|can't|error|not able|failed|wasn'?t able|don'?t have", re.I)


def guessed_ids(task: dict) -> list[str]:
    """Ids used in a call that no earlier tool result of this task contained."""
    seen, guessed = "", []
    for turn in task["turns"]:
        for call in turn["calls"]:
            arguments = call["arguments"]
            values = [arguments.get("contact_id"), arguments.get("event_id")]
            attendees = arguments.get("attendee_ids")
            values += attendees if isinstance(attendees, list) else [attendees]
            for value in values:
                if value is not None and str(value) not in seen:
                    guessed.append(str(value))
        for call in turn["calls"]:                       # results of this turn are known from the next turn on
            seen += json.dumps(call["result"])
    return guessed


def failure_type(task: dict) -> str:
    call_turns = [turn for turn in task["turns"] if turn["calls"]]
    answer = task.get("final_answer") or ""
    last_had_error = bool(call_turns) and any("error" in call["result"] for call in call_turns[-1]["calls"])
    if task["ended"] != "answered":
        return "other"
    if last_had_error:
        if ASKS.search(answer):
            return TYPES[0]
        return TYPES[1] if FAILED_WORDS.search(answer) else TYPES[2]
    made = collections.Counter(call["name"] for turn in call_turns for call in turn["calls"] if "error" not in call["result"])
    needed = collections.Counter(call["name"] for call in TASKS[task["id"]]["reference_calls"])
    attempted = {call["name"] for turn in call_turns for call in turn["calls"]}
    if attempted - set(needed):
        return TYPES[3]
    if any(made[name] < count for name, count in needed.items()):
        return TYPES[4]
    return TYPES[5] if call_turns else "other"


def transcript(task: dict) -> list[str]:
    lines = [f'**`{task["id"]}`** — {TASKS[task["id"]]["user"]}', "", "```"]
    for turn in task["turns"]:
        text = turn["text"].replace("\n", " ")
        lines.append(f'{turn["turn"]} MODEL {text[:300]}{" ..." if len(text) > 300 else ""}')
        for call in turn["calls"]:
            result = json.dumps(call["result"], ensure_ascii=False)
            lines.append(f'  TOOL  {result[:160]}{" ..." if len(result) > 160 else ""}')
    verdict = ", ".join(key for key, value in task["verdict"].items() if key != "correct" and not value)
    return lines + ["```", f"Checker: not right: {verdict or 'nothing (did not end with an answer)'}.", ""]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def failures_report() -> None:
    out = ["# `mine_multistep` (second fine-tune, Q4_K_M): failed tasks by type", "",
           "Built by `runtime/agent/analyze_part6.py` from the result files in this folder. Runtime with cache reuse, "
           "RTX 5090, 100 tasks, one run per row, not timed. With forced JSON on and off the model wrote the same text "
           "on every turn of every task (checked below), so each layout is one set of 100 transcripts.", ""]
    runs = {}
    for layout in ("user", "system"):
        on = load(f"rt_cache_json_{layout}_mine_multistep_q4km_gpu_r1.json")
        off = load(f"rt_cache_{layout}_mine_multistep_q4km_gpu_r1.json")
        same = sum([u["text"] for u in a["turns"]] == [u["text"] for u in b["turns"]] for a, b in zip(on["tasks"], off["tasks"]))
        out.append(f"- Tool list in the {layout} message: {on['summary']['correct']} of 100 correct; "
                   f"{same} of 100 transcripts identical with forced JSON on and off.")
        runs[layout] = on
    out += ["", "## Counts", "", "| Type | User layout | System layout |", "|---|---|---|"]
    typed = {layout: collections.defaultdict(list) for layout in runs}
    for layout, run in runs.items():
        for task in run["tasks"]:
            if not task["correct"]:
                typed[layout][failure_type(task)].append(task)
    for kind in TYPES:
        out.append(f"| {kind} | {len(typed['user'][kind])} | {len(typed['system'][kind])} |")
    out.append(f"| **failed tasks** | **{sum(map(len, typed['user'].values()))}** | **{sum(map(len, typed['system'].values()))}** |")
    out += ["", "Beside the types:", "", "| | User layout | System layout |", "|---|---|---|"]
    for name, test in (("failed tasks in which the model guessed an id", lambda t: bool(guessed_ids(t))),
                       ("failed tasks with at least one tool error", lambda t: any("error" in c["result"] for u in t["turns"] for c in u["calls"])),
                       ("failed tasks that ended on the turn right after their first tool error",
                        lambda t: [i for i, u in enumerate(t["turns"]) if any("error" in c["result"] for c in u["calls"])][:1] == [len(t["turns"]) - 2]),
                       ("failed tasks that made no call to search_contacts or list_events at all",
                        lambda t: not any(c["name"] in ("search_contacts", "list_events") for u in t["turns"] for c in u["calls"]))):
        cells = [sum(test(t) for kind in typed[layout].values() for t in kind) for layout in ("user", "system")]
        out.append(f"| {name} | {cells[0]} | {cells[1]} |")
    out += ["", "By task family (failed of 10):", "", "| Family | User layout | System layout |", "|---|---|---|"]
    for family in sorted({t["family"] for t in runs["user"]["tasks"]}):
        cells = [sum(not t["correct"] for t in runs[layout]["tasks"] if t["family"] == family) for layout in ("user", "system")]
        out.append(f"| {family} | {cells[0]} | {cells[1]} |")
    out += ["", "## Two transcripts per type", "",
            "The first failed task of the type in each layout, two different tasks.", ""]
    for kind in TYPES:
        # one from each layout, and never the same task twice
        picks = typed["user"][kind][:1]
        picks += [t for t in typed["system"][kind] if t["id"] not in {p["id"] for p in picks}][:1]
        if len(picks) < 2:
            picks = (picks + [t for t in typed["user"][kind] + typed["system"][kind] if t not in picks])[:2]
        if not picks:
            continue
        out += [f"### {kind}", ""]
        for task in picks:
            layout = "user" if task in typed["user"][kind] else "system"
            out += [f"Layout: tool list in the {layout} message."] + transcript(task)
    (RESULTS / "FAILURES_mine_multistep.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("\n".join(out[:40]))


def rescore_report() -> None:
    out = ["# Every row under the current rule and the stricter rule for the final answer", "",
           "Built by `runtime/agent/analyze_part6.py`. The current rule accepts any final text once the world is in "
           "the expected state. The stricter rule (`runtime/agent/strict.py`) also needs the final reply to be plain "
           "text: not empty, not starting with `{` or `[`, not JSON for an object or list, and not a quoted JSON string "
           "holding one (the form a tool result has in the prompt). One line per result file, timed repeats included.", "",
           "| Set | Model | Device | Tool list in | Program | Forced JSON | Tasks | Correct, current rule | Correct, stricter rule | "
           "Correct only under the current rule | of those, the answer repeats a tool result of the task word for word | Answered tasks whose final answer is not plain text |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULTS.glob("*.json"))]
    records = [r for r in records if r.get("part") == 6 and "summary" in r]
    for record in sorted(records, key=row_label):
        tasks = record["tasks"]
        current = sum(bool(t["correct"]) for t in tasks)
        strict = sum(strict_correct(t) for t in tasks)
        answered = [t for t in tasks if t["ended"] == "answered"]
        not_text = sum(not answer_is_text(t.get("final_answer")) for t in answered)
        label = row_label(record).replace("forced JSON ", "")
        echoes = 0
        for task in tasks:
            if task["correct"] and not strict_correct(task):
                answer = task["final_answer"].strip()
                try:
                    inner = json.loads(answer)
                    inner = inner if isinstance(inner, str) else answer
                except ValueError:
                    inner = answer
                shown = {json.dumps(c["result"]) for u in task["turns"] for c in u["calls"]}
                shown |= {'{"error": "the tool call was not valid JSON"}' for u in task["turns"] if u["broken_call"]}
                echoes += inner.strip() in shown
        out.append(f"| {label} | {len(tasks)} | {current} | {strict} | {current - strict} | {echoes} | {not_text} |")
    (RESULTS / "RESCORE.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out[4:]))


if __name__ == "__main__":
    failures_report()
    rescore_report()
