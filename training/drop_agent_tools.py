"""Remove from a training file every example that touches the 100 agent tasks.

    python training/drop_agent_tools.py --agent <agent_tasks.json> --out <dir> <train.jsonl>

<agent_tasks.json> is a copy, made from agent/tasks without changing it,
of the six tool names of agent/tools/schemas and the 100 task questions of
agent/tasks/tasks_v1.jsonl, with the commit they were read from.

An example is dropped when its text
  - names one of the six tools as a tool ("name": "<tool>", however the quotes are escaped;
    the same pattern as agent/tasks/check_training_overlap.py), or
  - contains one of the 100 task questions word for word.
Because the tool list is part of every prompt of a conversation, this removes whole
conversations, not single turns. The kept examples are written in their original order to
<out>/train.jsonl, and <out>/agent_overlap_check.json records what was dropped and a second
count on the written file, which must be zero.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("train")
    args = parser.parse_args()
    agent = json.loads(Path(args.agent).read_text(encoding="utf-8"))
    patterns = {tool: re.compile(r'name\\*"\s*:\s*\\*"' + re.escape(tool) + r'\\*"') for tool in agent["tools"]}
    questions = agent["questions"]

    def reasons(line: str) -> list[str]:
        found = [tool for tool, pattern in patterns.items() if pattern.search(line)]
        text = json.loads(line)
        if any(q in text["prompt"] or q in text["completion"] for q in questions):
            found.append("task question")
        return found

    lines = [line for line in Path(args.train).read_text(encoding="utf-8").splitlines() if line]
    kept, dropped, by_source = [], collections.Counter(), collections.Counter()
    for line in lines:
        why = reasons(line)
        if why:
            for reason in why:
                dropped[reason] += 1
            by_source[json.loads(line)["source"].split("-")[0]] += 1
        else:
            kept.append(line)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    remaining = sum(bool(reasons(line)) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines() if line)
    layouts = collections.Counter(json.loads(line).get("layout", "user") for line in kept)
    kinds = collections.Counter(json.loads(line)["kind"] for line in kept)
    report = {"agent_tasks_from": agent["source"], "tools": agent["tools"], "task_questions": len(questions),
              "input_file": args.train, "input_examples": len(lines), "examples_dropped": len(lines) - len(kept),
              "dropped_naming_each_tool": {tool: dropped[tool] for tool in agent["tools"]},
              "dropped_containing_a_task_question": dropped["task question"],
              "dropped_by_source": dict(by_source), "examples_kept": len(kept),
              "kept_by_layout": dict(layouts), "kept_by_kind": dict(kinds),
              "examples_still_overlapping_after_writing": remaining}
    (out / "agent_overlap_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if remaining:
        raise SystemExit("the written file still overlaps the agent tasks")


if __name__ == "__main__":
    main()
