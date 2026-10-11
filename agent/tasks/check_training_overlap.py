"""Overlap check for the multi-step tasks: do the fine-tuning files contain these tasks or their tools?

    python agent/tasks/check_training_overlap.py --out <json> <train.jsonl>...

For every training file (one JSON object per line, any shape) it counts the examples that
  - contain a task question word for word (development set and held-out set, 200 questions);
  - name one of the six tools of agent/tools as a tool ("name": "<tool>"), per tool;
  - mention one of the contact ids or people of either set's directory.
The check reads each line as text, so it does not depend on how the example is laid out.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=[str(HERE / "tasks_v1.jsonl"), str(HERE / "heldout_v1.jsonl")],
                        help="task files; by default the development set and the held-out set together")
    parser.add_argument("--contacts", nargs="+", default=[str(HERE.parent / "tools" / "contacts.json"),
                                                          str(HERE.parent / "tools" / "contacts_heldout.json")])
    parser.add_argument("--out", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    tasks = [json.loads(line) for path in args.tasks for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    questions = [task["user"] for task in tasks]
    tools = [json.loads(path.read_text(encoding="utf-8"))["function"]["name"]
             for path in sorted((HERE.parent / "tools" / "schemas").glob("*.json"))]
    contacts = [c for path in args.contacts for c in json.loads(Path(path).read_text(encoding="utf-8"))]
    people = [c["name"] for c in contacts] + [c["contact_id"] for c in contacts]
    # the tool's name as a name, however the line escapes its quotes: "name": "x", \"name\": \"x\"
    tool_patterns = {tool: re.compile(r'name\\*"\s*:\s*\\*"' + re.escape(tool) + r'\\*"') for tool in tools}

    report = {"tasks_file": args.tasks, "tasks": len(tasks), "tools": tools, "files": {}}
    for name in args.files:
        counts = {"examples": 0, "with_a_task_question": 0, "with_a_contact": 0, "naming_a_tool": {tool: 0 for tool in tools},
                  "naming_any_tool": 0}
        with open(name, encoding="utf-8") as handle:
            for line in handle:
                counts["examples"] += 1
                plain = line.replace('\\"', '"').replace("\\\\", "\\")
                counts["with_a_task_question"] += any(q in line or q in plain for q in questions)
                counts["with_a_contact"] += any(p in line for p in people)
                hit = False
                for tool, pattern in tool_patterns.items():
                    if pattern.search(line):
                        counts["naming_a_tool"][tool] += 1
                        hit = True
                counts["naming_any_tool"] += hit
        report["files"][name] = counts
        print(name, json.dumps(counts))
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
