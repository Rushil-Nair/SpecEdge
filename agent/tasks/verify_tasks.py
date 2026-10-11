"""Prove every task can be solved and that the checker tells solved from not solved.

    python agent/tasks/verify_tasks.py --agent <specedge-agent> --tasks agent/tasks/tasks_v1.jsonl

No model is involved. For every task, through the runtime's own tools and checker (toolhost):
  1. the reference calls run without a tool error and the checker says correct;
  2. doing nothing is not correct;
  3. the reference calls with every decimal number cut at the decimal point (12.75 -> 12,
     the base model's habit) are not correct, for tasks that have a decimal argument;
  4. the reference calls followed by one extra message are not correct.
Exits 1 if any of these fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--tools", default=str(Path(__file__).parent.parent / "tools" / "schemas"))
    parser.add_argument("--contacts", default=str(Path(__file__).parent.parent / "tools" / "contacts.json"))
    args = parser.parse_args()

    host = subprocess.Popen([args.agent, "toolhost", "--tools", args.tools, "--contacts", args.contacts],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")

    def ask(request: dict) -> dict:
        host.stdin.write(json.dumps(request) + "\n")
        host.stdin.flush()
        return json.loads(host.stdout.readline())

    def play(task: dict, calls: list[dict], answer: str) -> tuple[bool, list[str]]:
        ask({"op": "reset", "initial": task["initial"]})
        errors = []
        for call in calls:
            result = ask({"op": "call", "name": call["name"], "arguments": call["arguments"]})["result"]
            if "error" in result:
                errors.append(f'{call["name"]}: {result["error"]}')
        verdict = ask({"op": "check", "expect": task["expect"], "final_answer": answer})
        return verdict["correct"], errors

    def cut_decimals(calls: list[dict]) -> tuple[list[dict], bool]:
        cut, changed = copy.deepcopy(calls), False
        for call in cut:
            for key, value in call["arguments"].items():
                if isinstance(value, float) and value != int(value):
                    call["arguments"][key] = int(value) if int(value) > 0 else 1
                    changed = True
        return cut, changed

    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    someone = json.loads(Path(args.contacts).read_text(encoding="utf-8"))[0]["contact_id"]     # a real contact of this set
    problems, with_decimals = [], 0
    for task in tasks:
        calls, answer = task["reference_calls"], task["reference_answer"]
        solved, errors = play(task, calls, answer)
        if not solved or errors:
            problems.append(f'{task["id"]}: reference solution not accepted {errors}')
        if play(task, [], "")[0]:
            problems.append(f'{task["id"]}: doing nothing counts as correct')
        cut, changed = cut_decimals(calls)
        if changed:
            with_decimals += 1
            if play(task, cut, answer)[0]:
                problems.append(f'{task["id"]}: cut decimals count as correct')
        extra = calls + [{"name": "send_message", "arguments": {"contact_id": someone, "text": "extra"}}]
        solved_with_extra, extra_errors = play(task, extra, answer)
        if extra_errors:
            problems.append(f'{task["id"]}: the extra message could not be sent {extra_errors}')
        if solved_with_extra:
            problems.append(f'{task["id"]}: an extra message counts as correct')
    host.stdin.close()
    host.wait(timeout=10)

    for problem in problems:
        print("PROBLEM", problem)
    print(f"{len(tasks)} tasks; {with_decimals} have a decimal argument; {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
