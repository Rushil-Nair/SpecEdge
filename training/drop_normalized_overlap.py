"""Stricter overlap check on a finished training file: tool names compared after normalising.

    python training/drop_normalized_overlap.py --agent <agent tools and questions .json> \
        [--agent-tasks <tasks.jsonl> ...] --out <dir> <train.jsonl>

Why: the earlier filters compare tool names by exact
spelling, so `auto_complete` in training and `autocomplete` in the test set were treated as
different tools. Here a name is lower-cased and its underscores, hyphens and spaces are
removed before comparing.

An example is dropped when
  - any tool name in its prompt or completion (every `"name": "<x>"`) equals, after
    normalising, the name of a tool of the frozen test set or one of the agent tasks' six tools; or
  - it contains a frozen test question or one of the agent-task questions (development and
    held-out sets), compared after lower-casing and removing punctuation, or word for word; or
  - its text contains the name or the contact id of one of the agent tasks' contacts (either set),
    searched as plain text the way agent/tasks/check_training_overlap.py does. That search also hits strings
    such as `wc_2023` for the id `c_202`; those examples are dropped too, so that the agent tasks'
    check reads zero on the finished file.
The tool list is part of every prompt of a conversation, so whole conversations go.

Writes <out>/train.jsonl (kept examples, original order) and
<out>/normalized_overlap_check.json: what was dropped, by reason and by source, every pair of
names that matched, and a second count on the written file, which must be zero.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from prepare import TEST, ROOT, question_key

NAME = re.compile(r'"name"\s*:\s*"([^"\\]{1,120})"')
USER_TURN = re.compile(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.S)


def norm(name: str) -> str:
    return re.sub(r"[_\-\s]", "", name.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", required=True)
    parser.add_argument("--agent-tasks", nargs="*", default=[], help="agent task files (each line has a 'user' field)")
    parser.add_argument("--agent-contacts", nargs="*", default=[], help="agent contact files (name and contact_id of each entry)")
    parser.add_argument("--out", required=True)
    parser.add_argument("train")
    args = parser.parse_args()

    test = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    test_tools = {tool["function"]["name"] for task in test for tool in task["tools"]}
    agent = json.loads(Path(args.agent).read_text(encoding="utf-8"))
    agent_questions = list(agent["questions"])
    for path in args.agent_tasks:
        agent_questions += [json.loads(line)["user"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    agent_questions = sorted(set(agent_questions))
    forbidden = {}                                        # normalised name -> (whose, the name as spelled there)
    for name in sorted(test_tools):
        forbidden.setdefault(norm(name), ("frozen test set", name))
    for name in agent["tools"]:
        forbidden[norm(name)] = ("agent", name)
    people = [value for path in args.agent_contacts for c in json.loads(Path(path).read_text(encoding="utf-8"))
              for value in (c["name"], c["contact_id"])]
    test_keys = {question_key(task["messages"][0]["content"]) for task in test}
    agent_keys = {question_key(q) for q in agent_questions}

    def reasons(line: str) -> tuple[list[str], list[tuple[str, str, str]]]:
        row = json.loads(line)
        text = row["prompt"] + "\n" + row["completion"]
        found, pairs = [], []
        for name in set(NAME.findall(text)):
            hit = forbidden.get(norm(name))
            if hit:
                pairs.append((name, hit[1], hit[0]))
        if any(whose == "frozen test set" for _, _, whose in pairs):
            found.append("tool name like a frozen-test tool")
        if any(whose == "agent" for _, _, whose in pairs):
            found.append("tool name like an agent tool")
        users = [question_key(u) for u in USER_TURN.findall(row["prompt"])]
        # the tool list may sit inside the first user turn; the question is then its last paragraph
        users += [question_key(u.rsplit("\n\n", 1)[-1]) for u in USER_TURN.findall(row["prompt"])]
        if any(u in test_keys for u in users):
            found.append("frozen test question")
        if any(u in agent_keys for u in users) or any(q in text for q in agent_questions):
            found.append("agent task question")
        if any(p in line for p in people):                 # plain text search, exactly as the agent tasks' own check does it
            found.append("agent contact name or id")
        return found, pairs

    lines = [line for line in Path(args.train).read_text(encoding="utf-8").splitlines() if line]
    kept, by_reason, by_source, matched = [], collections.Counter(), collections.Counter(), collections.Counter()
    for line in lines:
        why, pairs = reasons(line)
        if why:
            for reason in why:
                by_reason[reason] += 1
            for pair in pairs:
                matched[pair] += 1
            by_source[json.loads(line)["source"].split("-")[0]] += 1
        else:
            kept.append(line)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    written = [line for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    remaining = sum(bool(reasons(line)[0]) for line in written)
    exact = sum(1 for (train_name, other, _), _n in matched.items() if train_name == other)
    report = {
        "file_checked": args.train, "test_file": str(TEST.relative_to(ROOT)), "frozen_test_tools": len(test_tools),
        "agent_contact_files": args.agent_contacts, "agent_contact_names_and_ids": len(people),
        "agent_tools": agent["tools"], "agent_questions": len(agent_questions), "agent_task_files": args.agent_tasks,
        "normalisation": "lower case; underscores, hyphens and spaces removed",
        "examples_before": len(lines), "examples_dropped": len(lines) - len(kept), "examples_kept": len(kept),
        "dropped_by_reason": dict(by_reason), "dropped_by_source": dict(by_source),
        "distinct_name_pairs_matched": len(matched), "of_those_spelled_identically": exact,
        "name_pairs_matched": [{"in_training": a, "matches": b, "of": c, "examples": n} for (a, b, c), n in matched.most_common()],
        "kept_by_kind": dict(collections.Counter(json.loads(line)["kind"] for line in kept)),
        "kept_by_layout": dict(collections.Counter(json.loads(line).get("layout", "user") for line in kept)),
        "kept_from_recovery_generator": sum(json.loads(line)["source"].startswith("recovery-") for line in kept),
        "second_count_on_the_written_file": remaining,
    }
    (out / "normalized_overlap_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    brief = {k: v for k, v in report.items() if k != "name_pairs_matched"}
    print(json.dumps(brief, indent=1))
    print("first matched pairs:", report["name_pairs_matched"][:25])
    if remaining:
        raise SystemExit("the written file still has overlapping examples")


if __name__ == "__main__":
    main()
