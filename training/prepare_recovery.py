"""Add error-recovery conversations to the fine-tuning file (third run of part 2).

    python training/prepare_recovery.py --base-train /workspace/data/train_v2_clean/train.jsonl \
        --agent results/part2/agent_tasks_tools_and_questions.json --out /workspace/data/train_v3_recovery

The conversations are written by training/recovery_world.py (read its header for why and
what): 3,000 conversations on 12 topics, each with its own tools backed by a small simulated
world. No outside dataset and no model-written text. Two more topics (tool hire, parking) are
never written into the training file; bench/recovery_eval.py runs them live.

Rendering is the same as in prepare_multistep.py: the base model's own chat template, one
training example per assistant turn, the prompt being the conversation up to that turn, the
tool list in the user or the system message at random. A call that failed on a made-up id
appears only inside prompts: no example has it as its completion, so the model is shown how
to go on after the mistake, not how to make it.

Checks, written to <out>/overlap_check.json; the script fails if any is not zero:
  - no tool of the new conversations has the name of a frozen-test tool or of one of the
    B's six tools, and none is a held-out topic's tool
  - no user message is a frozen-test question or one of the agent-task questions
  - the finished file, searched as text, names no agent-task tool and no held-out tool
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import random
import re
from pathlib import Path

from prepare import TEST, ROOT, question_key
import recovery_world as rw

SEED, CONVERSATIONS = 2028, 3000


def name_pattern(tool: str) -> re.Pattern:
    return re.compile(r'name\\*"\s*:\s*\\*"' + re.escape(tool) + r'\\*"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--agent", required=True, help="the agent tasks' six tool names and 100 questions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default="/workspace/models/llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--conversations", type=int, default=CONVERSATIONS)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng, layout_rng = random.Random(SEED), random.Random(SEED + 1)
    test = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    test_tools = {tool["function"]["name"] for task in test for tool in task["tools"]}
    test_questions = {question_key(task["messages"][0]["content"]) for task in test}
    agent = json.loads(Path(args.agent).read_text(encoding="utf-8"))
    agent_questions = {question_key(q) for q in agent["questions"]}
    heldout_tools = {spec[0] for name in rw.HELDOUT_TOPICS for spec in rw.TOPICS[name]["tools"].values()}
    train_tools = {spec[0] for name in rw.TRAIN_TOPICS for spec in rw.TOPICS[name]["tools"].values()}

    conversations = rw.conversations(rw.TRAIN_TOPICS, args.conversations, SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    start = datetime.date(2024, 1, 1)
    examples, kinds, layouts, by_pattern, by_topic = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    bad_questions = 0
    for number, conversation in enumerate(conversations):
        tools = list(conversation["tools"])
        if rng.random() < 0.4:                                      # some tools of another topic that the task does not need
            other = rw.World(rng.choice([t for t in rw.TRAIN_TOPICS if t != conversation["topic"]]), random.Random(rng.random())).tools()
            tools += rng.sample(other, rng.randrange(1, 4))
        rng.shuffle(tools)
        messages = conversation["messages"]
        bad_questions += sum(m["role"] == "user" and (question_key(m["content"]) in test_questions or question_key(m["content"]) in agent_questions)
                             for m in messages)
        date = (start + datetime.timedelta(days=rng.randrange(1100))).strftime("%d %b %Y")
        by_pattern[conversation["pattern"]] += 1
        by_topic[conversation["topic"]] += 1
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or index in conversation["context_only"]:
                continue
            after_result = messages[index - 1]["role"] == "tool"
            if message.get("tool_calls"):
                function = message["tool_calls"][0]["function"]
                completion = json.dumps({"name": function["name"], "parameters": function["arguments"]}, ensure_ascii=False)
                kind = "call_after_tool_result" if after_result else "call"
            else:
                completion, kind = message["content"], "text_after_tool_result" if after_result else "text"
            layout = layout_rng.choice(["user", "system"])
            prompt = tokenizer.apply_chat_template(messages[:index], tools=tools, tokenize=False, add_generation_prompt=True,
                                                   date_string=date, tools_in_user_message=(layout == "user"))
            kinds[kind] += 1
            layouts[layout] += 1
            examples.append({"prompt": prompt, "completion": completion + "<|eot_id|>", "kind": kind,
                             "source": f"recovery-{conversation['topic']}-{conversation['pattern']}-{number}-turn{index}", "layout": layout})

    first = [line for line in Path(args.base_train).read_text(encoding="utf-8").splitlines() if line]
    new_lines = [json.dumps(e, ensure_ascii=False) for e in examples]
    lines = first + new_lines
    random.Random(SEED).shuffle(lines)
    (out / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    written = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    forbidden = {tool: name_pattern(tool) for tool in list(agent["tools"]) + sorted(heldout_tools)}
    in_file = {tool: sum(bool(pattern.search(line)) for line in written) for tool, pattern in forbidden.items()}
    lengths = [len(tokenizer(e["prompt"] + e["completion"], add_special_tokens=False)["input_ids"]) for e in rng.sample(examples, min(500, len(examples)))]
    check = {
        "test_file": str(TEST.relative_to(ROOT)), "test_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(),
        "agent_file": args.agent, "agent_commit": agent["source"]["commit"],
        "new_tools": len(train_tools),
        "new_tools_named_like_a_frozen_test_tool": sorted(train_tools & test_tools),
        "new_tools_named_like_an_agent_tool": sorted(train_tools & set(agent["tools"])),
        "new_tools_that_are_held_out_tools": sorted(train_tools & heldout_tools),
        "held_out_tools_named_like_a_test_or_agent_tool": sorted(heldout_tools & (test_tools | set(agent["tools"]))),
        "new_user_messages_equal_to_a_test_or_agent_question": bad_questions,
        "lines_of_the_finished_file_naming_a_agent_or_held_out_tool": {k: v for k, v in in_file.items() if v},
        "first_file": args.base_train, "first_file_examples": len(first), "examples_added": len(examples), "examples_total": len(lines),
    }
    (out / "overlap_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    stats = {"seed": SEED, "conversations": len(conversations), "conversations_by_pattern": dict(by_pattern),
             "conversations_by_topic": dict(by_topic), "examples_added_by_kind": dict(kinds), "layout_of_added_examples": dict(layouts),
             "examples_total": len(lines), "share_of_new_examples": round(len(examples) / len(lines), 3),
             "tokens_per_added_example_sample_of_500": {"median": sorted(lengths)[len(lengths) // 2], "max": max(lengths)},
             "calls_shown_only_as_context": sum(len(c["context_only"]) for c in conversations)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(check, indent=1))
    print(json.dumps(stats, indent=1))
    sample = next(e for e in examples if "recover_create" in e["source"] and e["kind"] == "call_after_tool_result")
    print("SAMPLE PROMPT:\n" + sample["prompt"])
    print("SAMPLE COMPLETION:", sample["completion"])
    failed = (check["new_tools_named_like_a_frozen_test_tool"] or check["new_tools_named_like_an_agent_tool"]
              or check["new_tools_that_are_held_out_tools"] or check["held_out_tools_named_like_a_test_or_agent_tool"]
              or bad_questions or check["lines_of_the_finished_file_naming_a_agent_or_held_out_tool"])
    if failed:
        raise SystemExit("the new conversations overlap test data; see overlap_check.json")


if __name__ == "__main__":
    main()
