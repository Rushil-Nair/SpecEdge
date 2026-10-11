"""The stricter rule for a task's final answer, and helpers shared by the analysis scripts.

Current rule (kept): a task is correct when it ended with a reply that makes no tool call,
the world is in the expected state, the required tools were called, and the reply contains
the task's answer words where the task has any. Any text at all passes as the reply.

Stricter rule (added beside it): the same, and the final reply must be plain text written
to the user. It fails when the reply
  - is empty;
  - starts with "{" or "[";
  - is JSON for an object or a list; or
  - is a quoted JSON string whose content is an object or a list, which is how the prompt
    shows a tool result ("{\\"ok\\": true}").
So a model that ends by repeating or inventing a tool result no longer passes.

The same rule is in agent/tools/world.cpp (answer_is_text); this file re-scores result files
that were written before that existed.
"""

from __future__ import annotations

import json


def answer_is_text(answer: str | None) -> bool:
    text = (answer or "").strip()
    if not text or text[0] in "{[":
        return False
    try:
        value = json.loads(text)
    except ValueError:
        return True
    if isinstance(value, (dict, list)):
        return False
    if isinstance(value, str):
        inner = value.strip()
        if inner[:1] in ("{", "["):
            try:
                return not isinstance(json.loads(inner), (dict, list))
            except ValueError:
                return True
    return True


def strict_correct(task_row: dict) -> bool:
    return bool(task_row["correct"]) and answer_is_text(task_row.get("final_answer"))


def row_label(record: dict) -> str:
    config = record["config"]
    if record["program"].startswith("specedge-agent"):
        who = "runtime"
    else:
        who = "stock server, " + ("chat endpoint" if config.get("mode") == "chat" else "raw text")
    machine = {"1536000 100000": "first 5090", "768000 100000": "second 5090"}.get(
        record.get("hardware", {}).get("cpu_quota", ""), "machine not recognised")
    device = machine + (" GPU" if config.get("n_gpu_layers", 0) else " CPU")
    which = "held-out" if "heldout" in record.get("tasks_file", "") else "development"
    return (f'{which} | {record["model"]["file"].replace(".gguf", "")} | {device} | {config.get("tools_in", "user")} | {who} | '
            f'forced JSON {"on" if config.get("forced_json") else "off"}')
