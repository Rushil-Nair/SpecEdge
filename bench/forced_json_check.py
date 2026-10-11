"""Check bench/forced_json.py against a real server. Runs on the rented machine, CPU only.

    python bench/forced_json_check.py <gguf>

1. Every task of the frozen set: the server must accept the task's grammar.
2. The first 12 call tasks: generate with and without the grammar; a forced reply must be
   either plain text or start exactly with {"name": . Prints three pairs and any violation.
It uses its own port and the CPU, so it can run while the GPU is busy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import first_hour as fh
import forced_json
import toolcall_eval as te

fh.PORT = 8097
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    arguments = argparse.Namespace(context=4096, slots=1, cpu=True, threads=4, head=None, draft_max=0, draft_p_min=0.0,
                                   target=sys.argv[1], spec_type="draft-eagle3", server_bin=None, extra="--jinja")
    server = fh.start_server(arguments, Path("/tmp/forced_json_check.server.log"))
    rejected, violations, shown = [], 0, 0
    try:
        tasks = [json.loads(line) for line in (ROOT / "data/toolset/v1/toolset_v1.jsonl").read_text(encoding="utf-8").splitlines()]
        for task in tasks:
            try:
                fh.post("/completion", {"prompt": "Call a tool:", "n_predict": 2, "temperature": 0.0,
                                        "grammar": forced_json.grammar_for(task["tools"])})
            except Exception:  # noqa: BLE001
                rejected.append(task["id"])
        print("grammars rejected:", len(rejected), "of", len(tasks), rejected[:3])
        for task in [t for t in tasks if t["type"] == "call"][:12]:
            prompt = fh.post("/apply-template", {"messages": task["messages"], "tools": te.sendable(task["tools"])})["prompt"]
            body = {"prompt": prompt, "n_predict": 160, "temperature": 0.0, "top_k": 1}
            free = fh.post("/completion", body)["content"]
            forced = fh.post("/completion", {**body, "grammar": forced_json.grammar_for(task["tools"])})["content"]
            fine = not forced.lstrip().startswith("{") or forced.startswith('{"name": ')
            violations += not fine
            if shown < 3 or not fine:
                shown += 1
                print("ok      " if fine else "VIOLATES", "FORCED", forced[:170].replace("\n", "\\n"))
                print("         FREE  ", free[:170].replace("\n", "\\n"))
        print("forced replies that break the grammar:", violations, "of 12")
    finally:
        server.terminate()
        server.wait(timeout=60)


if __name__ == "__main__":
    main()
