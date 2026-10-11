"""Is the runtime's prompt the same text, and the same tokens, as stock llama-server's?

    python check_prompt.py --model <gguf> --agent <specedge-agent> --tools <schemas dir> --contacts <file>

Runs on the rented machine, CPU only. Starts llama-server with the model, and for a set of
conversations compares:
  - the text from the server's /apply-template with the runtime's `render` (via toolhost),
    using the date the server put in its "Today Date" line;
  - the server's /tokenize of that text with the runtime's `tokenize`;
  - the number of prompt tokens /completion says it evaluated with the runtime's count.
Prints one line per conversation and exits 1 if any text differs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PORT = 8171
SERVER = "/workspace/llama.cpp/build/bin/llama-server"

CALL_1 = '{"name": "search_contacts", "parameters": {"query": "Priya"}}'
RESULT_1 = '{"contacts": [{"contact_id": "c_101", "name": "Priya Nair", "team": "design", "email": "priya.nair@example.com"}]}'
CALL_2 = ('{"name": "create_event", "parameters": {"title": "Design sync", "date": "2026-11-12", "start_time": "09:00", '
          '"duration_hours": 1.5, "attendee_ids": ["c_101"]}}')
TWO_CALLS = ('{"name": "cancel_event", "parameters": {"event_id": "e_101"}}; '
             '{"name": "cancel_event", "parameters": {"event_id": "e_102"}}')

CONVERSATIONS = {
    "system and user": [
        {"role": "system", "content": "You are an office assistant."},
        {"role": "user", "content": "Schedule a meeting with Priya."}],
    "no system message": [
        {"role": "user", "content": "Schedule a meeting with Priya."}],
    "spaces and non-ASCII": [
        {"role": "system", "content": "  You are an office assistant.\n"},
        {"role": "user", "content": "\n Book “Café réunion” for Zoë at 09:00  "}],
    "one call and its result": [
        {"role": "system", "content": "You are an office assistant."},
        {"role": "user", "content": "Schedule a meeting with Priya."},
        {"role": "assistant", "content": CALL_1},
        {"role": "tool", "content": RESULT_1}],
    "two rounds": [
        {"role": "system", "content": "You are an office assistant."},
        {"role": "user", "content": "Schedule a meeting with Priya."},
        {"role": "assistant", "content": CALL_1},
        {"role": "tool", "content": RESULT_1},
        {"role": "assistant", "content": CALL_2},
        {"role": "tool", "content": '{"ok": true, "event_id": "e_501"}'}],
    "two calls in one reply": [
        {"role": "user", "content": "Cancel both."},
        {"role": "assistant", "content": TWO_CALLS},
        {"role": "tool", "content": '{"ok": true, "cancelled": "e_101", "title": "A"}'},
        {"role": "tool", "content": '{"ok": true, "cancelled": "e_102", "title": "B"}'}],
    "non-ASCII in a tool result": [
        {"role": "user", "content": "Find Zo\u00eb."},
        {"role": "assistant", "content": '{"name": "search_contacts", "parameters": {"query": "Zo\u00eb"}}'},
        {"role": "tool", "content": '{"contacts": [{"name": "Zo\u00eb \u201cZ\u201d M\u00fcller", "note": "a\\\\b\\n"}]}'}],
    "a final answer in the history": [
        {"role": "user", "content": "When is my review?"},
        {"role": "assistant", "content": "It starts at 14:30."},
        {"role": "user", "content": "Thanks. And tomorrow?"}],
}


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def first_difference(a: str, b: str) -> str:
    at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at character {at}: server {a[max(0, at - 40):at + 60]!r} | runtime {b[max(0, at - 40):at + 60]!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--contacts", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args()

    host = subprocess.Popen([args.agent, "toolhost", "--tools", args.tools, "--contacts", args.contacts],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")

    def ask(request: dict) -> dict:
        host.stdin.write(json.dumps(request) + "\n")
        host.stdin.flush()
        return json.loads(host.stdout.readline())

    tools = ask({"op": "tools"})["tools"]
    log = open(Path(tempfile.gettempdir()) / "check_prompt.server.log", "w")
    server = subprocess.Popen([SERVER, "-m", args.model, "--port", str(PORT), "--host", "127.0.0.1", "-c", "4096",
                               "-np", "1", "-ngl", "0", "-t", str(args.threads), "--jinja"], stdout=log, stderr=log)
    rows, bad = [], 0
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:  # noqa: BLE001 - still loading
                time.sleep(1)
        # Each conversation with tool results is sent twice: with the result as a string (the
        # stock template then writes it as a quoted, escaped JSON string) and as an object.
        variants = []
        for name, messages in CONVERSATIONS.items():
            variants.append((name, messages, messages))
            if name in ("system and user", "no system message", "two rounds"):
                variants.append((name + " (tools in system)", messages, messages))
            if any(m["role"] == "tool" for m in messages):
                as_objects = [dict(m, content=json.loads(m["content"])) if m["role"] == "tool" else m for m in messages]
                variants.append((name + " (results as objects)", as_objects, messages))
        for name, sent, messages in variants:
            row = {"conversation": name}
            try:
                in_system = name.endswith("(tools in system)")
                kwargs = {"chat_template_kwargs": {"tools_in_user_message": False}} if in_system else {}
                served = post("/apply-template", {"messages": sent, "tools": tools, **kwargs})["prompt"]
            except Exception as err:  # noqa: BLE001 - the server refusing a conversation is itself a finding
                row.update(server_error=f"{type(err).__name__}: {err}")
                rows.append(row)
                print(f"{name:46s} server refused: {row['server_error']}")
                continue
            date = re.search(r"Today Date: ([^\n]*)\n", served).group(1)
            mine = ask({"op": "render", "messages": messages, "date": date,
                        "tools_in": "system" if in_system else "user"})["prompt"]
            # The server leaves <|begin_of_text|> out of the text and adds it when tokenizing.
            row["server_text_has_bos"] = served.startswith("<|begin_of_text|>")
            body = mine[len("<|begin_of_text|>"):] if not row["server_text_has_bos"] else mine
            row["same_text"] = served == body
            if served != body:
                bad += 1
                row["difference"] = first_difference(served, body)
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="") as handle:
                handle.write(mine)
            my_tokens = json.loads(subprocess.run([args.agent, "tokenize", "--model", args.model, "--file", handle.name],
                                                  capture_output=True, text=True, check=True).stdout)
            with_special = post("/tokenize", {"content": served, "add_special": True, "parse_special": True})["tokens"]
            without = post("/tokenize", {"content": served, "add_special": False, "parse_special": True})["tokens"]
            evaluated = post("/completion", {"prompt": served, "n_predict": 1, "temperature": 0.0,
                                             "cache_prompt": False})["tokens_evaluated"]
            row.update(runtime_tokens=len(my_tokens), server_tokens_add_special=len(with_special),
                       server_tokens_no_special=len(without), completion_tokens_evaluated=evaluated,
                       same_tokens_as_no_special=my_tokens == without, same_tokens_as_add_special=my_tokens == with_special)
            rows.append(row)
            print(f"{name:46s} text same: {row['same_text']}  tokens: runtime {len(my_tokens)}, /tokenize "
                  f"{len(with_special)} (add_special) {len(without)} (not), /completion evaluated {evaluated}")
            if "difference" in row:
                print("    " + row["difference"])
    finally:
        server.terminate()
        server.wait(timeout=60)
        host.stdin.close()
        host.wait(timeout=10)
    if args.out:
        Path(args.out).write_text(json.dumps({"model": Path(args.model).name, "rows": rows}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
