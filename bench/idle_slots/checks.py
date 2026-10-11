"""Idle-slot fix: output-match check and prompt-cache check. Runs on the rented machine.

    python bench/idle_slots/checks.py match --server <llama-server> --label pinned --out <dir>
    python bench/idle_slots/checks.py cache --server <llama-server> --label pinned --out <dir>
    python bench/idle_slots/checks.py compare --out <dir>

Both start one llama-server with 16 slots (Qwen3-8B Q4_K_M, GPU, 4,096 tokens per slot, the
default per-slot KV cache) and talk to it directly.

match   The 30 Chinese tuning prompts of part 10 (in no test set), greedy, thinking off, up to
        192 tokens: first one request at a time, then four at a time. The generated token ids
        are saved. `compare` counts prompts whose ids are identical between the two servers.
        With one request at a time the patch cannot change the arithmetic, so the outputs must
        be identical. With four at a time, which requests share a forward pass differs between
        the servers, and llama.cpp's output is known to depend on that, so a match rate is
        reported there.

cache   A workload with repeated prefixes, built to show what least-recently-used selection is
        for. Six conversations run at the same time; each has its own long opening text and
        eight turns, every turn resending the whole conversation so far (cache_prompt on), with
        a 0.5 s pause between turns. Six more clients send unrelated one-off requests, each
        after a random wait of up to a second, so the number in flight keeps changing. While a conversation pauses, its slot is free: least-recently-used leaves it
        alone, the lowest-free-slot rule may hand it to a one-off request and throw its cache
        away. IDLE_SERVER_EXTRA="--cache-ram 0" turns off the server's host-memory prompt
        cache, which otherwise saves a slot's state before the slot is reused. Recorded per request, from the server's own timings: prompt tokens taken from the
        cache (cache_n) and prompt tokens processed (prompt_n). Two passes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import first_hour as fh  # noqa: E402

QWEN = "/workspace/models/gguf/Qwen3-8B-Q4_K_M.gguf"
SLOTS = 16
ONE_OFF_CLIENTS = 6


def start(server: str, log: Path):
    args = argparse.Namespace(context=4096 * SLOTS, slots=SLOTS, cpu=False, threads=None, head=None, draft_max=0, draft_p_min=0.0,
                              target=QWEN, spec_type="draft-eagle3", server_bin=server, extra=os.environ.get("IDLE_SERVER_EXTRA"))
    return fh.start_server(args, log)


def lines(kind: str) -> list[dict]:
    return [json.loads(line) for line in (fh.PROMPTS / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]


def match(server: str, label: str, out: Path) -> None:
    prompts = lines("chinese_tune")[:30]
    process = start(server, out / f"match_{label}.server.log")
    record = {"label": label, "server": server, "slots": SLOTS, "prompts": len(prompts), "runs": {}}
    try:
        fh.generate({"id": "warmup", "messages": [{"role": "user", "content": "Hello"}]}, 16)
        for users in (1, 4):
            with ThreadPoolExecutor(users) as pool:
                results = list(pool.map(lambda r: fh.generate(r, 192), prompts))
            record["runs"][str(users)] = {r["id"]: r["tokens"] for r in results}
    finally:
        process.terminate()
        process.wait(timeout=60)
    (out / f"match_{label}.json").write_text(json.dumps(record))
    print(f"match {label}: saved {len(prompts)} prompts at 1 and 4 requests at a time", flush=True)


def chat(messages: list[dict], max_tokens: int) -> dict:
    return fh.post("/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "seed": 0,
                                            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})


def cache(server: str, label: str, out: Path) -> None:
    chat_prompts = [p["messages"][0]["content"] for p in lines("chat")]
    code_prompts = [p["messages"][0]["content"] for p in lines("code")]
    oneoff = [p["messages"][0]["content"] for p in lines("reasoning")] + [p["messages"][0]["content"] for p in lines("chinese")]
    process = start(server, out / f"cache_{label}.server.log")
    rows, lock, stop = [], threading.Lock(), threading.Event()

    def note(kind: str, who: int, turn: int, reply: dict, seconds: float) -> None:
        t = reply["timings"]
        with lock:
            rows.append({"kind": kind, "client": who, "turn": turn, "cache_n": t.get("cache_n", 0), "prompt_n": t["prompt_n"],
                         "predicted_n": t["predicted_n"], "seconds": seconds})

    def conversation(number: int, repeat: int) -> None:
        opening = " ".join(chat_prompts[(number * 7 + i) % len(chat_prompts)] for i in range(12))
        messages = [{"role": "system", "content": f"Session {repeat}-{number}. Background notes for this session: {opening}"}]
        for turn in range(8):
            messages.append({"role": "user", "content": code_prompts[(number * 11 + turn + repeat * 31) % len(code_prompts)]})
            begin = time.time()
            reply = chat(messages, 48)
            note("conversation", number, turn, reply, time.time() - begin)
            messages.append({"role": "assistant", "content": reply["choices"][0]["message"]["content"]})
            time.sleep(0.5)

    def oneoffs(number: int) -> None:
        index, rng = number * 30, random.Random(number)
        while not stop.is_set():
            time.sleep(rng.uniform(0.0, 1.0))                  # so the number of one-off requests in flight keeps changing
            begin = time.time()
            reply = chat([{"role": "user", "content": oneoff[index % len(oneoff)]}], 48)
            note("one-off", number, index, reply, time.time() - begin)
            index += 1

    begin = time.time()
    try:
        chat([{"role": "user", "content": "Hello"}], 8)
        for repeat in range(2):
            stop.clear()
            extra = [threading.Thread(target=oneoffs, args=(n,)) for n in range(ONE_OFF_CLIENTS)]
            with ThreadPoolExecutor(6) as pool:
                futures = [pool.submit(conversation, n, repeat) for n in range(6)]
                time.sleep(0.3)                                # the conversations take their slots first
                for thread in extra:
                    thread.start()
                for future in futures:
                    future.result()
            stop.set()
            for thread in extra:
                thread.join()
    finally:
        process.terminate()
        process.wait(timeout=60)
    wall = time.time() - begin

    def total(kind: str, key: str, later: bool = False) -> int:
        return sum(r[key] for r in rows if r["kind"] == kind and (not later or r["turn"] > 0))

    conv = [r for r in rows if r["kind"] == "conversation"]
    later = [r for r in conv if r["turn"] > 0]
    summary = {
        "label": label, "server": server, "slots": SLOTS, "one_off_clients": ONE_OFF_CLIENTS, "wall_seconds": round(wall, 1),
        "mean_seconds_per_conversation_turn_after_the_first": round(sum(r["seconds"] for r in later) / max(len(later), 1), 3),
        "mean_seconds_per_one_off_request": round(sum(r["seconds"] for r in rows if r["kind"] == "one-off") / max(len(rows) - len(conv), 1), 3),
        "conversation_requests": len(conv), "one_off_requests": len(rows) - len(conv),
        "conversation_turns_after_the_first": len(later),
        "of_those_with_no_cache_hit": sum(r["cache_n"] == 0 for r in later),
        "prompt_tokens_from_cache_turns_after_the_first": total("conversation", "cache_n", True),
        "prompt_tokens_processed_turns_after_the_first": total("conversation", "prompt_n", True),
        "share_from_cache_turns_after_the_first": round(total("conversation", "cache_n", True) / max(total("conversation", "cache_n", True) + total("conversation", "prompt_n", True), 1), 4),
        "mean_seconds_per_conversation_turn": round(sum(r["seconds"] for r in conv) / len(conv), 3),
        "tokens_written": sum(r["predicted_n"] for r in rows), "tokens_written_per_second_of_wall": round(sum(r["predicted_n"] for r in rows) / wall, 1),
    }
    (out / f"cache_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary, indent=1), flush=True)


def compare(out: Path) -> None:
    a, b = json.loads((out / "match_pinned.json").read_text()), json.loads((out / "match_patched.json").read_text())
    result = {}
    for users in ("1", "4"):
        same = sum(a["runs"][users][key] == b["runs"][users][key] for key in a["runs"][users])
        result[f"{users}_at_a_time_identical"] = f"{same} of {len(a['runs'][users])}"
    # the same server against itself: how much does four-at-a-time differ from one-at-a-time?
    for name, rec in (("pinned", a), ("patched", b)):
        same = sum(rec["runs"]["1"][key] == rec["runs"]["4"][key] for key in rec["runs"]["1"])
        result[f"{name}_one_against_four_at_a_time_identical"] = f"{same} of {len(rec['runs']['1'])}"
    (out / "match_compare.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["match", "cache", "compare"])
    parser.add_argument("--server")
    parser.add_argument("--label")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "match":
        match(args.server, args.label, out)
    elif args.step == "cache":
        cache(args.server, args.label, out)
    else:
        compare(out)
