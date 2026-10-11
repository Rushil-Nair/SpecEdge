"""First-hour measurement runner. Runs on the rented machine.

    python bench/first_hour.py --name gpu_plain --kinds chat code
    python bench/first_hour.py --name gpu_angelslim --head models/gguf/Qwen3-8B_eagle3.gguf --kinds chat code
    python bench/first_hour.py --name cpu_plain --cpu --threads 7 --limit 10 --repeats 3 --kinds chat

One invocation = one configuration (target, optional draft head, GPU or CPU). It starts
its own llama-server, sends every prompt one at a time with greedy decoding, and writes
results/first_hour/<name>.json holding, per prompt, the generated token ids and the
server's own timing counters. Nothing is estimated: every figure in the file is read
back from the server.

Why token ids are recorded: the project's claim is that speculative decoding does not
change the output, and that has to be checked token for token against plain decoding.
Comparing text would hide differences that detokenise to the same string.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path("/workspace/llama.cpp/build/bin/llama-server")
TARGET = Path("/workspace/models/gguf/Qwen3-8B-Q4_K_M.gguf")
PROMPTS = ROOT / "data" / "first_hour"
# Another job may run its own server on this machine; SPECEDGE_PORT keeps them apart.
PORT = int(os.environ.get("SPECEDGE_PORT", "8090"))


def post(path: str, body: dict, timeout: float = 1200) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    command = [
        str(args.server_bin or SERVER), "-m", str(args.target or TARGET), "--port", str(PORT), "--host", "127.0.0.1",
        "-c", str(args.context), "-np", str(args.slots), "--seed", "0",
        "-ngl", "0" if args.cpu else "99",
        "--metrics",
    ]
    if args.threads:
        command += ["-t", str(args.threads)]
    if args.head:
        command += [
            "-md", args.head, "--spec-type", args.spec_type,
            "--spec-draft-n-max", str(args.draft_max),
            "--spec-draft-p-min", str(args.draft_p_min),
        ]
        if not args.cpu:
            command += ["-ngld", "99"]
    command += shlex.split(args.extra or "")
    environment = dict(os.environ)
    if args.cpu:
        # Hide the GPU entirely, so nothing can be offloaded by accident.
        environment["CUDA_VISIBLE_DEVICES"] = ""
    log = open(log_path, "w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)

    deadline = time.time() + 600
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return process
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    process.kill()
    raise RuntimeError("llama-server did not become healthy in 10 minutes")


def generate(request: dict, max_tokens: int) -> dict:
    """Render the chat template on the server, then run a raw completion on that text.

    Two steps instead of the chat endpoint, because the raw completion endpoint can
    return the generated token ids, which the lossless check needs.
    """
    template_body = {
        "messages": request["messages"],
        "chat_template_kwargs": {"enable_thinking": bool(request.get("thinking"))},
    }
    if request.get("tools"):
        template_body["tools"] = request["tools"]
    prompt = post("/apply-template", template_body)["prompt"]

    start = time.perf_counter()
    reply = post("/completion", {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,       # greedy: the most likely token every step
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,    # every prompt is processed from scratch, so timings are comparable
        "return_tokens": True,
    })
    wall = time.perf_counter() - start
    return {
        "id": request["id"],
        "tokens": reply.get("tokens"),
        "text": reply.get("content"),
        "timings": reply.get("timings"),
        "tokens_predicted": reply.get("tokens_predicted"),
        "tokens_evaluated": reply.get("tokens_evaluated"),
        "stop_type": reply.get("stop_type"),
        "wall_seconds": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--kinds", nargs="+", required=True)
    parser.add_argument("--head", help="GGUF draft head; omit for plain decoding")
    parser.add_argument("--spec-type", default="draft-eagle3",
                        help="draft-eagle3 for a draft head, draft-simple for a separate small draft model")
    parser.add_argument("--target", help="GGUF target; default is the 4-bit 8B model")
    parser.add_argument("--server-bin", help="llama-server binary; default is the pinned build")
    parser.add_argument("--extra", help="extra llama-server arguments, as one quoted string")
    parser.add_argument("--cpu", action="store_true", help="run with the GPU hidden")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--limit", type=int, default=100, help="prompts per kind")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-tokens-thinking", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--draft-max", type=int, default=8)
    parser.add_argument("--draft-p-min", type=float, default=0.5)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "first_hour"
    out_dir.mkdir(parents=True, exist_ok=True)
    server = start_server(args, out_dir / f"{args.name}.server.log")
    try:
        # One short throwaway generation so model loading and first-use setup are not timed.
        generate({"id": "warmup", "messages": [{"role": "user", "content": "Hello"}]}, 16)

        runs = []
        for repeat in range(args.repeats):
            for kind in args.kinds:
                lines = (PROMPTS / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()[: args.limit]
                for line in lines:
                    request = json.loads(line)
                    limit = args.max_tokens_thinking if request.get("thinking") else args.max_tokens
                    result = generate(request, limit)
                    result.update(kind=kind, repeat=repeat)
                    runs.append(result)
                print(f"repeat {repeat} {kind}: {len(lines)} prompts done", flush=True)

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10) as response:
                metrics_text = response.read().decode()
        except (urllib.error.URLError, OSError) as err:
            metrics_text = f"unavailable: {err}"
    finally:
        server.terminate()
        server.wait(timeout=60)

    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    record = {
        "name": args.name,
        "config": vars(args),
        "target": Path(args.target or TARGET).name,
        "llama_cpp_commit": Path("/workspace/llama_commit.txt").read_text().split()[0]
        if not args.server_bin else f"alternative build: {args.server_bin}",
        "hardware": {
            "cpu_model": cpu_model,
            "cpus_visible": os.cpu_count(),
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "platform": platform.platform(),
            "gpu": "hidden (CUDA_VISIBLE_DEVICES empty)" if args.cpu else subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip(),
        },
        "finished_unix": time.time(),
        "server_metrics": metrics_text,
        "runs": runs,
    }
    (out_dir / f"{args.name}.json").write_text(json.dumps(record))
    print(f"wrote {out_dir / (args.name + '.json')}")


if __name__ == "__main__":
    main()
