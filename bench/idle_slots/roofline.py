"""Memory-bandwidth roofline for the idle-slot runs (Qwen3-8B Q4_K_M, RTX 5090).

Single-sequence decode reads every weight once per forward pass, so the best possible
speed is memory bandwidth / bytes read per token. This script computes that bound,
compares it with the measured 1-user speed, and shows what extra forward passes per
decode step cost. Writes results/idle_slots/roofline.md.
"""
from pathlib import Path

# Model: Qwen3-8B (config.json from Qwen/Qwen3-8B)
LAYERS, KV_HEADS, HEAD_DIM, HIDDEN, VOCAB = 36, 8, 128, 4096, 151936
GGUF_BYTES = 5_027_783_488          # Qwen/Qwen3-8B-GGUF, Qwen3-8B-Q4_K_M.gguf
EMBED_BPW = 4.5                     # assumption: token_embd stored as Q4_K (4.5 bits/weight)
KV_BYTES = 2                        # f16 K and V cache

# GPU: RTX 5090 memory bandwidth (NVIDIA spec)
BANDWIDTH = 1792e9                  # bytes/s

# Measured (results/idle_slots/, 16 slots, no --kv-unified)
TPS_1_USER = 228.3
TPS_8_BEFORE, TPS_8_AFTER = 333.6, 878.9
PASSES_BEFORE, PASSES_AFTER = 4.34, 1.05   # forward passes per decode step, diagnostic counter


def weight_bytes():
    # The token-embedding table is only indexed (one row per token), not read in full.
    embed = VOCAB * HIDDEN * EMBED_BPW / 8
    return GGUF_BYTES - embed


def kv_bytes(context):
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * KV_BYTES * context


def main():
    w = weight_bytes()
    rows = []
    for ctx in (0, 512, 1024, 2048):
        bound = BANDWIDTH / (w + kv_bytes(ctx))
        rows.append((ctx, bound, TPS_1_USER / bound))
    whole_file_bound = BANDWIDTH / GGUF_BYTES

    out = [
        "# Roofline for the idle-slot runs",
        "",
        "Qwen3-8B Q4_K_M on an RTX 5090 (1,792 GB/s). Produced by `bench/idle_slots/roofline.py`.",
        "",
        f"Weights read per forward pass: {w/1e9:.2f} GB (file size minus the token-embedding table, "
        f"assumed Q4_K). Counting the whole file instead: {GGUF_BYTES/1e9:.2f} GB.",
        "",
        "## 1 user: measured speed against the bandwidth limit",
        "",
        "| Context tokens | Limit (tokens/s) | Measured 228.3 tokens/s as % of limit |",
        "|---|---|---|",
    ]
    out += [f"| {c} | {b:.0f} | {p:.1%} |" for c, b, p in rows]
    out += [
        "",
        f"Whole file counted, no KV cache: limit {whole_file_bound:.0f} tokens/s, measured = "
        f"{TPS_1_USER / whole_file_bound:.1%}.",
        "",
        "These are estimates from bytes moved and the rated bandwidth, not profiler measurements.",
        "",
        "## Why split forward passes cost so much",
        "",
        f"Each forward pass reads the {w/1e9:.2f} GB of weights again, whatever the batch. Before the fix "
        f"the counter showed {PASSES_BEFORE} passes per decode step, after it {PASSES_AFTER}: "
        f"{PASSES_BEFORE/PASSES_AFTER:.1f}x fewer weight reads per step. The measured 8-user speedup was "
        f"{TPS_8_AFTER/TPS_8_BEFORE:.2f}x, lower than the pass ratio because per-sequence work "
        "(attention over each request's KV cache, sampling, server overhead) is not repeated per pass.",
        "",
        "Caveat: the pass counts come from the CPU check on master 7182ef71, the speeds from the "
        "RTX 5090 runs on 3d65c90d.",
    ]
    path = Path(__file__).resolve().parents[2] / "results" / "idle_slots" / "roofline.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
