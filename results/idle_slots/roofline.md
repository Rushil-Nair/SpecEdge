# Roofline for the idle-slot runs

Qwen3-8B Q4_K_M on an RTX 5090 (1,792 GB/s). Produced by `bench/idle_slots/roofline.py`.

Weights read per forward pass: 4.68 GB (file size minus the token-embedding table, assumed Q4_K). Counting the whole file instead: 5.03 GB.

## 1 user: measured speed against the bandwidth limit

| Context tokens | Limit (tokens/s) | Measured 228.3 tokens/s as % of limit |
|---|---|---|
| 0 | 383 | 59.6% |
| 512 | 377 | 60.6% |
| 1024 | 371 | 61.5% |
| 2048 | 360 | 63.4% |

Whole file counted, no KV cache: limit 356 tokens/s, measured = 64.1%.

These are estimates from bytes moved and the rated bandwidth, not profiler measurements.

## Why split forward passes cost so much

Each forward pass reads the 4.68 GB of weights again, whatever the batch. Before the fix the counter showed 4.34 passes per decode step, after it 1.05: 4.1x fewer weight reads per step. The measured 8-user speedup was 2.63x, lower than the pass ratio because per-sequence work (attention over each request's KV cache, sampling, server overhead) is not repeated per pass.

Caveat: the pass counts come from the CPU check on master 7182ef71, the speeds from the RTX 5090 runs on 3d65c90d.
