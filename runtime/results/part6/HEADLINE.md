# Part 6 — summary tables

Built by `runtime/agent/headline.py` from the result files in this folder. All models are the 4-bit (Q4_K_M) files; llama.cpp `f498f864`; greedy decoding; at most 10 turns and 256 tokens per turn.

## 1. Timed: the stock llama.cpp server against the runtime

Machine booked for these runs alone, quiet check before each, 3 repeats. Seconds are per task (mean over the tasks of a run), shown as the median repeat with the lowest and highest. For the runtime, seconds are the whole task inside one program; for the stock server they include the HTTP calls of the Python loop that drives it. Prompt and generation are the engine's own times; the stock server's are its `timings`. "Raw text" rows read the server's text with the runtime's parser, so they differ from the runtime only in who runs the model.

### mine_multistep (second) — development set (100 tasks), tool list in the system message — first 5090, GPU

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.21 (0.21-0.21) | 21 | 21 | 70 |
| Stock server, raw text, no cache | 3 | 0.46 (0.46-0.46) | 0.16 (0.16-0.16) | 0.29 (0.29-0.29) | 21 | 21 | 100 |
| Stock server, raw text, its own prompt cache | 3 | 0.32 (0.32-0.33) | 0.02 (0.02-0.02) | 0.29 (0.29-0.29) | 22 | 22 | 100 |
| Runtime, nothing on | 3 | 0.48 (0.48-0.48) | 0.13 (0.13-0.13) | 0.35 (0.35-0.35) | 21 | 21 | 100 |
| Runtime, cache reuse | 3 | 0.34 (0.34-0.34) | 0.01 (0.01-0.01) | 0.33 (0.33-0.33) | 22 | 22 | 100 |
| Runtime, forced JSON | 3 | 0.48 (0.48-0.48) | 0.13 (0.13-0.13) | 0.35 (0.35-0.35) | 21 | 21 | 100 |
| Runtime, cache reuse and forced JSON | 3 | 0.34 (0.34-0.34) | 0.01 (0.01-0.01) | 0.33 (0.33-0.33) | 22 | 22 | 100 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 1.26x against the stock chat endpoint; 1.38x against the stock server without cache; 0.97x against the stock server with its cache.

### mine_multistep (second) — development set (100 tasks), tool list in the user message — first 5090, GPU

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 0.40 (0.40-0.40) | 0.12 (0.12-0.12) | 0.20 (0.20-0.20) | 27 | 27 | 70 |
| Stock server, raw text, no cache | 3 | 0.44 (0.44-0.44) | 0.15 (0.15-0.15) | 0.28 (0.28-0.28) | 27 | 27 | 100 |
| Stock server, raw text, its own prompt cache | 3 | 0.32 (0.32-0.32) | 0.02 (0.02-0.02) | 0.28 (0.28-0.28) | 29 | 29 | 100 |
| Runtime, nothing on | 3 | 0.45 (0.45-0.45) | 0.12 (0.12-0.12) | 0.33 (0.33-0.33) | 27 | 27 | 100 |
| Runtime, cache reuse | 3 | 0.33 (0.33-0.33) | 0.01 (0.01-0.01) | 0.32 (0.32-0.32) | 29 | 29 | 100 |
| Runtime, forced JSON | 3 | 0.45 (0.45-0.45) | 0.12 (0.12-0.12) | 0.33 (0.33-0.33) | 27 | 27 | 100 |
| Runtime, cache reuse and forced JSON | 3 | 0.33 (0.33-0.33) | 0.01 (0.01-0.01) | 0.32 (0.32-0.32) | 29 | 29 | 100 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 1.23x against the stock chat endpoint; 1.36x against the stock server without cache; 0.97x against the stock server with its cache.

### mine_recovery (third, final) — development set (100 tasks), tool list in the system message — first 5090, GPU

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 0.41 (0.41-0.41) | 0.15 (0.15-0.15) | 0.21 (0.21-0.21) | 55 | 55 | 91 |
| Stock server, raw text, no cache | 3 | 0.42 (0.42-0.42) | 0.16 (0.16-0.16) | 0.24 (0.24-0.24) | 55 | 55 | 100 |
| Stock server, raw text, its own prompt cache | 3 | 0.27 (0.27-0.27) | 0.02 (0.02-0.02) | 0.23 (0.23-0.23) | 53 | 53 | 100 |
| Runtime, nothing on | 3 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) | 55 | 55 | 100 |
| Runtime, cache reuse | 3 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) | 53 | 53 | 100 |
| Runtime, forced JSON | 3 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) | 55 | 55 | 100 |
| Runtime, cache reuse and forced JSON | 3 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) | 53 | 53 | 100 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 1.46x against the stock chat endpoint; 1.50x against the stock server without cache; 0.98x against the stock server with its cache.

### mine_recovery (third, final) — development set (100 tasks), tool list in the user message — first 5090, GPU

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 0.39 (0.39-0.39) | 0.15 (0.15-0.15) | 0.20 (0.20-0.20) | 41 | 41 | 90 |
| Stock server, raw text, no cache | 3 | 0.42 (0.42-0.42) | 0.17 (0.17-0.17) | 0.23 (0.23-0.23) | 49 | 49 | 100 |
| Stock server, raw text, its own prompt cache | 3 | 0.28 (0.28-0.28) | 0.02 (0.02-0.02) | 0.24 (0.24-0.24) | 50 | 50 | 100 |
| Runtime, nothing on | 3 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) | 49 | 49 | 100 |
| Runtime, cache reuse | 3 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) | 50 | 50 | 100 |
| Runtime, forced JSON | 3 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) | 49 | 49 | 100 |
| Runtime, cache reuse and forced JSON | 3 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) | 50 | 50 | 100 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 1.38x against the stock chat endpoint; 1.49x against the stock server without cache; 0.98x against the stock server with its cache.

### stock 3B — development set (20 tasks), tool list in the system message — first 5090, CPU, 7 threads

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 53.06 (52.82-53.18) | 37.03 (36.81-37.11) | 12.27 (12.26-12.30) | 2 | 2 | 5 |
| Stock server, raw text, no cache | 3 | 39.81 (39.60-39.91) | 30.64 (30.43-30.74) | 9.13 (9.13-9.15) | 4 | 4 | 19 |
| Stock server, raw text, its own prompt cache | 3 | 11.68 (11.67-11.70) | 2.29 (2.28-2.30) | 9.36 (9.35-9.36) | 4 | 4 | 18 |
| Runtime, nothing on | 3 | 40.75 (40.49-40.78) | 31.17 (30.92-31.22) | 9.50 (9.49-9.50) | 4 | 4 | 19 |
| Runtime, cache reuse | 3 | 11.69 (11.67-11.70) | 2.27 (2.26-2.28) | 9.41 (9.41-9.41) | 4 | 4 | 18 |
| Runtime, forced JSON | 3 | 33.83 (33.83-34.10) | 26.45 (26.43-26.70) | 7.34 (7.32-7.34) | 6 | 6 | 20 |
| Runtime, cache reuse and forced JSON | 3 | 12.11 (11.99-12.13) | 2.75 (2.59-2.75) | 9.38 (9.36-9.39) | 6 | 6 | 19 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 4.38x against the stock chat endpoint; 3.29x against the stock server without cache; 0.96x against the stock server with its cache.

### stock 3B — development set (100 tasks), tool list in the system message — first 5090, GPU

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 0.98 (0.98-0.98) | 0.29 (0.29-0.29) | 0.57 (0.57-0.57) | 13 | 13 | 27 |
| Stock server, raw text, no cache | 3 | 0.80 (0.80-0.81) | 0.28 (0.28-0.28) | 0.50 (0.50-0.50) | 25 | 24 | 92 |
| Stock server, raw text, its own prompt cache | 3 | 0.58 (0.58-0.58) | 0.03 (0.03-0.03) | 0.52 (0.52-0.52) | 24 | 23 | 93 |
| Runtime, nothing on | 3 | 0.82 (0.82-0.82) | 0.22 (0.22-0.22) | 0.59 (0.59-0.59) | 25 | 24 | 92 |
| Runtime, cache reuse | 3 | 0.60 (0.60-0.60) | 0.01 (0.01-0.01) | 0.59 (0.59-0.59) | 24 | 23 | 93 |
| Runtime, forced JSON | 3 | 0.70 (0.70-0.70) | 0.20 (0.20-0.20) | 0.49 (0.49-0.49) | 36 | 35 | 94 |
| Runtime, cache reuse and forced JSON | 3 | 0.46 (0.46-0.46) | 0.01 (0.01-0.01) | 0.45 (0.45-0.45) | 38 | 38 | 96 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 2.11x against the stock chat endpoint; 1.73x against the stock server without cache; 1.25x against the stock server with its cache.

### stock 3B — development set (100 tasks), tool list in the user message — first 5090, GPU

| Configuration | Repeats | Seconds per task | Prompt | Generation | Tasks correct | Correct, stricter rule | Ended with an answer |
|---|---|---|---|---|---|---|---|
| Stock server, chat endpoint | 3 | 1.31 (1.31-1.31) | 0.43 (0.43-0.43) | 0.74 (0.74-0.74) | 0 | 0 | 3 |
| Stock server, raw text, no cache | 3 | 1.63 (1.63-1.64) | 0.60 (0.60-0.60) | 0.94 (0.94-0.94) | 0 | 0 | 24 |
| Stock server, raw text, its own prompt cache | 3 | 1.13 (1.13-1.13) | 0.07 (0.07-0.07) | 0.97 (0.97-0.97) | 0 | 0 | 25 |
| Runtime, nothing on | 3 | 1.69 (1.69-1.69) | 0.51 (0.51-0.51) | 1.17 (1.17-1.17) | 0 | 0 | 19 |
| Runtime, cache reuse | 3 | 1.16 (1.16-1.16) | 0.02 (0.02-0.02) | 1.13 (1.13-1.13) | 0 | 0 | 21 |
| Runtime, forced JSON | 3 | 1.52 (1.52-1.52) | 0.48 (0.48-0.48) | 1.03 (1.03-1.03) | 0 | 0 | 34 |
| Runtime, cache reuse and forced JSON | 3 | 1.14 (1.14-1.14) | 0.02 (0.02-0.02) | 1.11 (1.11-1.11) | 0 | 0 | 30 |

Time of the other row divided by the time of the full runtime (cache reuse and forced JSON): 1.15x against the stock chat endpoint; 1.44x against the stock server without cache; 1.00x against the stock server with its cache.

## 2. Tasks correct, every model, both task sets

Runtime with cache reuse, 100 tasks, GPU. Each cell is tasks correct under the current rule, with the stricter answer rule in brackets. Rows marked timed are the 3 repeats of section 1 (they agreed unless a range is shown); the others are single runs on a machine that was not booked. The two task sets, the two machines and the two layouts are separate rows; nothing is added across them.

| Model | Task set | Tool list in | Machine | Timed | Forced JSON off | Forced JSON on | Held-out, same wording as development (of 50): off / on | Held-out, new wording (of 50): off / on |
|---|---|---|---|---|---|---|---|---|
| mine (first fine-tune) | development | system | first 5090 | no | 20 (11) | 9 (9) |  |  |
| mine (first fine-tune) | development | user | first 5090 | no | 0 (0) | 0 (0) |  |  |
| mine_multistep (second) | development | system | first 5090 | no | 22 (22) | 22 (22) |  |  |
| mine_multistep (second) | development | user | first 5090 | no | 29 (29) | 29 (29) |  |  |
| mine_multistep (second) | held-out | system | first 5090 | no | 21 (21) | 21 (21) | 13 (13) / 13 (13) | 8 (8) / 8 (8) |
| mine_multistep (second) | held-out | user | first 5090 | no | 21 (21) | 21 (21) | 12 (12) / 12 (12) | 9 (9) / 9 (9) |
| mine_recovery (third, final) | development | system | second 5090 | no | 53 (53) | 53 (53) |  |  |
| mine_recovery (third, final) | development | user | second 5090 | no | 50 (50) | 50 (50) |  |  |
| mine_recovery (third, final) | held-out | system | first 5090 | no | 49 (49) | 49 (49) | 26 (26) / 26 (26) | 23 (23) / 23 (23) |
| mine_recovery (third, final) | held-out | system | second 5090 | no | 49 (49) | 49 (49) | 26 (26) / 26 (26) | 23 (23) / 23 (23) |
| mine_recovery (third, final) | held-out | user | first 5090 | no | 50 (50) | 50 (50) | 25 (25) / 25 (25) | 25 (25) / 25 (25) |
| mine_recovery (third, final) | held-out | user | second 5090 | no | 50 (50) | 50 (50) | 25 (25) / 25 (25) | 25 (25) / 25 (25) |
| stock 3B | development | system | first 5090 | no | 24 (23) | 38 (38) |  |  |
| stock 3B | development | system | second 5090 | no | 24 (23) | 38 (38) |  |  |
| stock 3B | development | user | first 5090 | no | 0 (0) | 0 (0) |  |  |
| stock 3B | development | user | second 5090 | no | 0 (0) | 0 (0) |  |  |
| stock 3B | held-out | system | first 5090 | no | 31 (31) | 37 (35) | 18 (18) / 22 (21) | 13 (13) / 15 (14) |
| stock 3B | held-out | system | second 5090 | no | 31 (31) | 37 (35) | 18 (18) / 22 (21) | 13 (13) / 15 (14) |
| stock 3B | held-out | user | first 5090 | no | 1 (1) | 0 (0) | 0 (0) / 0 (0) | 1 (1) / 0 (0) |
| stock 3B | held-out | user | second 5090 | no | 1 (1) | 0 (0) | 0 (0) / 0 (0) | 1 (1) / 0 (0) |
| mine_multistep (second) | development | system | first 5090 | yes | 22 (22) | 22 (22) |  |  |
| mine_multistep (second) | development | user | first 5090 | yes | 29 (29) | 29 (29) |  |  |
| mine_recovery (third, final) | development | system | first 5090 | yes | 53 (53) | 53 (53) |  |  |
| mine_recovery (third, final) | development | user | first 5090 | yes | 50 (50) | 50 (50) |  |  |
| stock 3B | development | system | first 5090 | yes | 24 (23) | 38 (38) |  |  |
| stock 3B | development | user | first 5090 | yes | 0 (0) | 0 (0) |  |  |

## 3. Held-out set: the stock server against the runtime, every configuration (not timed)

One run each on the first 5090's GPU, 100 tasks. Tasks correct under the current rule, stricter rule in brackets; then the two wording halves, 50 tasks each.

### mine_multistep (second) — held-out set, tool list in the system message — first 5090, GPU

| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |
|---|---|---|---|---|
| Stock server, chat endpoint | 20 (20) | 13 (13) | 7 (7) | 72 |
| Stock server, raw text, no cache | 20 (20) | 13 (13) | 7 (7) | 100 |
| Stock server, raw text, its own prompt cache | 21 (21) | 13 (13) | 8 (8) | 100 |
| Runtime, nothing on | 20 (20) | 13 (13) | 7 (7) | 100 |
| Runtime, cache reuse | 21 (21) | 13 (13) | 8 (8) | 100 |
| Runtime, forced JSON | 20 (20) | 13 (13) | 7 (7) | 100 |
| Runtime, cache reuse and forced JSON | 21 (21) | 13 (13) | 8 (8) | 100 |

### mine_multistep (second) — held-out set, tool list in the user message — first 5090, GPU

| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |
|---|---|---|---|---|
| Stock server, chat endpoint | 21 (21) | 11 (11) | 10 (10) | 74 |
| Stock server, raw text, no cache | 21 (21) | 11 (11) | 10 (10) | 100 |
| Stock server, raw text, its own prompt cache | 21 (21) | 12 (12) | 9 (9) | 100 |
| Runtime, nothing on | 21 (21) | 11 (11) | 10 (10) | 100 |
| Runtime, cache reuse | 21 (21) | 12 (12) | 9 (9) | 100 |
| Runtime, forced JSON | 21 (21) | 11 (11) | 10 (10) | 100 |
| Runtime, cache reuse and forced JSON | 21 (21) | 12 (12) | 9 (9) | 100 |

### mine_recovery (third, final) — held-out set, tool list in the system message — first 5090, GPU

| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |
|---|---|---|---|---|
| Stock server, chat endpoint | 49 (49) | 28 (28) | 21 (21) | 96 |
| Stock server, raw text, no cache | 49 (49) | 28 (28) | 21 (21) | 100 |
| Stock server, raw text, its own prompt cache | 49 (49) | 26 (26) | 23 (23) | 100 |
| Runtime, nothing on | 49 (49) | 28 (28) | 21 (21) | 100 |
| Runtime, cache reuse | 49 (49) | 26 (26) | 23 (23) | 100 |
| Runtime, forced JSON | 49 (49) | 28 (28) | 21 (21) | 100 |
| Runtime, cache reuse and forced JSON | 49 (49) | 26 (26) | 23 (23) | 100 |

### mine_recovery (third, final) — held-out set, tool list in the user message — first 5090, GPU

| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |
|---|---|---|---|---|
| Stock server, chat endpoint | 44 (44) | 20 (20) | 24 (24) | 95 |
| Stock server, raw text, no cache | 48 (48) | 24 (24) | 24 (24) | 100 |
| Stock server, raw text, its own prompt cache | 50 (50) | 25 (25) | 25 (25) | 100 |
| Runtime, nothing on | 48 (48) | 24 (24) | 24 (24) | 100 |
| Runtime, cache reuse | 50 (50) | 25 (25) | 25 (25) | 100 |
| Runtime, forced JSON | 48 (48) | 24 (24) | 24 (24) | 100 |
| Runtime, cache reuse and forced JSON | 50 (50) | 25 (25) | 25 (25) | 100 |

### stock 3B — held-out set, tool list in the system message — first 5090, GPU

| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |
|---|---|---|---|---|
| Stock server, chat endpoint | 11 (11) | 6 (6) | 5 (5) | 25 |
| Stock server, raw text, no cache | 28 (28) | 15 (15) | 13 (13) | 88 |
| Stock server, raw text, its own prompt cache | 31 (31) | 18 (18) | 13 (13) | 89 |
| Runtime, nothing on | 28 (28) | 15 (15) | 13 (13) | 88 |
| Runtime, cache reuse | 31 (31) | 18 (18) | 13 (13) | 89 |
| Runtime, forced JSON | 32 (29) | 17 (15) | 15 (14) | 90 |
| Runtime, cache reuse and forced JSON | 37 (35) | 22 (21) | 15 (14) | 91 |

### stock 3B — held-out set, tool list in the user message — first 5090, GPU

| Configuration | Tasks correct | Same wording as development, of 50 | New wording, of 50 | Ended with an answer |
|---|---|---|---|---|
| Stock server, chat endpoint | 0 (0) | 0 (0) | 0 (0) | 2 |
| Stock server, raw text, no cache | 1 (0) | 0 (0) | 1 (0) | 25 |
| Stock server, raw text, its own prompt cache | 2 (1) | 0 (0) | 2 (1) | 28 |
| Runtime, nothing on | 0 (0) | 0 (0) | 0 (0) | 22 |
| Runtime, cache reuse | 1 (1) | 0 (0) | 1 (1) | 27 |
| Runtime, forced JSON | 0 (0) | 0 (0) | 0 (0) | 30 |
| Runtime, cache reuse and forced JSON | 0 (0) | 0 (0) | 0 (0) | 29 |

