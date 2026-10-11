# Part 6 — agent runtime against the stock llama.cpp server

Built by `runtime/agent/report.py` from the files in this folder. Seconds are per task, mean over the tasks of a run, then median over repeats (lowest-highest). Rows without seconds are functional runs on a machine that was not booked for timing. The development set and the held-out set are never in one table, and neither are the two machines. In held-out tables the last two columns give the tasks correct in each wording half under the current rule, with the stricter rule in brackets.

## Llama-3.2-3B-Instruct-Q4_K_M.gguf, first 5090, CPU, 7 threads (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | yes | 3 | 4 of 20 | 4 of 20 | 19 | 6 | 40.75 (40.49-40.78) | 31.17 (30.92-31.22) | 9.50 (9.49-9.50) |
| system message | runtime | off | on | stock | yes | 3 | 6 of 20 | 6 of 20 | 20 | 7 | 33.83 (33.83-34.10) | 26.45 (26.43-26.70) | 7.34 (7.32-7.34) |
| system message | runtime | on | off | stock | yes | 3 | 4 of 20 | 4 of 20 | 18 | 5 | 11.69 (11.67-11.70) | 2.27 (2.26-2.28) | 9.41 (9.41-9.41) |
| system message | runtime | on | on | stock | yes | 3 | 6 of 20 | 6 of 20 | 19 | 7 | 12.11 (11.99-12.13) | 2.75 (2.59-2.75) | 9.38 (9.36-9.39) |
| system message | stock server, chat endpoint | off | off | stock | yes | 3 | 2 of 20 | 2 of 20 | 5 | 3 | 53.06 (52.82-53.18) | 37.03 (36.81-37.11) | 12.27 (12.26-12.30) |
| system message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 4 of 20 | 4 of 20 | 19 | 6 | 39.81 (39.60-39.91) | 30.64 (30.43-30.74) | 9.13 (9.13-9.15) |
| system message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 4 of 20 | 4 of 20 | 18 | 5 | 11.68 (11.67-11.70) | 2.29 (2.28-2.30) | 9.36 (9.35-9.36) |

## Llama-3.2-3B-Instruct-Q4_K_M.gguf, first 5090, CPU, 8 threads (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | on | off | stock | no | 1 | 27 of 100 | 27 of 100 | 93 | 36 |  |  |  |
| system message | runtime | on | on | stock | no | 1 | 39 of 100 | 39 of 100 | 94 | 47 |  |  |  |
| system message | stock server, chat endpoint | off | off | stock | no | 1 | 11 of 100 | 11 of 100 | 21 | 16 |  |  |  |
| system message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 27 of 100 | 27 of 100 | 93 | 36 |  |  |  |
| user message | stock server, chat endpoint | off | off | stock | no | 1 | 0 of 100 | 0 of 100 | 3 | 20 |  |  |  |
| user message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 0 of 100 | 0 of 100 | 17 | 39 |  |  |  |

## Llama-3.2-3B-Instruct-Q4_K_M.gguf, first 5090, GPU (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | yes | 3 | 25 of 100 | 24 of 100 | 92 | 33 | 0.82 (0.82-0.82) | 0.22 (0.22-0.22) | 0.59 (0.59-0.59) |
| system message | runtime | off | on | stock | yes | 3 | 36 of 100 | 35 of 100 | 94 | 44 | 0.70 (0.70-0.70) | 0.20 (0.20-0.20) | 0.49 (0.49-0.49) |
| system message | runtime | on | off | stock | no | 1 | 24 of 100 | 23 of 100 | 93 | 33 |  |  |  |
| system message | runtime | on | off | stock | yes | 3 | 24 of 100 | 23 of 100 | 93 | 33 | 0.60 (0.60-0.60) | 0.01 (0.01-0.01) | 0.59 (0.59-0.59) |
| system message | runtime | on | on | stock | no | 1 | 38 of 100 | 38 of 100 | 96 | 45 |  |  |  |
| system message | runtime | on | on | stock | yes | 3 | 38 of 100 | 38 of 100 | 96 | 45 | 0.46 (0.46-0.46) | 0.01 (0.01-0.01) | 0.45 (0.45-0.45) |
| system message | stock server, chat endpoint | off | off | stock | yes | 3 | 13 of 100 | 13 of 100 | 27 | 18 | 0.98 (0.98-0.98) | 0.29 (0.29-0.29) | 0.57 (0.57-0.57) |
| system message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 25 of 100 | 24 of 100 | 92 | 33 | 0.80 (0.80-0.81) | 0.28 (0.28-0.28) | 0.50 (0.50-0.50) |
| system message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 24 of 100 | 23 of 100 | 93 | 33 | 0.58 (0.58-0.58) | 0.03 (0.03-0.03) | 0.52 (0.52-0.52) |
| user message | runtime | off | off | stock | yes | 3 | 0 of 100 | 0 of 100 | 19 | 36 | 1.69 (1.69-1.69) | 0.51 (0.51-0.51) | 1.17 (1.17-1.17) |
| user message | runtime | off | on | stock | yes | 3 | 0 of 100 | 0 of 100 | 34 | 46 | 1.52 (1.52-1.52) | 0.48 (0.48-0.48) | 1.03 (1.03-1.03) |
| user message | runtime | on | off | stock | no | 1 | 0 of 100 | 0 of 100 | 21 | 34 |  |  |  |
| user message | runtime | on | off | stock | yes | 3 | 0 of 100 | 0 of 100 | 21 | 34 | 1.16 (1.16-1.16) | 0.02 (0.02-0.02) | 1.13 (1.13-1.13) |
| user message | runtime | on | on | stock | no | 1 | 0 of 100 | 0 of 100 | 30 | 48 |  |  |  |
| user message | runtime | on | on | stock | yes | 3 | 0 of 100 | 0 of 100 | 30 | 48 | 1.14 (1.14-1.14) | 0.02 (0.02-0.02) | 1.11 (1.11-1.11) |
| user message | stock server, chat endpoint | off | off | stock | yes | 3 | 0 of 100 | 0 of 100 | 3 | 20 | 1.31 (1.31-1.31) | 0.43 (0.43-0.43) | 0.74 (0.74-0.74) |
| user message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 0 of 100 | 0 of 100 | 24 | 36 | 1.63 (1.63-1.64) | 0.60 (0.60-0.60) | 0.94 (0.94-0.94) |
| user message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 0 of 100 | 0 of 100 | 25 | 34 | 1.13 (1.13-1.13) | 0.07 (0.07-0.07) | 0.97 (0.97-0.97) |

## Llama-3.2-3B-Instruct-Q4_K_M.gguf, first 5090, GPU (held-out set, tasks 49acc1e7)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation | Same wording as the development set, of 50 | New wording, of 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | no | 1 | 28 of 100 | 28 of 100 | 88 | 35 |  |  |  | 15 (15) | 13 (13) |
| system message | runtime | off | on | stock | no | 1 | 32 of 100 | 29 of 100 | 90 | 37 |  |  |  | 17 (15) | 15 (14) |
| system message | runtime | on | off | stock | no | 1 | 31 of 100 | 31 of 100 | 89 | 40 |  |  |  | 18 (18) | 13 (13) |
| system message | runtime | on | on | stock | no | 1 | 37 of 100 | 35 of 100 | 91 | 44 |  |  |  | 22 (21) | 15 (14) |
| system message | stock server, chat endpoint | off | off | stock | no | 1 | 11 of 100 | 11 of 100 | 25 | 16 |  |  |  | 6 (6) | 5 (5) |
| system message | stock server, raw text + runtime's parser | off | off | stock | no | 1 | 28 of 100 | 28 of 100 | 88 | 35 |  |  |  | 15 (15) | 13 (13) |
| system message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 31 of 100 | 31 of 100 | 89 | 40 |  |  |  | 18 (18) | 13 (13) |
| user message | runtime | off | off | stock | no | 1 | 0 of 100 | 0 of 100 | 22 | 31 |  |  |  | 0 (0) | 0 (0) |
| user message | runtime | off | on | stock | no | 1 | 0 of 100 | 0 of 100 | 30 | 45 |  |  |  | 0 (0) | 0 (0) |
| user message | runtime | on | off | stock | no | 1 | 1 of 100 | 1 of 100 | 27 | 36 |  |  |  | 0 (0) | 1 (1) |
| user message | runtime | on | on | stock | no | 1 | 0 of 100 | 0 of 100 | 29 | 50 |  |  |  | 0 (0) | 0 (0) |
| user message | stock server, chat endpoint | off | off | stock | no | 1 | 0 of 100 | 0 of 100 | 2 | 19 |  |  |  | 0 (0) | 0 (0) |
| user message | stock server, raw text + runtime's parser | off | off | stock | no | 1 | 1 of 100 | 0 of 100 | 25 | 31 |  |  |  | 0 (0) | 1 (0) |
| user message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 2 of 100 | 1 of 100 | 28 | 36 |  |  |  | 0 (0) | 2 (1) |

## Llama-3.2-3B-Instruct-Q4_K_M.gguf, second 5090, GPU (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | on | off | stock | no | 1 | 24 of 100 | 23 of 100 | 93 | 33 |  |  |  |
| system message | runtime | on | on | stock | no | 1 | 38 of 100 | 38 of 100 | 96 | 45 |  |  |  |
| user message | runtime | on | off | stock | no | 1 | 0 of 100 | 0 of 100 | 21 | 34 |  |  |  |
| user message | runtime | on | on | stock | no | 1 | 0 of 100 | 0 of 100 | 30 | 48 |  |  |  |

## Llama-3.2-3B-Instruct-Q4_K_M.gguf, second 5090, GPU (held-out set, tasks 49acc1e7)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation | Same wording as the development set, of 50 | New wording, of 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | on | off | stock | no | 1 | 31 of 100 | 31 of 100 | 89 | 40 |  |  |  | 18 (18) | 13 (13) |
| system message | runtime | on | on | stock | no | 1 | 37 of 100 | 35 of 100 | 91 | 44 |  |  |  | 22 (21) | 15 (14) |
| user message | runtime | on | off | stock | no | 1 | 1 of 100 | 1 of 100 | 27 | 36 |  |  |  | 0 (0) | 1 (1) |
| user message | runtime | on | on | stock | no | 1 | 0 of 100 | 0 of 100 | 29 | 50 |  |  |  | 0 (0) | 0 (0) |

## mine-Q4_K_M.gguf, first 5090, GPU (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | on | off | stock | no | 1 | 20 of 100 | 11 of 100 | 27 | 35 |  |  |  |
| system message | runtime | on | on | stock | no | 1 | 9 of 100 | 9 of 100 | 17 | 25 |  |  |  |
| user message | runtime | on | off | stock | no | 1 | 0 of 100 | 0 of 100 | 0 | 19 |  |  |  |
| user message | runtime | on | on | stock | no | 1 | 0 of 100 | 0 of 100 | 0 | 19 |  |  |  |

## mine_multistep-Q4_K_M.gguf, first 5090, GPU (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | yes | 3 | 21 of 100 | 21 of 100 | 100 | 28 | 0.48 (0.48-0.48) | 0.13 (0.13-0.13) | 0.35 (0.35-0.35) |
| system message | runtime | off | on | stock | yes | 3 | 21 of 100 | 21 of 100 | 100 | 28 | 0.48 (0.48-0.48) | 0.13 (0.13-0.13) | 0.35 (0.35-0.35) |
| system message | runtime | on | off | stock | no | 1 | 22 of 100 | 22 of 100 | 100 | 29 |  |  |  |
| system message | runtime | on | off | stock | yes | 3 | 22 of 100 | 22 of 100 | 100 | 29 | 0.34 (0.34-0.34) | 0.01 (0.01-0.01) | 0.33 (0.33-0.33) |
| system message | runtime | on | on | stock | no | 1 | 22 of 100 | 22 of 100 | 100 | 29 |  |  |  |
| system message | runtime | on | on | stock | yes | 3 | 22 of 100 | 22 of 100 | 100 | 29 | 0.34 (0.34-0.34) | 0.01 (0.01-0.01) | 0.33 (0.33-0.33) |
| system message | stock server, chat endpoint | off | off | stock | yes | 3 | 21 of 100 | 21 of 100 | 70 | 28 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.21 (0.21-0.21) |
| system message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 21 of 100 | 21 of 100 | 100 | 28 | 0.46 (0.46-0.46) | 0.16 (0.16-0.16) | 0.29 (0.29-0.29) |
| system message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 22 of 100 | 22 of 100 | 100 | 29 | 0.32 (0.32-0.33) | 0.02 (0.02-0.02) | 0.29 (0.29-0.29) |
| user message | runtime | off | off | stock | yes | 3 | 27 of 100 | 27 of 100 | 100 | 34 | 0.45 (0.45-0.45) | 0.12 (0.12-0.12) | 0.33 (0.33-0.33) |
| user message | runtime | off | on | stock | yes | 3 | 27 of 100 | 27 of 100 | 100 | 34 | 0.45 (0.45-0.45) | 0.12 (0.12-0.12) | 0.33 (0.33-0.33) |
| user message | runtime | on | off | stock | no | 1 | 29 of 100 | 29 of 100 | 100 | 36 |  |  |  |
| user message | runtime | on | off | stock | yes | 3 | 29 of 100 | 29 of 100 | 100 | 36 | 0.33 (0.33-0.33) | 0.01 (0.01-0.01) | 0.32 (0.32-0.32) |
| user message | runtime | on | on | stock | no | 1 | 29 of 100 | 29 of 100 | 100 | 36 |  |  |  |
| user message | runtime | on | on | stock | yes | 3 | 29 of 100 | 29 of 100 | 100 | 36 | 0.33 (0.33-0.33) | 0.01 (0.01-0.01) | 0.32 (0.32-0.32) |
| user message | stock server, chat endpoint | off | off | stock | yes | 3 | 27 of 100 | 27 of 100 | 70 | 34 | 0.40 (0.40-0.40) | 0.12 (0.12-0.12) | 0.20 (0.20-0.20) |
| user message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 27 of 100 | 27 of 100 | 100 | 34 | 0.44 (0.44-0.44) | 0.15 (0.15-0.15) | 0.28 (0.28-0.28) |
| user message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 29 of 100 | 29 of 100 | 100 | 36 | 0.32 (0.32-0.32) | 0.02 (0.02-0.02) | 0.28 (0.28-0.28) |

## mine_multistep-Q4_K_M.gguf, first 5090, GPU (held-out set, tasks 49acc1e7)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation | Same wording as the development set, of 50 | New wording, of 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | no | 1 | 20 of 100 | 20 of 100 | 100 | 25 |  |  |  | 13 (13) | 7 (7) |
| system message | runtime | off | on | stock | no | 1 | 20 of 100 | 20 of 100 | 100 | 25 |  |  |  | 13 (13) | 7 (7) |
| system message | runtime | on | off | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 13 (13) | 8 (8) |
| system message | runtime | on | on | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 13 (13) | 8 (8) |
| system message | stock server, chat endpoint | off | off | stock | no | 1 | 20 of 100 | 20 of 100 | 72 | 25 |  |  |  | 13 (13) | 7 (7) |
| system message | stock server, raw text + runtime's parser | off | off | stock | no | 1 | 20 of 100 | 20 of 100 | 100 | 25 |  |  |  | 13 (13) | 7 (7) |
| system message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 13 (13) | 8 (8) |
| user message | runtime | off | off | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 11 (11) | 10 (10) |
| user message | runtime | off | on | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 11 (11) | 10 (10) |
| user message | runtime | on | off | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 12 (12) | 9 (9) |
| user message | runtime | on | on | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 12 (12) | 9 (9) |
| user message | stock server, chat endpoint | off | off | stock | no | 1 | 21 of 100 | 21 of 100 | 74 | 26 |  |  |  | 11 (11) | 10 (10) |
| user message | stock server, raw text + runtime's parser | off | off | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 11 (11) | 10 (10) |
| user message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 21 of 100 | 21 of 100 | 100 | 26 |  |  |  | 12 (12) | 9 (9) |

## mine_recovery-Q4_K_M.gguf, first 5090, GPU (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | yes | 3 | 55 of 100 | 55 of 100 | 100 | 59 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) |
| system message | runtime | off | on | stock | yes | 3 | 55 of 100 | 55 of 100 | 100 | 59 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) |
| system message | runtime | on | off | stock | yes | 3 | 53 of 100 | 53 of 100 | 100 | 57 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) |
| system message | runtime | on | on | stock | yes | 3 | 53 of 100 | 53 of 100 | 100 | 57 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) |
| system message | stock server, chat endpoint | off | off | stock | yes | 3 | 55 of 100 | 55 of 100 | 91 | 59 | 0.41 (0.41-0.41) | 0.15 (0.15-0.15) | 0.21 (0.21-0.21) |
| system message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 55 of 100 | 55 of 100 | 100 | 59 | 0.42 (0.42-0.42) | 0.16 (0.16-0.16) | 0.24 (0.24-0.24) |
| system message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 53 of 100 | 53 of 100 | 100 | 57 | 0.27 (0.27-0.27) | 0.02 (0.02-0.02) | 0.23 (0.23-0.23) |
| user message | runtime | off | off | stock | yes | 3 | 49 of 100 | 49 of 100 | 100 | 57 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) |
| user message | runtime | off | on | stock | yes | 3 | 49 of 100 | 49 of 100 | 100 | 57 | 0.42 (0.42-0.42) | 0.13 (0.13-0.13) | 0.29 (0.29-0.29) |
| user message | runtime | on | off | stock | yes | 3 | 50 of 100 | 50 of 100 | 100 | 58 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) |
| user message | runtime | on | on | stock | yes | 3 | 50 of 100 | 50 of 100 | 100 | 58 | 0.28 (0.28-0.28) | 0.01 (0.01-0.01) | 0.27 (0.27-0.27) |
| user message | stock server, chat endpoint | off | off | stock | yes | 3 | 41 of 100 | 41 of 100 | 90 | 49 | 0.39 (0.39-0.39) | 0.15 (0.15-0.15) | 0.20 (0.20-0.20) |
| user message | stock server, raw text + runtime's parser | off | off | stock | yes | 3 | 49 of 100 | 49 of 100 | 100 | 57 | 0.42 (0.42-0.42) | 0.17 (0.17-0.17) | 0.23 (0.23-0.23) |
| user message | stock server, raw text + runtime's parser | on | off | stock | yes | 3 | 50 of 100 | 50 of 100 | 100 | 58 | 0.28 (0.28-0.28) | 0.02 (0.02-0.02) | 0.24 (0.24-0.24) |

## mine_recovery-Q4_K_M.gguf, first 5090, GPU (held-out set, tasks 49acc1e7)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation | Same wording as the development set, of 50 | New wording, of 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | off | off | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 28 (28) | 21 (21) |
| system message | runtime | off | on | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 28 (28) | 21 (21) |
| system message | runtime | on | off | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 26 (26) | 23 (23) |
| system message | runtime | on | on | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 26 (26) | 23 (23) |
| system message | stock server, chat endpoint | off | off | stock | no | 1 | 49 of 100 | 49 of 100 | 96 | 51 |  |  |  | 28 (28) | 21 (21) |
| system message | stock server, raw text + runtime's parser | off | off | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 28 (28) | 21 (21) |
| system message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 26 (26) | 23 (23) |
| user message | runtime | off | off | stock | no | 1 | 48 of 100 | 48 of 100 | 100 | 51 |  |  |  | 24 (24) | 24 (24) |
| user message | runtime | off | on | stock | no | 1 | 48 of 100 | 48 of 100 | 100 | 51 |  |  |  | 24 (24) | 24 (24) |
| user message | runtime | on | off | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 53 |  |  |  | 25 (25) | 25 (25) |
| user message | runtime | on | on | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 53 |  |  |  | 25 (25) | 25 (25) |
| user message | stock server, chat endpoint | off | off | stock | no | 1 | 44 of 100 | 44 of 100 | 95 | 47 |  |  |  | 20 (20) | 24 (24) |
| user message | stock server, raw text + runtime's parser | off | off | stock | no | 1 | 48 of 100 | 48 of 100 | 100 | 51 |  |  |  | 24 (24) | 24 (24) |
| user message | stock server, raw text + runtime's parser | on | off | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 53 |  |  |  | 25 (25) | 25 (25) |

## mine_recovery-Q4_K_M.gguf, second 5090, GPU (development set, tasks 0d3a4a36)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | on | off | stock | no | 1 | 53 of 100 | 53 of 100 | 100 | 57 |  |  |  |
| system message | runtime | on | on | stock | no | 1 | 53 of 100 | 53 of 100 | 100 | 57 |  |  |  |
| user message | runtime | on | off | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 58 |  |  |  |
| user message | runtime | on | on | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 58 |  |  |  |

## mine_recovery-Q4_K_M.gguf, second 5090, GPU (held-out set, tasks 49acc1e7)

| Tool list in | Program | Cache reuse | Forced JSON | Tool results | Timed | Runs | Tasks correct | Correct, stricter answer rule | Ended with an answer | Reached the expected state at some turn | Seconds per task | Prompt | Generation | Same wording as the development set, of 50 | New wording, of 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| system message | runtime | on | off | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 26 (26) | 23 (23) |
| system message | runtime | on | on | stock | no | 1 | 49 of 100 | 49 of 100 | 100 | 51 |  |  |  | 26 (26) | 23 (23) |
| user message | runtime | on | off | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 53 |  |  |  | 25 (25) | 25 (25) |
| user message | runtime | on | on | stock | no | 1 | 50 of 100 | 50 of 100 | 100 | 53 |  |  |  | 25 (25) | 25 (25) |
