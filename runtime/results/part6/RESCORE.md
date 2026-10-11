# Every row under the current rule and the stricter rule for the final answer

Built by `runtime/agent/analyze_part6.py`. The current rule accepts any final text once the world is in the expected state. The stricter rule (`runtime/agent/strict.py`) also needs the final reply to be plain text: not empty, not starting with `{` or `[`, not JSON for an object or list, and not a quoted JSON string holding one (the form a tool result has in the prompt). One line per result file, timed repeats included.

| Set | Model | Device | Tool list in | Program | Forced JSON | Tasks | Correct, current rule | Correct, stricter rule | Correct only under the current rule | of those, the answer repeats a tool result of the task word for word | Answered tasks whose final answer is not plain text |
|---|---|---|---|---|---|---|---|---|---|---|---|
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 100 | 27 | 27 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 100 | 39 | 39 | 0 | 0 | 11 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 20 | 6 | 6 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 20 | 6 | 6 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 20 | 6 | 6 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 20 | 6 | 6 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 20 | 6 | 6 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | runtime | on | 20 | 6 | 6 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, chat endpoint | off | 100 | 11 | 11 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, chat endpoint | off | 20 | 2 | 2 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, chat endpoint | off | 20 | 2 | 2 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, chat endpoint | off | 20 | 2 | 2 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 100 | 27 | 27 | 0 | 0 | 2 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | system | stock server, raw text | off | 20 | 4 | 4 | 0 | 0 | 1 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | user | stock server, chat endpoint | off | 100 | 0 | 0 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 CPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 25 | 24 | 1 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 25 | 24 | 1 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 25 | 24 | 1 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 38 | 38 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 38 | 38 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 38 | 38 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 38 | 38 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 36 | 35 | 1 | 1 | 12 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 36 | 35 | 1 | 1 | 12 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 36 | 35 | 1 | 1 | 12 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 13 | 13 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 13 | 13 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 13 | 13 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 25 | 24 | 1 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 25 | 24 | 1 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 25 | 24 | 1 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 9 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 9 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 9 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 9 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 0 | 0 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 0 | 0 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 0 | 0 | 0 | 0 | 0 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 0 | 0 | 0 | 0 | 4 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | system | runtime | off | 100 | 24 | 23 | 1 | 0 | 8 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | system | runtime | on | 100 | 38 | 38 | 0 | 0 | 10 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 3 |
| development | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 9 |
| development | mine-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 20 | 11 | 9 | 4 | 9 |
| development | mine-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 9 | 9 | 0 | 0 | 0 |
| development | mine-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 0 |
| development | mine-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 22 | 22 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 21 | 21 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 29 | 29 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 27 | 27 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 55 | 55 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 41 | 41 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 41 | 41 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 41 | 41 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 49 | 49 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | second 5090 GPU | system | runtime | off | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | second 5090 GPU | system | runtime | on | 100 | 53 | 53 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | second 5090 GPU | user | runtime | off | 100 | 50 | 50 | 0 | 0 | 0 |
| development | mine_recovery-Q4_K_M | second 5090 GPU | user | runtime | on | 100 | 50 | 50 | 0 | 0 | 0 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 31 | 31 | 0 | 0 | 6 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 28 | 28 | 0 | 0 | 5 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 37 | 35 | 2 | 0 | 13 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 32 | 29 | 3 | 0 | 14 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 11 | 11 | 0 | 0 | 0 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 31 | 31 | 0 | 0 | 6 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 28 | 28 | 0 | 0 | 6 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 1 | 1 | 0 | 0 | 2 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 0 | 0 | 0 | 0 | 3 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 4 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 6 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 0 | 0 | 0 | 0 | 0 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 2 | 1 | 1 | 0 | 3 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 1 | 0 | 1 | 0 | 6 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | system | runtime | off | 100 | 31 | 31 | 0 | 0 | 6 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | system | runtime | on | 100 | 37 | 35 | 2 | 0 | 13 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | user | runtime | off | 100 | 1 | 1 | 0 | 0 | 2 |
| held-out | Llama-3.2-3B-Instruct-Q4_K_M | second 5090 GPU | user | runtime | on | 100 | 0 | 0 | 0 | 0 | 4 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 20 | 20 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 20 | 20 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 20 | 20 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 20 | 20 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_multistep-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 21 | 21 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | off | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | runtime | on | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, chat endpoint | off | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | system | stock server, raw text | off | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 50 | 50 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | off | 100 | 48 | 48 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 50 | 50 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | runtime | on | 100 | 48 | 48 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, chat endpoint | off | 100 | 44 | 44 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 50 | 50 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | first 5090 GPU | user | stock server, raw text | off | 100 | 48 | 48 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | second 5090 GPU | system | runtime | off | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | second 5090 GPU | system | runtime | on | 100 | 49 | 49 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | second 5090 GPU | user | runtime | off | 100 | 50 | 50 | 0 | 0 | 0 |
| held-out | mine_recovery-Q4_K_M | second 5090 GPU | user | runtime | on | 100 | 50 | 50 | 0 | 0 | 0 |
