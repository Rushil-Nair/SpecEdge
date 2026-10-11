# Training-data overlap with the 100 multi-step tasks

Check: `agent/tasks/check_training_overlap.py`, run on the rented machine on 2026-10-08 at
06:49 UTC, when `mine_multistep-Q4_K_M.gguf` appeared. Result: `training_overlap_check_final.json`.

| Training file | Used by | Examples | With a task question | With a contact | Naming one of the six tools |
|---|---|---|---|---|---|
| `/workspace/data/train_v1/train.jsonl` | `mine` (first fine-tune) | 20,000 | 0 | 0 | 0 |
| `/workspace/data/train_v2_clean/train.jsonl` | `mine_multistep` (second fine-tune) | 33,252 | 0 | 0 | 0 |

Matched to the second run:
- its config `training/configs/lora_r16_multistep.json` has `"train_file": "/workspace/data/train_v2_clean/train.jsonl"`;
- its log `/workspace/queue/logs/train_multistep.log` starts "run lora_r16_multistep: training file
  /workspace/data/train_v2_clean/train.jsonl" and "33111 training examples, 141 left out as longer than 2048 tokens";
- the file checked has 33,252 lines, sha256 `413c090d5b65f7b5c38cb4f6ff22ed12392e27629d69286f814a1ce10e8b8fc7`,
  last written 02:51 UTC, before that run started.

The earlier file `train_v2_both` (23 examples naming a tool called `create_event`) was not used
for either model and was no longer on the machine at the final check.

First run: its config `training/configs/lora_r16_full.json` has `"train_file": "/workspace/data/train_v1/train.jsonl"`
and its log says "19985 training examples, 15 left out as longer than 2048 tokens".

## Third fine-tune (`lora_r16_recovery`), checked 2026-10-08 10:49 UTC on the second 5090

From this check on, `check_training_overlap.py` covers both agent task sets: the 200 questions
(development and held-out), the 24 people and their contact ids, and the six tool names.
Result: `training_overlap_check_v3_second5090.json`.

| Training file | Used by | Examples | With a task question (200) | With a contact (24 people, 24 ids) | Naming one of the six tools |
|---|---|---|---|---|---|
| `/workspace/data/train_v3_recovery/train.jsonl` | third fine-tune | 39,990 | 0 | 0 | 0 |

Matched to the run:
- its config, `training/configs/lora_r16_recovery.json`, has
  `"train_file": "/workspace/data/train_v3_recovery/train.jsonl"`;
- its log `/workspace/queue/logs/train_recovery.log` starts "run lora_r16_recovery: training file
  /workspace/data/train_v3_recovery/train.jsonl" and "39880 training examples, 110 left out as
  longer than 2048 tokens";
- the file checked has 39,990 lines, sha256 `1053a7000f20617e41abb5ca8075a4aeec2aee23535ce0662a281705620d649b`,
  last written 10:47:23 UTC (`training_file_v3_second5090.txt`).

What this check cannot see: whether the generated practice conversations in that file are
built like the ten task families. It looks for exact questions, names, ids and tool names only.

A second check ran on that machine with a copy of the agent tasks at `99dce58`
(`/workspace/agent_check`, named in its `normalized_overlap_check.json`). That copy
includes the held-out task file. No log on the machine shows a model run on it; the only logs
that mention it are the build and overlap logs.
