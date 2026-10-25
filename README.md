# SpecEdge

A tool-calling agent for small hardware, built around Llama 3.2 3B: a LoRA fine-tune for
function calling, the cost of quantizing it measured on that task (llama.cpp's levels and
Qualcomm's w4a16 for the Snapdragon NPU), a C++17 agent runtime on the llama.cpp API, and a
fix to llama.cpp's server for slow batching when it has idle slots.

Every number below comes from a run in this repository. Result files are under `results/` and
`runtime/results/part6/`; `results/part2/summary_tables.md` has the full tool-call table.

## Results

### Fine-tune for tool calls

Llama 3.2 3B Instruct with a LoRA add-on (rank 16), scored on a frozen set of 591 single-turn
tasks whose tools and questions are not in the training data: 371 where a call is right, 130
where the model should ask for missing information, 90 where no tool fits.

| Model | All 591 | 572 tasks that show tools | Call tasks (371) |
|---|---|---|---|
| Base Llama 3.2 3B | 48.6% | 47.6% | 73.3% |
| **Fine-tuned (`mine_recovery`, the final model)** | **65.7%** | **64.5%** | **85.2%** |
| Salesforce xLAM-2-3b-fc-r, as a yardstick | 62.8% | 64.0% | 79.2% |

- +17.1 points over the base; task by task, 110 fixed and 9 broken.
- On par with xLAM-2-3b overall (64.5% against 64.0% on the 572 tasks that show tools). Paired
  per category: better on call tasks (85.2% against 79.2%, McNemar p = 0.002), worse at refusing
  when tools are shown but none fits (38.0% against 50.7%, p = 0.02), no significant difference
  elsewhere.
- Weak spot: when information is missing the model still calls a tool four times out of five.

### Multi-step agent tasks

100 held-out multi-step tasks over six tools (contacts, calendar, expenses, messages), frozen
before any model was run on them. 4-bit, GPU.

| Model | Stock llama.cpp chat endpoint | C++ runtime |
|---|---|---|
| Stock Llama 3.2 3B | 11 | 37 (its best setting) |
| **Final model (`mine_recovery`)** | **49** | **49 / 50** (two prompt layouts) |

The chat-endpoint column is the like-for-like comparison: same loop, prompt layout, tasks and
checker. The final model was trained with 1,900 script-generated error-recovery conversations
(look the id up first, retry after a tool error) on tools and topics different from the test's.
It still fails about half the tasks.

### Quantization in llama.cpp (final model)

| Level | File | All 591 | GPU tokens/s (RTX 5090) | GPU memory |
|---|---|---|---|---|
| F16 | 6.43 GB | 65.7% | 216.8 | 7.55 GB |
| Q8_0 | 3.42 GB | 65.7% | 337.8 | 4.71 GB |
| Q5_K_M | 2.32 GB | 66.5% | 412.6 | 3.68 GB |
| Q4_K_M | 2.02 GB | 65.0% | 438.8 | 3.40 GB |
| Q4_0 | 1.92 GB | 65.5% | 477.1 | 3.30 GB |

Q4_K_M is 69% smaller than F16 and 2.0x faster on the GPU, with no measurable accuracy loss (5
tasks gained, 9 lost; McNemar p = 0.42). On a desktop CPU (Ryzen 7 7800X3D, 7 threads) the model
writes 7.7 tokens/s at F16 and 21.8 at Q4_K_M.

### Qualcomm AIMET w4a16 and the Snapdragon 8 Elite NPU

Scored on a fixed 200-task subset inside Qualcomm's own pipeline (simulation), each model against
its own full-precision control in the same pipeline.

| Model | Control | w4a16, default | w4a16, tool-call calibration and forced JSON |
|---|---|---|---|
| Stock Llama 3.2 3B | 42.0% | 34.0% | not run |
| Second fine-tune (`mine_multistep`) | 67.0% | 63.5% | 66.0% |
| **Final model (`mine_recovery`)** | 65.0% | not run | 62.5% |

- With calibration on the model's own training text and a JSON grammar, the final model stays
  within 2.5 points of its control (10 tasks lost, 5 gained).
- On a hosted Galaxy S25 (Snapdragon 8 Elite) through Qualcomm AI Hub, every operation runs on
  the NPU. Token step summed over the three model parts: 43.8 ms for the final model, about 22
  tokens/s. That is estimated from separately profiled parts, not an end-to-end run; no physical
  phone was used.
- The first fine-tune's w4a16 run produced junk because Qualcomm's tools (transformers 4.45) read
  the merged model's config with the wrong RoPE settings (`rope_theta` 10000 instead of 500000,
  no Llama 3 scaling). `training/merge.py` now copies the base config unchanged, and a
  full-precision control through Qualcomm's engine runs before every quantizer run.

### llama.cpp server: slot selection with idle slots

With per-slot KV caches llama.cpp only batches requests whose slot numbers are consecutive, and
the server picks the least recently used free slot, so with fewer requests than slots each
decode step is split into several forward passes. A 12-line change to pick the lowest free slot
instead (Qwen3-8B Q4_K_M, RTX 5090, 16 slots):

| Users | Before, tokens/s | After, tokens/s |
|---|---|---|
| 1 | 228.3 | 228.1 |
| 4 | 262.5 | 487.8 |
| 8 | 333.6 | 878.9 (2.6x) |
| 16 | 1,143.6 | 1,136.2 |

Patch: `results/idle_slots/machine/idle_fix.patch`. Runs, diagnostic counters and output checks:
`results/idle_slots/`. Not posted upstream yet.

Why the split passes cost this much: decode at these batch sizes is limited by memory bandwidth,
and every forward pass reads all the weights again (about 4.7 GB for Qwen3-8B Q4_K_M). At 1 user
the server writes 228 tokens/s, an estimated 61% of the RTX 5090's 1,792 GB/s limit; before the
fix an 8-user decode step took about 4 passes, after it about 1. These are estimates from bytes
moved and rated bandwidth, not profiler measurements: `bench/idle_slots/roofline.py`, output in
`results/idle_slots/roofline.md`.

## Repository layout

| Folder | What |
|---|---|
| `data/` | The frozen 591-task test set (`data/toolset/v1/`) and its builder |
| `training/` | Training-file builders, the error-recovery conversation generator, overlap checks, LoRA training and merge |
| `bench/` | The tool-call scorer, forced JSON, paired comparison, the idle-slot study scripts |
| `quant/` | GGUF files at five levels; accuracy, speed and memory per level |
| `npu/` | Qualcomm's quantizer on a custom checkpoint, scoring in its simulation, calibration, AI Hub export and profile |
| `agent/` | The six in-memory tools, the development and held-out agent tasks, the stock-server control loop |
| `runtime/` | The C++17 agent runtime (`runtime/agent/`) and its drafting policy (`runtime/controller/`) |
| `serving/` | The load generator used for the idle-slot runs |
| `scripts/` | The Qualcomm study scripts |
| `results/` | Result summaries, one file per run |

## Building and running

The C++ runtime builds against a llama.cpp checkout at the pinned commit (details in
`runtime/README.md`):

    git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout f498f864
    cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF && cmake --build llama.cpp/build -j --target llama
    cmake -S runtime/agent -B build/agent -DLLAMA_DIR=$PWD/llama.cpp && cmake --build build/agent -j
    ctest --test-dir build/agent

The fine-tune and the measurements ran on a rented RTX 5090; Qualcomm's quantizer needs about
220 GB of memory and ran on an A100 machine with 259 GB.

    python training/prepare.py --out <dir>             # first training file, with the overlap check
    python training/prepare_multistep.py ...           # adds multi-step conversations
    python training/prepare_recovery.py ...            # adds error-recovery conversations
    python training/drop_normalized_overlap.py ...     # stricter overlap check on the finished file
    python training/train.py --config training/configs/lora_r16_recovery.json
    python training/merge.py --base <base> --adapter <add-on> --out <merged>
    bash quant/make.sh <merged> <name>                 # GGUF files at five levels
    python bench/toolcall_eval.py --name <run> --model <gguf> --tasks data/toolset/v1/toolset_v1.jsonl
    python quant/run.py --name <name> --what accuracy  # and --what speed --device gpu|cpu
    bash scripts/part12_short_a100.sh ...              # Qualcomm's pipeline, on the large-memory machine

Pinned: llama.cpp `f498f864`, `meta-llama/Llama-3.2-3B-Instruct` revision `0cb88a4f`,
`qai-hub-models` 0.63.0, torch 2.14.1, transformers 5.19.0, peft 0.21.2. The gated models and
data need a Hugging Face account that has accepted their terms.

## Limits

- The scorer has not been checked against independent hand labels. It counts "no call was
  made" as correct for a should-not-call task, whatever the reply says.
- Test and training tasks come from the same public sources (different tools, different
  questions). 865 training examples whose tool names match a test tool after normalising were
  removed from the final model's training file.
- The agent tasks and the w4a16 subset are 100 to 200 tasks; differences of a few tasks are not
  established.
- Qualcomm numbers are from its simulation and from separately profiled parts on a hosted device.

## Data and licences

Code: MIT (`LICENSE`). Only the code is published here, not model weights (Llama 3.2 Community
License).

The frozen test set (`data/toolset/v1/toolset_v1.jsonl`) is built from two CC-BY-4.0 datasets:
312 tasks from Salesforce's xLAM function-calling 60k
(`Salesforce/xlam-function-calling-60k`) and 279 from NVIDIA's When2Call (`nvidia/When2Call`,
test split), converted and filtered as described in `data/build_toolset_v1.py`; see
`data/toolset/v1/SOURCES.txt`. The agent tasks in `agent/tasks/` are written by the generators
in that folder. Training data (not included): xLAM function-calling 60k and When2Call
(CC-BY-4.0), ToolACE and Glaive function-calling v2 (Apache 2.0). xLAM-2-3b-fc-r is
CC-BY-NC-4.0 and was used as a yardstick only.
