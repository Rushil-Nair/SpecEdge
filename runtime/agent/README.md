# specedge-agent

One C++17 program, `specedge-agent`, on the llama.cpp API (`llama.h` only) at the pinned
commit `f498f864`. It loads tool schemas from JSON files, builds the prompt in Llama 3.2's
tool format, keeps the key-value cache between turns, writes replies under a grammar built
from the schemas, reads the calls with its own parser, runs the tools and loops until the
model answers in plain text or a turn limit is reached.

The tools, the 100 tasks and the control are in `agent/`.

## Files

| File | What it does |
|---|---|
| `src/tools.*` | Loads tool schemas (one JSON file per tool, or a list) |
| `src/prompt.*` | The prompt, following the chat template of `meta-llama/Llama-3.2-3B-Instruct` (revision `0cb88a4f`) |
| `src/grammar.*` | Forced JSON: a GBNF grammar from the schemas |
| `src/parser.*` | Reads Llama 3.2's calls, including several in one reply |
| `src/engine.*` | Model, tokenizer, cache reuse, greedy decoding with or without the grammar |
| `src/loop.*` | The tool loop with a turn limit |
| `src/engine_draft.*` | The engine behind the drafting loop: served model with its grammar, a second model as drafter |
| `src/host.*`, `src/toolhost_main.cpp` | The tool host (tools, parser, prompt, checker over stdin/stdout); also built alone as `specedge-toolhost` |
| `src/policy_config.*` | Drafting-policy settings from a JSON file |
| `src/main.cpp` | Commands: `run`, `toolhost`, `render`, `grammar`, `tokenize`, `grammar-check` |
| `tests/test_units.cpp` | Checks that need no model |
| `tests/grammar_cases.json` | Replies the grammar must accept or refuse |
| `tests/check_prompt.py` | Compares the prompt text and tokens with stock llama-server's |
| `report.py` | Builds `runtime/results/part6/REPORT.md` from the result files |
| `../scripts/run_matrix.sh` | Runs every configuration; resumable |
| `headline.py`, `analyze_part6.py`, `strict.py` | Summary tables, failure breakdown and the stricter answer rule |

## Build and run (on the rented machine)

    cmake -S runtime/agent -B build -DLLAMA_DIR=/workspace/llama.cpp && cmake --build build -j4
    ./build/test_units agent/tools
    ./build/specedge-agent run --model <gguf> --tools agent/tools/schemas --contacts agent/tools/contacts.json \
        --tasks agent/tasks/tasks_v1.jsonl --system agent/tasks/system_prompt.txt --out runtime/results/part6/<run>.json \
        --cache 1 --forced-json 1 --tools-in user --tool-results stock --threads 8 --ngl 0

It links the shared libraries of an existing llama.cpp build and does not rebuild llama.cpp.

## Options that change what the model sees

| Option | Values | Meaning |
|---|---|---|
| `--cache` | `1`, `0` | Keep the longest common start of the prompt in the cache, between turns and between tasks; or empty the cache every turn |
| `--forced-json` | `1`, `0` | Write replies under the grammar, or freely |
| `--tools-in` | `user`, `system` | Where the template puts the tool list: the first user message (the template's default, and what the fine-tuning data uses) or the system message (`tools_in_user_message` false) |
| `--tool-results` | `stock`, `plain` | A tool result as a quoted, escaped JSON string (what the template does with a string, and the only form stock llama-server produces) or as plain JSON |

## What forced JSON allows

A reply is either one or more calls, each exactly `{"name": "<tool>", "parameters": {...}}`
joined by `"; "`, or plain text that does not start with `{`. Inside `parameters`: only that
tool's arguments, in schema order, each at most once, every required one present, every value
in the schema's JSON type. A `number` is a bare number, so the base model cannot write
`"0"` for 0.5. Control tokens such as `<|python_tag|>` are not offered while the grammar is on.
The model's first choice is tried against the grammar first; the whole vocabulary is filtered
only when that choice is refused.

## Checked so far (CPU of the rented machine, no timing)

| Check | Result |
|---|---|
| Unit checks (`test_units`) | 74 of 74 |
| Grammar cases through llama.cpp's grammar engine (`grammar-check`) | 25 of 25 |
| Prompt text and tokens against stock llama-server (`check_prompt.py`, 11 conversations, both layouts) | all the same |
| Tasks: reference solutions accepted; doing nothing, cut decimals and an extra message refused (`verify_tasks.py`) | 100 tasks, 0 problems |

Two things found about the stock server at this commit while checking:
- It writes a tool result given as a string as a quoted, escaped JSON string, and answers
  HTTP 400 when the result is given as an object.
- Its text from `/apply-template` has no `<|begin_of_text|>`; `/completion` adds it when
  tokenizing, so the token sequences are the same as the runtime's.

## Results

All results and the current state are in `runtime/README.md`.

## Drafting

`--draft-model M2 --policy FILE [--requests-in-flight N]` writes every reply through the
drafting policy of `runtime/controller/` (`src/engine_draft.*`). Without those options replies
are written by `Engine::generate`, the path every timed run used. No drafted run has been timed.
