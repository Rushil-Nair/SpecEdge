# C++ agent runtime

A tool-calling agent loop in C++17 on the llama.cpp API (commit `f498f864`), measured against
the stock llama.cpp server. Every number comes from a run in this repository; the result files
and generated tables are in `runtime/results/part6/` (`HEADLINE.md` is the summary, built by
`runtime/agent/headline.py`).

## Build and test

    # llama.cpp at the pinned commit, built as shared libraries
    git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout f498f864
    cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF && cmake --build llama.cpp/build -j --target llama
    # the runtime, the tool host and the unit checks
    cmake -S runtime/agent -B build/agent -DLLAMA_DIR=$PWD/llama.cpp && cmake --build build/agent -j
    ctest --test-dir build/agent
    # the drafting policy and its trace replay
    cmake -S runtime/controller -B build/controller && cmake --build build/controller -j
    ctest --test-dir build/controller

`-DSPECEDGE_WITH_LLAMA=OFF` builds only what needs no model (unit checks and the tool host); the
llama.cpp checkout is then used for its vendored JSON header alone. Usage of `specedge-agent`:
`runtime/agent/README.md`.

## End to end, on the held-out set

**Stock Llama 3.2 3B through the stock llama.cpp chat endpoint: 11 of 100 tasks correct with
the tool list in the system message, 0 of 100 with it in the user message. `mine_recovery`
through the runtime: 49 of 100 and 50 of 100.** Both at 4-bit on the first RTX 5090's GPU,
one run each, not timed; the held-out set was frozen before any model was run on it.

## In one paragraph

The C++ agent runtime is built and measured against the stock llama.cpp server on
three models, two task sets and two prompt layouts. **What the runtime adds is forced JSON and
a parser for Llama 3.2's several-calls-in-one-reply format; how much that is worth depends on
the model.** On the stock 3B it takes tasks correct from 13 (stock chat endpoint) to 38 of 100
and halves the time. On the final fine-tuned model, `mine_recovery`, forced JSON changes
nothing, the parser is worth 6 to 9 tasks in the default prompt layout and none in the other,
and the time is the server's. **Cache reuse works, and it only matches the stock server's own
prompt cache; it is not a gain over the stock server.** The largest effect measured in this
part is the model itself: 0 → 29 → 50 of 100 in the default layout across the fine-tunes, and
50 of 100 on a held-out set no model had been run on before.

## What is where

| | |
|---|---|
| `runtime/agent/` | The runtime (`specedge-agent`), its tests, the report scripts. Details: `runtime/agent/README.md` |
| `runtime/controller/` | The drafting policy, its recorded-trace tests, and the loop that connects it to the runtime. Details: `runtime/controller/README.md` |
| `agent/tools/` | Six in-memory tools, the world they act on, the checker |
| `agent/tasks/` | Development set (100 tasks) and held-out set (100 tasks). Details: `agent/tasks/README.md` |
| `agent/control/` | The control: a Python loop on the stock llama.cpp server |
| `runtime/results/part6/` | Every result file, and the generated tables: `HEADLINE.md` (summary), `REPORT.md` (every row), `RESCORE.md` (both answer rules), `FAILURES_mine_multistep.md`, `OVERLAP_NOTE.md` |

## How it was measured

- Models: the 4-bit (Q4_K_M) files of the stock Llama 3.2 3B and of the three fine-tunes
  (`mine`, `mine_multistep`, `mine_recovery`). llama.cpp commit `f498f864`. Greedy decoding, at
  most 10 turns per task and 256 tokens per turn.
- A task is **correct** when it ended with a plain-text reply, the calendar, expenses and
  messages are exactly what the task expects (nothing missing, nothing extra), the required
  tools were called, and the reply contains the answer where the task has one. The **stricter
  rule**, always reported beside it, also needs the final reply to be plain text, not a tool
  result given back.
- **Two task sets, never added together.** The development set was read task by task while
  studying failures. The held-out set (same tools and families, other people, ids, titles,
  dates and amounts; half of each family reworded) was frozen before any model was run on it.
- **Two prompt layouts**, both branches of Llama 3.2's own chat template: the tool list in the
  first user message (the template's default, and the training data's layout) or in the system
  message.
- **Two machines**, never mixed in one row: the first rented RTX 5090 (timing and
  correctness) and the second (correctness only). Where the same configuration ran on both
  GPUs, the counts were identical.
- **The control** is the unmodified llama.cpp server driven by a Python loop, in three forms:
  its chat endpoint (what a stock user gets: the server builds the prompt and parses the
  calls), raw text read by the runtime's parser with no cache, and the same with the server's
  own prompt cache on.
- **Timing** only on a machine with nothing else running, with a check
  before every run that nothing else was on the GPU and the load was under 1.5; 3 repeats;
  median with lowest and highest. Correctness-only runs are marked "not timed" and carry no
  seconds.

## Findings

### 1. The model matters most

Runtime with cache reuse and forced JSON, GPU, tasks correct of 100 (stricter rule gives the
same numbers except where shown in `runtime/results/part6/HEADLINE.md`).

| Model | Development, user layout | Development, system layout | Held-out, user layout | Held-out, system layout |
|---|---|---|---|---|
| Stock 3B | 0 | 38 | 0 | 37 |
| `mine` (first fine-tune) | 0 | 9 | not run | not run |
| `mine_multistep` (second) | 29 | 22 | 21 | 21 |
| `mine_recovery` (third, final) | 50 | 53 | 50 | 49 |

- The stock model in the default layout does not stop calling tools after a task is done: on
  the development set it reached the expected state in 34 tasks (48 with forced JSON) and
  went on. In the system layout it stops.
- `mine_recovery` holds its score on the held-out set (50 and 49 against 50 and 53) and on
  both wording halves (25 and 25 of 50 in the user layout; 26 and 23 in the system layout).
- Where the final model still fails, on both sets: `cancel_notify` and `move` (0 to 2 of 10),
  and `expenses_report` in the system layout (0 of 10).

### 2. The runtime against the stock server

First 5090, GPU, development set, 3 repeats (they agreed to 0.01 s). Seconds per task, tasks
correct. "Full runtime" is cache reuse and forced JSON.

| Model, layout | Stock chat endpoint | Stock server, raw text, no cache | Stock server, raw text, own cache | Full runtime |
|---|---|---|---|---|
| Stock 3B, system | 0.98 s, 13 | 0.80 s, 25 | 0.58 s, 24 | **0.46 s, 38** |
| Stock 3B, user | 1.31 s, 0 | 1.63 s, 0 | 1.13 s, 0 | 1.14 s, 0 |
| `mine_multistep`, user | 0.40 s, 27 | 0.44 s, 27 | 0.32 s, 29 | 0.33 s, 29 |
| `mine_multistep`, system | 0.42 s, 21 | 0.46 s, 21 | 0.32 s, 22 | 0.34 s, 22 |
| `mine_recovery`, user | 0.39 s, 41 | 0.42 s, 49 | 0.28 s, 50 | 0.28 s, 50 |
| `mine_recovery`, system | 0.41 s, 55 | 0.42 s, 55 | 0.27 s, 53 | 0.28 s, 53 |

CPU of the same machine, 7 threads, stock 3B, system layout, 20 tasks (two per family), 3 repeats:

| | Stock chat endpoint | Stock server, raw text, no cache | Stock server, raw text, own cache | Full runtime |
|---|---|---|---|---|
| Seconds per task, correct of 20 | 53.1 s, 2 | 39.8 s, 4 | 11.7 s, 4 | 12.1 s, 6 |

What each feature does:

- **Cache reuse matches the stock server's own prompt cache and is not a gain over it.**
  With forced JSON off, runtime against server: 0.60 s against 0.58 s (stock 3B, system),
  0.33 against 0.32 (`mine_multistep`, user), 0.28 against 0.28 and 0.28 against 0.27
  (`mine_recovery`), 11.7 against 11.7 on the CPU. With nothing on, the runtime is as fast as
  the server without cache (0.82 against 0.80 s; 0.42 against 0.42 s). Cache reuse itself is
  worth 1.4x to 1.5x on the GPU and 3.5x on the CPU, in the runtime and in the server alike.
- **Forced JSON helps a model that writes bad calls and does nothing for one that does not.**
  Stock 3B, system layout: 24 → 38 correct, and shorter tasks (0.60 → 0.46 s). `mine_multistep`
  and `mine_recovery`: the grammar refused no token in any run and every count is unchanged.
  On `mine` it hurt: 20 → 9 (11 → 9 under the stricter rule), because that model drops a
  closing brace, and recovering from the parser's error message worked better for it than
  being forced to finish the call.
- **The parser for several calls in one reply** is the difference between the chat endpoint and
  the raw-text rows. The stock server answered HTTP 500 ("output does not match the expected
  format") on 27 to 37 of 100 tasks with the stock model, 26 to 30 with `mine_multistep`, and
  4 to 10 with `mine_recovery`. For `mine_recovery` that costs the chat endpoint 9 tasks on the
  development set and 6 on the held-out set in the user layout (41 against 50; 44 against 50),
  and nothing in the system layout (55 against 53 to 55; 49 against 49).
- **Cache reuse changes a few answers.** The same tasks give 24 or 25, 27 or 29, 53 or 55
  correct with the cache on or off, the same in every repeat and the same in the server and in
  the runtime. llama.cpp's output depends slightly on how tokens are grouped
 ; this is that effect, not a fault of either program.

### 3. What these results do not show

- No advantage in engine speed over the stock server: none was found.
- Nothing about drafting: no drafted run of the runtime has been timed.
- CPU timing exists for the stock model only, on 20 tasks.
- The checker cannot tell a true final sentence from a false one once the state is right.
- One run per correctness row. Timed rows repeated exactly, so run-to-run noise is small, but
  differences of a few tasks between configurations are differences of a few tasks.
- 13 development tasks start with event ids that a newly created event can also get
  (`agent/tasks/README.md`). It happened in 2 recorded task runs, both already wrong.

## Training-data checks

`runtime/results/part6/OVERLAP_NOTE.md`. Each fine-tune's training file was checked for the
agent tasks' questions, people, contact ids and the six tool names, and matched to the file
named in the run's config and log.

| Model | Training file | Examples | Overlaps |
|---|---|---|---|
| `mine` | `train_v1` | 20,000 | 0 (development set) |
| `mine_multistep` | `train_v2_clean` | 33,252 | 0 (development set) |
| `mine_recovery` | `train_v3_recovery` | 39,990 | 0 (development and held-out sets) |

The check finds exact questions, names, ids and tool names. It cannot tell whether generated
practice conversations are built like the ten task families.

