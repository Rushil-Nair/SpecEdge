# The drafting policy

`policy.h` / `policy.cpp`: plain C++17 with no llama.cpp, no model and no clock. For each
request the caller asks `decide(requests_in_flight)` and gets a draft block length or "decode
plainly"; afterwards it reports `observe(block, drafted, accepted)` or `on_error()`.

Decided on 2026-10-08: the queue rule is the policy; the acceptance rule is kept only as a
fallback. The agent runtime links this library for its drafting loop (`runtime/agent/src/engine_draft.cpp`).

Three rules, checked in this order:

1. **Errors.** After a drafting error, the next 20 requests are decoded plainly. Three errors
   within 200 requests turn drafting off until `reset()`.
2. **Queue depth.** Up to 4 requests in flight: block 16. 5 to 8: block 4. More than 8: no
   drafting.
3. **Rolling acceptance.** The share of drafted tokens accepted, over the last 20,000 drafted
   tokens, tracked separately per block length. Below the band's lower threshold drafting goes
   off; then one request in four is drafted as a probe, and one probe accepted at or above the
   upper threshold turns drafting back on.

**There is no timing run of anything here.** The policy numbers below are a replay of times
recorded earlier in this project.

## Connected to the runtime's drafting (2026-10-08)

| File | What it is | State |
|---|---|---|
| `drafting.h`, `drafting.cpp` | One reply written through the policy: asks it once per request, runs the draft-and-verify loop or the plain loop, tells it what was drafted and accepted, and falls back to plain decoding when the drafter throws. Reaches the engine through two interfaces (`Target`, `DraftSource`), so it has no llama.cpp in it | Built and tested on the laptop: `tests/test_drafting.cpp`, 2,393 checks with made-up models |
| `../agent/src/engine_draft.*` | The llama.cpp engine behind those interfaces: `EngineTarget` (the served model, with its forced-JSON grammar) and `LlamaDraft` (a second, smaller model of the same vocabulary drafting greedily) | Built on the first 5090 after the timing ended; one functional run (below), no timing |
| `../agent/src/policy_config.*`, `policies/*.json` | Policy settings from a JSON file | Built; 5 unit checks in `test_units` |
| `specedge-agent run --draft-model M2 --policy FILE [--requests-in-flight N]` | Turns it on in the tool loop. Each turn's record gets the decision, block length, tokens drafted and accepted | Built; used in the functional run |

What the loop guarantees, and what the tests check: every token written is the target's own
choice in the pass that checked it. With a made-up target whose choices do not depend on how
tokens are grouped, the reply is identical to plain decoding for every drafter tried (always
right, always wrong, wrong every second, third or fifth token, returning too much, drafting
past the end, failing mid-reply). In llama.cpp the choices do depend slightly on grouping, so
there the match with plain decoding has to be measured.

One functional run on a real model, not timed (first 5090, GPU): stock 3B with the stock 1B
(F16) as drafter, a test policy that always drafts 4 tokens, 10 development tasks, system
layout. With forced JSON on, 1,201 of 1,600 drafted tokens were accepted and 3 tasks were
correct, as with plain decoding; with it off, 1,466 of 1,837 and 2, as with plain decoding.
Transcripts were identical to plain decoding in 6 of 10 and 4 of 10 tasks.

Still to do before a live test:
- Thresholds for the 1B drafting for the 3B do not exist: `--draft-model` refuses to run without
  `--policy`, and the only policy file is for Qwen3-8B with DFlash.
- EAGLE-3 and DFlash drafters need llama.cpp's `common` library; that `DraftSource` is not written.
- Queue depth is a number given on the command line. The runtime serves one request at a time;
  a real queue exists only in a server.

## Where the numbers come from

`traces/build_traces.py` read the load-check results of the speculative-decoding study (not in
this repository: Qwen3-8B 4-bit, z-lab DFlash block 16,
RTX 5090, llama.cpp `f498f864`, five kinds of text, 200 requests, three passes, at 1, 4 and 16
users, and block 4 with one user). It wrote the three trace files and `traces/ANALYSIS.md`,
which are included and are what the unit test replays.

| Setting | Value | Read off |
|---|---|---|
| Lower threshold, block 16, 1 in flight | 0.10 | Of recorded requests under 0.10 accepted, 43 of 44 were slower than plain; at or above, 1 of 140 |
| Lower threshold, block 16, up to 4 in flight | 0.12 | Under 0.12: 50 of 54 slower; at or above 0.15: 1 of 107 |
| No drafting above 8 in flight | | At 16 users drafting took 2.11 times plain decoding's generation time over the 200 requests; code, the best kind, 0.96 |
| Block 4 for 5 to 8 in flight, thresholds 0.40 / 0.45 | | **Not measured under load.** The thresholds are the one-user block-4 break-even. 8 is where llama.cpp's CUDA backend changes kernel |

The defaults belong to this model, drafter, card and build.

## Replay on the recorded traces

`policy_replay traces`. A drafted request costs what drafting took when recorded, a plain one
what plain decoding took (generation time, median of three passes). "Oracle" takes the faster
of the two for every request, which no policy can know in advance. **These are estimates from
recorded times, not measurements**: under load, one request's recorded time depends on what the
other requests were doing at that moment.

| Trace | Requests | Never draft, s | Always draft, s | Policy, s | Oracle, s | Requests drafted |
|---|---|---|---|---|---|---|
| 1 user, recorded order (five kinds mixed) | 200 | 156.9 | 119.7 | 119.7 | 111.3 | 200 |
| 4 users, recorded order | 200 | 567.1 | 469.8 | 472.4 | 426.4 | 196 |
| 16 users, recorded order | 200 | 481.0 | 1017.1 | 481.0 | 467.3 | 0 |
| 1 user, code only | 40 | 31.9 | 11.1 | 11.1 | 11.1 | 40 |
| 1 user, tool calls only | 40 | 8.3 | 4.0 | 4.0 | 4.0 | 40 |
| 1 user, chat only | 40 | 24.0 | 21.6 | 21.6 | 20.8 | 40 |
| 1 user, Chinese only | 40 | 27.4 | 34.0 | 29.5 | 26.4 | 12 |
| 1 user, reasoning only | 40 | 65.2 | 49.0 | 49.0 | 49.0 | 40 |
| 1 user: code, then Chinese, then code | 120 | 91.2 | 56.3 | 54.7 | 48.7 | 99 |
| 4 users: code, then Chinese, then code | 120 | 332.0 | 219.8 | 213.5 | 185.6 | 99 |
| 1 user: tool calls, chat, Chinese, reasoning in turn | 160 | 125.0 | 108.6 | 105.7 | 100.2 | 133 |

What the replay says:

- **The queue rule is where the gain is.** At 16 users always drafting takes 2.1 times as long
  as the policy; at 1 and 4 users never drafting takes 1.2 to 1.3 times as long. No single
  fixed setting is right at all three loads.
- **On evenly mixed traffic the acceptance rule gains nothing.** The policy equals "always
  draft" at 1 user and is 0.6% behind it at 4 users. The oracle, which knows each request's outcome in advance, is 8% to 11% ahead.
- **When traffic changes for a stretch it gains a little:** 3% on the code, Chinese, code traces.
  It noticed the Chinese stretch after 15 requests at 1 user (14 at 4 users).
- On Chinese alone it removes most of the loss (29.5 s against 34.0 s; plain is 27.4 s). The
  rest is the cost of probing.

## The window is a trade

Same replay with other windows (probe every 4, one good probe to resume):

| Window, drafted tokens | Mixed, 1 user, s | Mixed, 4 users, s | Code, Chinese, code at 1 user, s | Chinese requests before drafting stops |
|---|---|---|---|---|
| 1,500 | 128.7 | 502.3 | 52.7 | 3 |
| 3,000 | 125.5 | 487.1 | 53.5 | 4 |
| 6,000 | 125.1 | 495.5 | 53.3 | 6 |
| 10,000 | 121.0 | 486.4 | 54.3 | 8 |
| 20,000 (default) | 119.7 | 472.4 | 54.7 | 15 |
| always draft | 119.7 | 469.8 | 56.3 | |

A short window is filled by one or two badly drafted requests (a Chinese request drafts about
1,100 tokens to write 192), so on mixed traffic it turns drafting off while drafting still
pays overall. The default favours not losing on mixed traffic over noticing quickly.

## Limits

- The window, the probe interval and the probe count were chosen on the same traces the tests
  replay. There is no held-out trace: three load levels of one 200-request workload are all
  that is recorded.
- The traces hold totals per request. The policy can be called per decoding step, but that
  use is untested.
- Nothing between 4 and 16 users, and no block shorter than 16 under load, has been recorded.
- EAGLE-3 is not covered: its accepted share is on a different scale and it has no load traces.
- A probe is a whole request drafted at the band's block length.

## Build and test

    cmake -S runtime/controller -B build_controller && cmake --build build_controller -j2
    ./build_controller/test_policy runtime/controller/traces      # 374 checks
    ./build_controller/test_drafting                              # 2,393 checks
    ./build_controller/policy_replay runtime/controller/traces
