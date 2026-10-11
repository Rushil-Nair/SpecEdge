# Where drafting stops paying — from recorded runs

Built by `build_traces.py` from the load-check and hard-text results of the speculative-decoding study (not included in this repository). Qwen3-8B 4-bit on the RTX 5090, llama.cpp `f498f864`. Per request: generation time with drafting divided by generation time with plain decoding (median of three passes), against the share of drafted tokens that were accepted. Requests shorter than 32 tokens are left out of the per-request tables.

## DFlash block 16, 1 user in flight

| Kind | Accepted share | Drafting time / plain time |
|---|---|---|
| code | 0.462 | 0.35 |
| tool_calls | 0.300 | 0.48 |
| chat | 0.118 | 0.90 |
| chinese | 0.066 | 1.24 |
| reasoning | 0.155 | 0.75 |
| all 200 requests | | 0.76 |

| Accepted share | Requests below | of them slower than plain | Requests at or above | of them slower than plain |
|---|---|---|---|---|
| 0.06 | 17 | 17 | 167 | 27 |
| 0.08 | 31 | 31 | 153 | 13 |
| 0.10 | 44 | 43 | 140 | 1 |
| 0.12 | 54 | 44 | 130 | 0 |
| 0.15 | 76 | 44 | 108 | 0 |
| 0.20 | 105 | 44 | 79 | 0 |
| 0.30 | 131 | 44 | 53 | 0 |
| 0.40 | 150 | 44 | 34 | 0 |

## DFlash block 16, 4 users in flight

| Kind | Accepted share | Drafting time / plain time |
|---|---|---|
| code | 0.417 | 0.40 |
| tool_calls | 0.301 | 0.50 |
| chat | 0.119 | 1.02 |
| chinese | 0.066 | 1.34 |
| reasoning | 0.153 | 0.82 |
| all 200 requests | | 0.83 |

| Accepted share | Requests below | of them slower than plain | Requests at or above | of them slower than plain |
|---|---|---|---|---|
| 0.06 | 18 | 18 | 166 | 36 |
| 0.08 | 32 | 32 | 152 | 22 |
| 0.10 | 43 | 42 | 141 | 12 |
| 0.12 | 54 | 50 | 130 | 4 |
| 0.15 | 77 | 53 | 107 | 1 |
| 0.20 | 106 | 54 | 78 | 0 |
| 0.30 | 134 | 54 | 50 | 0 |
| 0.40 | 154 | 54 | 30 | 0 |

## DFlash block 16, 16 users in flight

| Kind | Accepted share | Drafting time / plain time |
|---|---|---|
| code | 0.455 | 0.96 |
| tool_calls | 0.297 | 1.30 |
| chat | 0.118 | 2.53 |
| chinese | 0.066 | 3.32 |
| reasoning | 0.155 | 2.07 |
| all 200 requests | | 2.11 |

| Accepted share | Requests below | of them slower than plain | Requests at or above | of them slower than plain |
|---|---|---|---|---|
| 0.06 | 19 | 19 | 165 | 133 |
| 0.08 | 33 | 33 | 151 | 119 |
| 0.10 | 40 | 40 | 144 | 112 |
| 0.12 | 55 | 55 | 129 | 97 |
| 0.15 | 76 | 76 | 108 | 76 |
| 0.20 | 104 | 104 | 80 | 48 |
| 0.30 | 129 | 129 | 55 | 23 |
| 0.40 | 151 | 147 | 33 | 5 |

## DFlash block 4, 1 user (Chinese and long reasoning only)

| Kind | Accepted share | Drafting time / plain time |
|---|---|---|
| chinese | 0.197 | 1.47 |
| reasoning | 0.425 | 0.97 |

| Accepted share | Requests below | of them slower than plain | Requests at or above | of them slower than plain |
|---|---|---|---|---|
| 0.15 | 19 | 19 | 98 | 55 |
| 0.20 | 35 | 35 | 82 | 39 |
| 0.25 | 46 | 46 | 71 | 28 |
| 0.30 | 51 | 51 | 66 | 23 |
| 0.35 | 56 | 56 | 61 | 18 |
| 0.40 | 72 | 72 | 45 | 2 |
| 0.45 | 99 | 74 | 18 | 0 |
| 0.50 | 109 | 74 | 8 | 0 |

## Not in any recorded run

- Block 4 (or any block shorter than 16) with more than one user.
- Any point between 4 and 16 users.
- Acceptance per decoding step: the files hold totals per request.
