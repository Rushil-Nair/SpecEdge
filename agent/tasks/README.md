# The agent tasks: two sets, reported apart

| | Development set | Held-out set |
|---|---|---|
| File | `tasks_v1.jsonl` | `heldout_v1.jsonl` |
| sha256 | `0d3a4a362424aee44570762c23a8b357996f803cda7f3aa62605d8e43cda8043` | `49acc1e7161c5021d6cbcbb0b7d3d2314ef6f826dd0f3da64d409cdeff85e5d3` |
| Contacts | `agent/tools/contacts.json` | `agent/tools/contacts_heldout.json` |
| Generator, seed | `build_tasks.py`, 20261008 | `build_heldout.py`, 20270211 |
| Tasks | 100, ten families of ten | 100, the same ten families of ten |
| Use | Read task by task while studying failures; fine-tuning decisions may be based on it | **No model was run on it until the models were final.** What a model is finally judged on |

Both sets use the same six tools (`agent/tools/schemas`), the same checker and the same system
prompt. Results are always given per set, never added together.

Decided on 2026-10-08: the held-out set keeps its half-and-half wording, and it is
the half in the development set's wording (50 tasks, `"wording": "as_dev"`) that is compared
with the development set. The new-wording half is reported beside it. `tasks_v1` stays frozen
as it is, with the id clash below documented; there is no v2.

## What differs in the held-out set

- Twelve other people with ids `c_201` to `c_212`; event ids from `e_3001`; other event titles;
  dates in February 2027; start times on the quarter hour; other lengths, rooms, amounts and notes.
- Calendars with one to four events on the day in question and one to three on another day.
- In every family five tasks use the development set's wording and five a new wording (field
  `wording`), so the two halves can be compared.
- `build_heldout.py` stops if any question, person, name part, contact id, event id, event title
  or date of this set appears in the development set. It found 0 of each.

## State of the held-out set: frozen

Frozen on 2026-10-08 at 10:09 UTC. `heldout_v1.jsonl` has sha256
`49acc1e7161c5021d6cbcbb0b7d3d2314ef6f826dd0f3da64d409cdeff85e5d3` and
`agent/tools/contacts_heldout.json` has sha256
`e1f19e297215d985749bd1cbd0200bbd9a24bc1ef312b77e330e22b274ba01cd`. A change to either needs a
new version number.

Verified with `runtime/scripts/verify_heldout.sh` on the second rented RTX 5090 (no model
loaded, GPU not used), through the runtime's own tools and checker built there from this
repository with the JSON header of llama.cpp `f498f864`:

| Check | Development set | Held-out set |
|---|---|---|
| Tasks | 100 | 100 |
| Reference solution runs without a tool error and is accepted | 100 | 100 |
| Doing nothing is refused | 100 | 100 |
| Tasks with a decimal argument; with decimals cut, refused | 57; 57 | 54; 54 |
| The reference solution plus one extra message is refused | 100 | 100 |
| Problems | 0 | 0 |

The first verification run reported every held-out task as a problem. The fault was in
`verify_tasks.py`, not in the tasks: its extra-message test wrote to `c_101`, a development-set
contact that does not exist in the held-out directory, so the message was never sent. It now
writes to a contact of the set being checked and fails if that message cannot be sent. The
task file and its checksum did not change.

- **No model has been run on this set.** `runtime/scripts/run_matrix.sh` refuses `SET=heldout`
  unless `ALLOW_HELDOUT=yes` is given.
- Written by a generator; its tasks were not read one by one, and are not to be while models
  are being developed.
- Fine-tuning data must be checked against this set as well as the development set
  (`check_training_overlap.py`, which now covers both by default): the 200 questions, the 24
  people and their contact ids, and the six tool names.

## A known flaw in the development set

13 development tasks (every family's task 10, and task 9 of three families) start with events
whose ids are `e_501` or above. The tools give new events ids from `e_501`, so in those tasks a
newly created event can get an id that is already on the calendar. In the result files so far
this happened in 2 task runs (`mine`, user layout, forced JSON on and off), both already wrong.
The set is frozen and has not been changed. The held-out set starts its ids at `e_3001`.
