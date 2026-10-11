#!/usr/bin/env bash
# Idle-slot slowdown, step 3: 1, 4, 8 and 16 users on a 16-slot server, plain decoding, three
# passes each, before (pinned commit) and after (pinned commit plus bench/idle_slots/fix_patch.py).
# Then the diagnostic counters with the fix (one pass, 8 users) to count forward passes.
set -uo pipefail
S=/workspace/specedge; PIN=/workspace/llama.cpp/build/bin/llama-server; FIX=/workspace/llama.cpp-fix/build/bin/llama-server
P=f498f864fbc0472004ee1c3616c1188c68eb157f
run() {   # name, users, repeats, server
  echo "$(date -u +%FT%TZ) start $1"
  USERS="$2" SLOTS=16 REPEATS="$3" SERVER="$4" LLAMA_COMMIT="$P" POLICIES=$S/bench/idle_slots/policies.json bash $S/serving/run_load.sh "$1" plain
  echo "$(date -u +%FT%TZ) done $1 (exit $?)"
}
run idle_before_s16 "1 4 8 16" 3 $PIN
run idle_after_s16  "1 4 8 16" 3 $FIX
echo "$(date -u +%FT%TZ) step03 finished"
