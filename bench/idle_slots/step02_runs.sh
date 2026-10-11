#!/usr/bin/env bash
# Idle-slot slowdown, step 2: how many forward passes per decode step (diagnostic build), and
# does the unified KV cache avoid the slowdown?
set -uo pipefail
S=/workspace/specedge; PIN=/workspace/llama.cpp/build/bin/llama-server; DIAG=/workspace/llama.cpp-diag/build/bin/llama-server
P=f498f864fbc0472004ee1c3616c1188c68eb157f
run() {   # name, slots, repeats, server, policy
  echo "$(date -u +%FT%TZ) start $1"
  USERS="8" SLOTS="$2" REPEATS="$3" SERVER="$4" LLAMA_COMMIT="$P" POLICIES=$S/bench/idle_slots/policies.json bash $S/serving/run_load.sh "$1" "$5"
  echo "$(date -u +%FT%TZ) done $1 (exit $?)"
}
run idle_diag2_s16 16 1 $DIAG plain
run idle_diag2_s8   0 1 $DIAG plain
run idle_kvu_s16   16 3 $PIN plain_kvu
run idle_kvu_s8     0 3 $PIN plain_kvu
echo "$(date -u +%FT%TZ) step02 finished"
