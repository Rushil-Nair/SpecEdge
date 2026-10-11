#!/usr/bin/env bash
# Idle-slot slowdown, steps 0 and 1: 8 users on a 16-slot server against 8 users on an 8-slot
# server, plain decoding, with the load generator (serving/run_load.sh, 200 requests of
# the five-kind mix per pass, three passes), for
#   pinned         llama.cpp f498f864 as built for the project
#   master         the latest master, built the same way
#   nographs       pinned, CUDA graphs off            (GGML_CUDA_DISABLE_GRAPHS=1)
#   noreuse        pinned, llama graph reuse off      (LLAMA_GRAPH_REUSE_DISABLE=1)
#   diag           pinned plus bench/idle_slots/diag_patch.py (counters only), one pass
# Each finished run is skipped when the script is started again.
#
#     tmux new-session -d -s idle "bash /workspace/specedge/bench/idle_slots/step01_runs.sh > /workspace/logs/idle_step01.log 2>&1"
set -uo pipefail
S=/workspace/specedge; PIN=/workspace/llama.cpp/build/bin/llama-server
MASTER=/workspace/llama.cpp-master/build/bin/llama-server; DIAG=/workspace/llama.cpp-diag/build/bin/llama-server
MASTER_COMMIT=$(cut -d' ' -f1 /workspace/llama_master_commit.txt)
run() {   # name, slots (0 = as many as users), repeats, server, commit, [VAR=value ...]
  local name="$1" slots="$2" repeats="$3" server="$4" commit="$5"; shift 5
  echo "$(date -u +%FT%TZ) start $name"
  env "$@" USERS="8" SLOTS="$slots" REPEATS="$repeats" SERVER="$server" LLAMA_COMMIT="$commit" bash $S/serving/run_load.sh "$name" plain
  echo "$(date -u +%FT%TZ) done $name (exit $?)"
}
P=f498f864fbc0472004ee1c3616c1188c68eb157f
run idle_pinned_s16   16 3 $PIN $P X=1
run idle_pinned_s8     0 3 $PIN $P X=1
run idle_master_s16   16 3 $MASTER $MASTER_COMMIT X=1
run idle_master_s8     0 3 $MASTER $MASTER_COMMIT X=1
run idle_nographs_s16 16 3 $PIN $P GGML_CUDA_DISABLE_GRAPHS=1
run idle_nographs_s8   0 3 $PIN $P GGML_CUDA_DISABLE_GRAPHS=1
run idle_noreuse_s8    0 3 $PIN $P LLAMA_GRAPH_REUSE_DISABLE=1
run idle_diag_s16     16 1 $DIAG $P X=1
run idle_diag_s8       0 1 $DIAG $P X=1
echo "$(date -u +%FT%TZ) step01 finished"
