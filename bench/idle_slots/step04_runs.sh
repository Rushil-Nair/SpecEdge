#!/usr/bin/env bash
# Idle-slot fix, step 4 (runs after step 3): forward-pass counters with the fix, the
# output-match check, the prompt-cache check, and an attempt to install Nsight Systems
# (given up after 15 minutes).
set -uo pipefail
S=/workspace/specedge; PIN=/workspace/llama.cpp/build/bin/llama-server; FIX=/workspace/llama.cpp-fix/build/bin/llama-server
DIAG=/workspace/llama.cpp-diag/build/bin/llama-server; P=f498f864fbc0472004ee1c3616c1188c68eb157f; PY=/workspace/venv_train/bin/python
O=$S/results/idle_slots/checks; mkdir -p $O
until grep -q "step03 finished" /workspace/logs/idle_step03.log; do sleep 20; done
echo "$(date -u +%FT%TZ) diag build with the fix"
$PY $S/bench/idle_slots/fix_patch.py /workspace/llama.cpp-diag && cmake --build /workspace/llama.cpp-diag/build --config Release -j 14 --target llama-server > /workspace/logs/diag_build4.log 2>&1
echo "$(date -u +%FT%TZ) start idle_diagfix_s16"
USERS="8" SLOTS=16 REPEATS=1 SERVER=$DIAG LLAMA_COMMIT=$P POLICIES=$S/bench/idle_slots/policies.json bash $S/serving/run_load.sh idle_diagfix_s16 plain
cd $S/bench/idle_slots
for pair in "pinned $PIN" "patched $FIX"; do set -- $pair
  echo "$(date -u +%FT%TZ) match $1"; SPECEDGE_PORT=8377 $PY checks.py match --server $2 --label $1 --out $O
done
SPECEDGE_PORT=8377 $PY checks.py compare --out $O
for pair in "pinned $PIN" "patched $FIX"; do set -- $pair
  echo "$(date -u +%FT%TZ) cache $1"; SPECEDGE_PORT=8377 $PY checks.py cache --server $2 --label $1 --out $O
done
echo "$(date -u +%FT%TZ) timing and checks finished; trying to install nsys"
(timeout 900 bash -c "apt-get update -qq > /workspace/logs/nsys_install.log 2>&1; apt-get install -y -qq nsight-systems-cli >> /workspace/logs/nsys_install.log 2>&1 || apt-get install -y -qq nsight-systems >> /workspace/logs/nsys_install.log 2>&1"; echo "nsys install exit $? $(which nsys 2>/dev/null)")
echo "$(date -u +%FT%TZ) step04 finished"
