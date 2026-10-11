#!/usr/bin/env bash
# Build what needs no model on a rented machine and verify both task sets against the checker.
#     MACHINE=GPU5090B bash runtime/scripts/verify_heldout.sh
# MACHINE names the <NAME>_HOST / <NAME>_PORT pair in .machines.env. Nothing here loads a
# model or touches a GPU: it compiles the tool host and the unit checks (2 jobs), then runs
# verify_tasks.py, which plays each task's reference solution through the runtime's own tools
# and checker. It needs llama.cpp only for its vendored JSON header, fetched at the pinned
# commit when the machine has no checkout yet.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
source "$root/.machines.env"
host_var="${MACHINE:-GPU5090}_HOST"; port_var="${MACHINE:-GPU5090}_PORT"
HOST="${!host_var}"; PORT="${!port_var}"
ssh_opts=(-i "$SSH_KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR)
tar -C "$root" --exclude='runtime/results' --exclude='__pycache__' -czf - runtime agent |
    ssh "${ssh_opts[@]}" "root@$HOST" 'mkdir -p /workspace/specedge_rt/specedge && cd /workspace/specedge_rt/specedge && rm -rf runtime agent && tar -xzf -'
ssh "${ssh_opts[@]}" "root@$HOST" 'bash -s' <<'SH'
set -euo pipefail
cd /workspace/specedge_rt
COMMIT=f498f864fbc0472004ee1c3616c1188c68eb157f
LLAMA=/workspace/llama.cpp
if [ ! -f "$LLAMA/vendor/nlohmann/json.hpp" ]; then
    LLAMA=/workspace/specedge_rt/llama_headers            # source only, never built
    if [ ! -f "$LLAMA/vendor/nlohmann/json.hpp" ]; then
        mkdir -p "$LLAMA" && cd "$LLAMA" && git init -q && git remote add origin https://github.com/ggml-org/llama.cpp
        git fetch -q --depth 1 origin "$COMMIT" && git checkout -q FETCH_HEAD && cd /workspace/specedge_rt
    fi
fi
echo "JSON header from $LLAMA at $(git -C "$LLAMA" rev-parse HEAD)"
cmake -S specedge/runtime/agent -B build_nomodel -DSPECEDGE_WITH_LLAMA=OFF -DLLAMA_DIR="$LLAMA" > build_nomodel.log 2>&1 || { cat build_nomodel.log; exit 1; }
cmake --build build_nomodel -j2 2>&1 | grep -E "error|warning|Built target" || true
PY="$(command -v python3 || echo /venv/main/bin/python)"
A=specedge/agent
echo "== unit checks"; ./build_nomodel/test_units $A/tools
echo "== development set"; sha256sum $A/tasks/tasks_v1.jsonl
"$PY" $A/tasks/verify_tasks.py --agent ./build_nomodel/specedge-toolhost --tasks $A/tasks/tasks_v1.jsonl --contacts $A/tools/contacts.json
echo "== held-out set"; sha256sum $A/tasks/heldout_v1.jsonl $A/tools/contacts_heldout.json
"$PY" $A/tasks/verify_tasks.py --agent ./build_nomodel/specedge-toolhost --tasks $A/tasks/heldout_v1.jsonl --contacts $A/tools/contacts_heldout.json
echo "VERIFY_DONE on $(hostname) at $(date -u +%Y-%m-%dT%H:%MZ)"
SH
