#!/usr/bin/env bash
# The part 7 load runs on the rented GPU machine: one llama-server per policy, then
# serving/loadgen.py at every user count.
#
# By default requests go straight to llama-server, as in the 7 October load check, so the rows
# can be set beside it. GATEWAY=1 sends them through the metrics gateway instead; those rows
# are named <policy>_via_gateway and show what the gateway costs.
#
#     bash serving/run_load.sh <run name> [policy ...]          # default: every policy in policies.json
#     USERS="1 4 8 16" SLOTS=16 REPEATS=3 bash serving/run_load.sh gpu5090_v1
#
# Timing run: run nothing else on the machine. The load
# generator checks the machine is quiet before every user count and stops if it is not.
# Finished <policy>_u<users>.json files are skipped, so a stopped run can be started again.
# SLOTS=0 starts a server with exactly as many slots as users for each user count.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/.." && pwd)"
run="${1:?give a run name}"; shift
SERVER="${SERVER:-/workspace/llama.cpp/build/bin/llama-server}"
LLAMA_COMMIT="${LLAMA_COMMIT:-$(cut -d' ' -f1 /workspace/llama_commit.txt)}"
USERS="${USERS:-1 4 8 16}"; SLOTS="${SLOTS:-16}"; REPEATS="${REPEATS:-3}"
KIND_DIR="${KIND_DIR:-$root/data/first_hour}"; WORKLOAD="${WORKLOAD:-}"
PORT="${PORT:-8091}"; GATEWAY_PORT="${GATEWAY_PORT:-8090}"; GATEWAY="${GATEWAY:-0}"
POLICIES="${POLICIES:-$here/policies.json}"
out="$root/serving/results/part7/$run"; mkdir -p "$out"
field() { python3 -c "import json,sys; d=json.load(open('$POLICIES')); print(d$1)"; }
model="$(field "['model']")"; common="$(field "['common_args']")"; context="$(field "['context_per_slot']")"
if [ "$#" -eq 0 ]; then set -- $(python3 -c "import json; print(' '.join(json.load(open('$POLICIES'))['policies']))"); fi
hardware="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1); $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
if [ -n "$WORKLOAD" ]; then workload=(--workload "$WORKLOAD"); else workload=(--kind-dir "$KIND_DIR"); fi

one_server() {   # policy, slots, user counts...
    local policy="$1" slots="$2"; shift 2
    local name="$policy" url="http://127.0.0.1:$PORT"
    if [ "$GATEWAY" = 1 ]; then name="${policy}_via_gateway"; url="http://127.0.0.1:$GATEWAY_PORT"; fi
    local todo=()
    for users in "$@"; do [ -f "$out/${name}_u${users}.json" ] || todo+=("$users"); done
    if [ "${#todo[@]}" -eq 0 ]; then echo "$name slots=$slots: all done, skipped"; return 0; fi
    local extra; extra="$(field "['policies']['$policy']")"
    local args="-m $model --host 127.0.0.1 --port $PORT -c $((context * slots)) -np $slots $common $extra"
    local log="$out/${name}_slots${slots}.server.log"
    # shellcheck disable=SC2086
    "$SERVER" $args > "$log" 2>&1 &
    local server=$! gateway=""
    if [ "$GATEWAY" = 1 ]; then
        python3 "$root/monitoring/gateway/gateway.py" --listen "127.0.0.1:$GATEWAY_PORT" --upstream "http://127.0.0.1:$PORT" \
            > "$out/${name}_slots${slots}.gateway.log" 2>&1 &
        gateway=$!
    fi
    # shellcheck disable=SC2064
    trap "kill $server $gateway 2>/dev/null || true" EXIT
    for _ in $(seq 1 600); do
        curl -fs "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
        kill -0 "$server" 2>/dev/null || { echo "llama-server exited; see $log" >&2; return 1; }
        sleep 1
    done
    local status=0
    python3 "$here/loadgen.py" --url "$url" --policy "$name" --server-args "$args" \
        --out "$out" --users "${todo[@]}" --repeats "$REPEATS" "${workload[@]}" \
        --hardware "$hardware" --llama-commit "$LLAMA_COMMIT" --quiet-check --server-port "$PORT" || status=$?
    curl -fs "http://127.0.0.1:$PORT/metrics" > "$out/${name}_slots${slots}.server_metrics.txt" || true
    if [ "$GATEWAY" = 1 ]; then
        curl -fs "http://127.0.0.1:$GATEWAY_PORT/metrics" > "$out/${name}_slots${slots}.gateway_metrics.txt" || true
    fi
    # shellcheck disable=SC2086
    kill $server $gateway 2>/dev/null || true
    # shellcheck disable=SC2086
    wait $server $gateway 2>/dev/null || true
    trap - EXIT
    python3 "$here/step_stats.py" "$log" --users "${todo[@]}" --repeats "$REPEATS" \
        > "$out/${name}_slots${slots}.step_stats.md" 2>&1 || true
    return "$status"
}

for policy in "$@"; do
    if [ "$SLOTS" = 0 ]; then
        for users in $USERS; do one_server "$policy" "$users" "$users"; done
    else
        # shellcheck disable=SC2086
        one_server "$policy" "$SLOTS" $USERS
    fi
done
python3 "$here/report.py" "$out" --write
