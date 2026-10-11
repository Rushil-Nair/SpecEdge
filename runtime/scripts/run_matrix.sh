#!/usr/bin/env bash
# Run the part 6 table on the rented machine: every configuration, REPEATS times, one at a
# time. Run it on the machine from /workspace/specedge_rt:
#
#   DEVICE=cpu THREADS=4 REPEATS=1 TIMING=0 LAYOUT=user bash specedge/runtime/scripts/run_matrix.sh
#   DEVICE=gpu REPEATS=3 TIMING=1 bash specedge/runtime/scripts/run_matrix.sh      # booked machine only
#
# LAYOUT is where the prompt puts the tool list: user (the template's default) or system.
# ONLY=name,name runs just those configurations. IDS=id,id runs just those tasks.
# TIMING=1 marks the files as timing runs. Use it only when the machine is booked in
# nothing else is running on the device being timed.
# A finished configuration is skipped when the script is started again, so it can resume.
# Results go to results/part6/ here; copy them down to runtime/results/part6/.
set -euo pipefail
cd "$(dirname "$0")/../../.."                       # /workspace/specedge_rt
DEVICE="${DEVICE:-cpu}"; THREADS="${THREADS:-4}"; REPEATS="${REPEATS:-1}"; TIMING="${TIMING:-0}"
MODEL="${MODEL:-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"; TAG="${TAG:-stock3b_q4km}"
LIMIT="${LIMIT:-0}"; ONLY="${ONLY:-}"; NOTE="${NOTE:-}"; LAYOUT="${LAYOUT:-user}"; IDS="${IDS:-}"
ids=(); [ -n "$IDS" ] && ids=(--ids "$IDS")
A=specedge/agent; OUT=results/part6; mkdir -p "$OUT"
# SET=dev is the development set (tasks_v1). SET=heldout is the held-out set: no model is run
# on it until the models are final, so it also needs ALLOW_HELDOUT=yes.
SET="${SET:-dev}"
if [ "$SET" = heldout ]; then
    [ "${ALLOW_HELDOUT:-}" = yes ] || { echo "refusing: the held-out set is only run on final models (ALLOW_HELDOUT=yes)"; exit 2; }
    TASKS=$A/tasks/heldout_v1.jsonl; CONTACTS=$A/tools/contacts_heldout.json; TAG="heldout_$TAG"
else
    TASKS=$A/tasks/tasks_v1.jsonl; CONTACTS=$A/tools/contacts.json
fi
NGL=0; GPU=""
if [ "$DEVICE" = gpu ]; then
    NGL=99; GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    export CUDA_VISIBLE_DEVICES=""               # CPU rows: the GPU is hidden from this process
fi
MODEL_SHA="$(sha256sum "$MODEL" | cut -d' ' -f1)"; TASKS_SHA="$(sha256sum "$TASKS" | cut -d' ' -f1)"
PY=/venv/main/bin/python

# A timing run needs the machine to itself. Before each one: no other process on the GPU and a
# 1-minute load under 1.5; waits up to 10 minutes for that, then gives up on the whole matrix.
# What was seen is written to <run>.quiet.txt beside the result.
quiet() {
    [ "$TIMING" = 1 ] || return 0
    local tries=0 others load
    while :; do
        others="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null | grep -v specedge-agent || true)"
        load="$(cut -d' ' -f1 /proc/loadavg)"
        if [ -z "$others" ] && awk -v l="$load" 'BEGIN { exit !(l < 1.5) }'; then
            echo "$(date -u +%H:%M:%S) load $load, no other GPU process" > "$OUT/$1.quiet.txt"; return 0
        fi
        tries=$((tries + 1))
        [ "$tries" -gt 20 ] && { echo "NOT QUIET before $1: load $load, GPU processes: $others" | tee "$OUT/$1.quiet.txt"; exit 3; }
        sleep 30
    done
}

runtime() {   # name cache forced_json tool_results
    local name="$1_${LAYOUT}_${TAG}_${DEVICE}_r$5" out
    out="$OUT/$name.json"; [ -s "$out" ] && { echo "skip $name"; return; }
    quiet "$name"
    ./build/specedge-agent run --model "$MODEL" --tools $A/tools/schemas --contacts $CONTACTS \
        --tasks $TASKS --system $A/tasks/system_prompt.txt --threads "$THREADS" --ngl "$NGL" --gpu "$GPU" \
        --cache "$2" --forced-json "$3" --tool-results "$4" --tools-in "$LAYOUT" --limit "$LIMIT" "${ids[@]}" --timing "$TIMING" --note "$NOTE" \
        --model-sha256 "$MODEL_SHA" --tasks-sha256 "$TASKS_SHA" --name "$name" --out "$out.part" 2> "$OUT/$name.log"
    mv "$out.part" "$out"; tail -1 "$OUT/$name.log"
}
control() {   # name mode cache
    local name="$1_${LAYOUT}_${TAG}_${DEVICE}_r$4" out
    out="$OUT/$name.json"; [ -s "$out" ] && { echo "skip $name"; return; }
    quiet "$name"
    $PY $A/control/server_loop.py --model "$MODEL" --agent ./build/specedge-agent --tasks $TASKS --contacts $CONTACTS --threads "$THREADS" \
        --ngl "$NGL" --gpu "$GPU" --mode "$2" --cache "$3" --tools-in "$LAYOUT" --limit "$LIMIT" "${ids[@]}" --timing "$TIMING" --note "$NOTE" \
        --model-sha256 "$MODEL_SHA" --tasks-sha256 "$TASKS_SHA" --name "$name" --out "$out.part" > "$OUT/$name.log" 2>&1
    mv "$out.part" "$out"
    tail -1 "$OUT/$name.log"
}
want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }

for r in $(seq 1 "$REPEATS"); do
    want control_chat        && control control_chat        chat 0 "$r"
    want control_raw         && control control_raw         raw  0 "$r"
    want control_raw_cache   && control control_raw_cache   raw  1 "$r"
    want rt_plain            && runtime rt_plain            0 0 stock "$r"
    want rt_cache            && runtime rt_cache            1 0 stock "$r"
    want rt_json             && runtime rt_json             0 1 stock "$r"
    want rt_cache_json       && runtime rt_cache_json       1 1 stock "$r"
    want rt_cache_json_plainresults && runtime rt_cache_json_plainresults 1 1 plain "$r"
done
echo "matrix done: $DEVICE layout $LAYOUT repeats $REPEATS timing $TIMING"
