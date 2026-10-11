#!/usr/bin/env bash
# Part 12, shortened study for one model, ON the A100 (2026-10-08, for mine_recovery):
#
#     tmux new-session -d -s study2 "bash /workspace/specedge/scripts/part12_short_a100.sh <name> <adapter dir> <calibration .jsonl> <expected correct of 200> > /workspace/specedge/results/part12/<name>_study.log 2>&1"
#
#   merge        the LoRA add-on into the base weights on this machine (base config kept)
#   float        control and sanity check: full precision through Qualcomm's own engine, the
#                fixed 200 tasks. The script stops here if the control is more than 10 tasks
#                away from what the same model scores on those tasks in llama.cpp: that is how
#                a wrong config (the RoPE fault) shows before the quantizer is paid for.
#   recal        Qualcomm's w4a16 recipe with tool-call calibration text taken from the model's
#                own training file (nothing from any test set)
#   aihub        export, compile, link and profile every part on the hosted Galaxy S25,
#                started in the background beside the scoring step
#   recal_json   that checkpoint on the 200 tasks with forced JSON
#
# The default-calibration row and the single-fix rows are not run: mine_multistep's study
# compared those settings. Nothing else is started beside the quantizer. Steps leave markers
# in /workspace/part12/<name>/done; start the same command again after a failure.
set -uo pipefail
name="$1"; adapter="$2"; calibration="$3"; expected="$4"
W=/workspace; P=$W/part12/$name; R=$W/specedge/results/part12; mkdir -p "$P/done" "$R" $W/tmp
BASE=$W/models/llama/Llama-3.2-3B-Instruct; QAI=$W/venv_qai/bin/python; S=$W/specedge/npu/score_quantized.py
export TMPDIR=$W/tmp HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
step() {
  local marker="$1" log="$2"; shift 2
  [ -f "$P/done/$marker" ] && { echo "$(date -u +%FT%TZ) skip $marker"; return 0; }
  echo "$(date -u +%FT%TZ) start $marker"
  if "$@" > "$log" 2>&1; then date -u +%FT%TZ > "$P/done/$marker"; echo "$(date -u +%FT%TZ) done $marker"
  else echo "$(date -u +%FT%TZ) FAILED $marker (see $log)"; return 1; fi
}
step merge "$R/${name}_merge.log" $W/venv_merge/bin/python $W/specedge/training/merge.py --base $BASE --adapter "$adapter" --out "$P/merged" || exit 1
step float "$R/${name}_float_sub200.log" $QAI $S --float --checkpoint "$P/merged" --name "${name}_float_sub200" --subset 200 || exit 1
got=$(python3 -c "import json; print(sum(bool(r['fully_correct']) for r in json.load(open('$R/${name}_float_sub200.json'))['rows']))")
echo "$(date -u +%FT%TZ) control: $got of 200 correct (llama.cpp F16 on the same tasks: $expected)"
if [ $((got - expected)) -gt 10 ] || [ $((expected - got)) -gt 10 ]; then
  echo "$(date -u +%FT%TZ) FAILED sanity (the control is more than 10 tasks from the llama.cpp score; the quantizer was not started)"; exit 1
fi
step recal "$R/${name}_quantize_toolcal.log" /usr/bin/time -v $QAI $W/specedge/npu/quantize_with_calibration.py --calibration "$calibration" \
     --precision w4a16 --checkpoint "$P/merged" --output-dir "$P/w4a16_toolcal" || exit 1
if [ -f ~/.qai_hub/client.ini ]; then
  ( step aihub "$R/${name}_aihub_export.log" env -u HF_HUB_OFFLINE $QAI -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
        --checkpoint "$P/w4a16_toolcal" --device "Samsung Galaxy S25 (Family)" --output-dir "$P/aihub" ) &
  export_pid=$!
else
  echo "$(date -u +%FT%TZ) AI Hub is not logged in on this machine; the export is skipped"; export_pid=""
fi
step recal_json "$R/${name}_w4a16_toolcal_json_sub200.log" $QAI $S --forced-json --checkpoint "$P/w4a16_toolcal" --name "${name}_w4a16_toolcal_json_sub200" --subset 200
[ -n "$export_pid" ] && wait "$export_pid"
echo "$(date -u +%FT%TZ) study finished for $name"
