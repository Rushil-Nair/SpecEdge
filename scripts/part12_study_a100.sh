#!/usr/bin/env bash
# Part 12, the whole study for one model, ON the A100:
#
#     tmux new-session -d -s study "bash /workspace/specedge/scripts/part12_study_a100.sh <name> <adapter dir> <calibration .jsonl>"
#
#   merge        the LoRA add-on into the base weights on this machine
#   float        control: full precision through Qualcomm's own engine, the fixed 200 tasks
#   quantize     Qualcomm's w4a16 recipe, default (Wikitext) calibration
#   score        the w4a16 model on the 200 tasks
#   aihub        export, compile, link and profile every part on the hosted Galaxy S25
#                (started in the background beside the next scoring step: it mostly waits
#                on AI Hub and needs 14 GB of GPU memory beside the scorer's 46 GB)
#   json         the w4a16 model with forced JSON
#   recal        the same recipe with tool-call training examples as calibration text
#   recal_score, recal_json   that checkpoint without and with forced JSON
#
# Nothing else is ever started beside the quantizer: it peaks near 210 GB of the machine's
# 259 GB of memory, and twice on 2026-10-08 a side job broke a long run. Steps leave markers
# in /workspace/part12/<name>/done, so the script can be started again after a failure.
# The AI Hub step uses the owner's login on this machine; no token is read or printed.
set -uo pipefail
name="$1"; adapter="$2"; calibration="$3"
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
step quantize "$R/${name}_quantize.log" /usr/bin/time -v $QAI -m qai_hub_models.models.llama_v3_2_3b_instruct.quantize \
     --precision w4a16 --checkpoint "$P/merged" --output-dir "$P/w4a16" || exit 1
step score "$R/${name}_w4a16_sub200.log" $QAI $S --checkpoint "$P/w4a16" --name "${name}_w4a16_sub200" --subset 200 || exit 1
if [ -f ~/.qai_hub/client.ini ]; then
  ( step aihub "$R/${name}_aihub_export.log" env -u HF_HUB_OFFLINE $QAI -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
        --checkpoint "$P/w4a16" --device "Samsung Galaxy S25 (Family)" --output-dir "$P/aihub" ) &
  export_pid=$!
else
  echo "$(date -u +%FT%TZ) AI Hub is not logged in on this machine; the export is skipped"; export_pid=""
fi
step json "$R/${name}_w4a16_json_sub200.log" $QAI $S --forced-json --checkpoint "$P/w4a16" --name "${name}_w4a16_json_sub200" --subset 200
[ -n "$export_pid" ] && wait "$export_pid"          # the export must be over before the next quantizer run
step recal "$R/${name}_quantize_toolcal.log" /usr/bin/time -v $QAI $W/specedge/npu/quantize_with_calibration.py --calibration "$calibration" \
     --precision w4a16 --checkpoint "$P/merged" --output-dir "$P/w4a16_toolcal" || exit 1
step recal_score "$R/${name}_w4a16_toolcal_sub200.log" $QAI $S --checkpoint "$P/w4a16_toolcal" --name "${name}_w4a16_toolcal_sub200" --subset 200
step recal_json "$R/${name}_w4a16_toolcal_json_sub200.log" $QAI $S --forced-json --checkpoint "$P/w4a16_toolcal" --name "${name}_w4a16_toolcal_json_sub200" --subset 200
echo "$(date -u +%FT%TZ) study finished for $name"
