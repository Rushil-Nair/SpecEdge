#!/usr/bin/env bash
# Part 12, second half, ON the A100 after scripts/part12_a100.sh has finished for <name>:
#
#     tmux new-session -d -s followup "bash /workspace/specedge/scripts/part12_followup_a100.sh <name> <calibration .jsonl>"
#
#   float      control: my merged model at full precision through the same Qualcomm engine, same 200 tasks
#   json       my w4a16 model again with forced JSON during generation
#   recal      recovery attempt: quantize again, same recipe, calibration text = tool-call training examples
#   recal_*    score that checkpoint without and with forced JSON
# Steps leave markers in /workspace/part12/<name>/done and are skipped when run again.
set -uo pipefail
name="$1"; calibration="$2"
W=/workspace; P=$W/part12/$name; R=$W/specedge/results/part12; QAI=$W/venv_qai/bin/python
export TMPDIR=$W/tmp HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
step() {
  local marker="$1" log="$2"; shift 2
  [ -f "$P/done/$marker" ] && { echo "$(date -u +%FT%TZ) skip $marker"; return 0; }
  echo "$(date -u +%FT%TZ) start $marker"
  if "$@" > "$log" 2>&1; then date -u +%FT%TZ > "$P/done/$marker"; echo "$(date -u +%FT%TZ) done $marker"
  else echo "$(date -u +%FT%TZ) FAILED $marker (see $log)"; exit 1; fi
}
S=$W/specedge/npu/score_quantized.py
step float "$R/${name}_float_sub200.log" $QAI $S --float --checkpoint "$P/merged" --name "${name}_float_sub200" --subset 200
step json "$R/${name}_w4a16_json_sub200.log" $QAI $S --forced-json --checkpoint "$P/w4a16" --name "${name}_w4a16_json_sub200" --subset 200
step recal "$R/${name}_quantize_toolcal.log" /usr/bin/time -v $QAI $W/specedge/npu/quantize_with_calibration.py --calibration "$calibration" \
     --precision w4a16 --checkpoint "$P/merged" --output-dir "$P/w4a16_toolcal"
step recal_score "$R/${name}_w4a16_toolcal_sub200.log" $QAI $S --checkpoint "$P/w4a16_toolcal" --name "${name}_w4a16_toolcal_sub200" --subset 200
step recal_json "$R/${name}_w4a16_toolcal_json_sub200.log" $QAI $S --forced-json --checkpoint "$P/w4a16_toolcal" --name "${name}_w4a16_toolcal_json_sub200" --subset 200
echo "$(date -u +%FT%TZ) follow-up finished for $name"
