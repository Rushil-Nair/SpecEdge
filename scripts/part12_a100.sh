#!/usr/bin/env bash
# Part 12 on the A100: my fine-tuned model through Qualcomm's w4a16 quantizer, scored, and
# sent to AI Hub for the hosted Galaxy S25. Runs ON the A100, in tmux:
#
#     tmux new-session -d -s part12 "bash /workspace/specedge/scripts/part12_a100.sh <name> <adapter dir>"
#
# <adapter dir> holds the LoRA add-on copied from the training machine. It is merged here
# against the base weights already on this machine, with the same package versions as on the
# training machine. Each step leaves a marker in /workspace/part12/<name>/done, so the script
# can be started again after a failure. The AI Hub step uses the owner's login on this
# machine himself; this script never reads or prints a token.
set -uo pipefail
name="$1"; adapter="$2"
W=/workspace; P=$W/part12/$name; R=$W/specedge/results/part12; mkdir -p "$P/done" "$R" $W/tmp
BASE=$W/models/llama/Llama-3.2-3B-Instruct
QAI=$W/venv_qai/bin/python
export TMPDIR=$W/tmp HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
step() {  # step <marker> <log> <command...>
  local marker="$1" log="$2"; shift 2
  [ -f "$P/done/$marker" ] && { echo "$(date -u +%FT%TZ) skip $marker"; return 0; }
  echo "$(date -u +%FT%TZ) start $marker"
  if "$@" > "$log" 2>&1; then date -u +%FT%TZ > "$P/done/$marker"; echo "$(date -u +%FT%TZ) done $marker"
  else echo "$(date -u +%FT%TZ) FAILED $marker (see $log)"; exit 1; fi
}
step merge "$R/${name}_merge.log" $W/venv_merge/bin/python $W/specedge/training/merge.py --base $BASE --adapter "$adapter" --out "$P/merged"
step quantize "$R/${name}_quantize.log" /usr/bin/time -v $QAI -m qai_hub_models.models.llama_v3_2_3b_instruct.quantize \
     --precision w4a16 --checkpoint "$P/merged" --output-dir "$P/w4a16"
step score "$R/${name}_w4a16_sub200.log" $QAI $W/specedge/npu/score_quantized.py --checkpoint "$P/w4a16" --name "${name}_w4a16_sub200" --subset 200
# Export, compile, link and profile every part on the hosted Galaxy S25 (Qualcomm's own pipeline).
[ -f ~/.qai_hub/client.ini ] || { echo "AI Hub is not logged in on this machine; stopping before the export"; exit 2; }
step aihub "$R/${name}_aihub_export.log" env -u HF_HUB_OFFLINE $QAI -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
     --checkpoint "$P/w4a16" --device "Samsung Galaxy S25 (Family)" --output-dir "$P/aihub"
echo "$(date -u +%FT%TZ) part12 finished for $name"
