#!/usr/bin/env bash
# Part 12, ON the A100, beside the study script: decide whether calibration runs.
#
#     tmux new-session -d -s gate "bash /workspace/specedge/scripts/part12_gate_a100.sh <name>"
#
# Waits for the study's `score` step, then compares the default w4a16 score with the
# full-precision control (npu/calibration_gate.py). Under 3 points of loss: writes the
# markers of the three calibration steps, so the study script skips them and ends after the
# forced-JSON row and the AI Hub export. 3 points or more: does nothing, the study goes on.
# It only reads two result files; it starts nothing on the GPU.
set -uo pipefail
name="$1"; W=/workspace; P=$W/part12/$name; R=$W/specedge/results/part12; L=$R/${name}_calibration_gate.log
until [ -f "$P/done/score" ]; do
  grep -q "FAILED" "$R/${name}_study.log" 2>/dev/null && { echo "$(date -u +%FT%TZ) the study failed before scoring; no decision" >> "$L"; exit 1; }
  sleep 30
done
echo "$(date -u +%FT%TZ) score step finished" >> "$L"
python3 $W/specedge/npu/calibration_gate.py "$R/${name}_float_sub200.json" "$R/${name}_w4a16_sub200.json" >> "$L" 2>&1
status=$?
if [ $status -eq 0 ]; then
  for marker in recal recal_score recal_json; do echo "skipped by the calibration gate $(date -u +%FT%TZ)" > "$P/done/$marker"; done
  echo "$(date -u +%FT%TZ) calibration steps marked as skipped" >> "$L"
elif [ $status -eq 3 ]; then
  echo "$(date -u +%FT%TZ) calibration runs as planned" >> "$L"
else
  echo "$(date -u +%FT%TZ) the gate could not decide (status $status); calibration runs as planned" >> "$L"
fi
