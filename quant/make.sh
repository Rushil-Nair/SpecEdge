#!/usr/bin/env bash
# Part 3: make the GGUF files of one model at every quantization level. Runs on the rented machine.
#
#     bash quant/make.sh <merged model dir> <name>        e.g. /workspace/runs/lora_r16_full/merged mine
#
# Writes /workspace/models/gguf/<name>-{F16,Q8_0,Q5_K_M,Q4_K_M,Q4_0}.gguf and
# results/part3/<name>_files.json with each file's size and SHA-256 and the llama.cpp commit.
# F16 is the control: it is converted from the model, the others are quantized from it.
set -euo pipefail
model_dir="$1"; name="$2"
W=/workspace; G=$W/models/gguf; R=$W/specedge/results/part3; mkdir -p "$G" "$R"
if [ ! -f "$G/$name-F16.gguf" ]; then
  $W/venv_convert/bin/python $W/llama.cpp/convert_hf_to_gguf.py "$model_dir" --outtype f16 --outfile "$G/$name-F16.gguf.part"
  mv "$G/$name-F16.gguf.part" "$G/$name-F16.gguf"
fi
for level in Q8_0 Q5_K_M Q4_K_M Q4_0; do
  # written under a temporary name first: another job watches this folder for new files
  [ -f "$G/$name-$level.gguf" ] || { $W/llama.cpp/build/bin/llama-quantize "$G/$name-F16.gguf" "$G/$name-$level.gguf.part" "$level" && mv "$G/$name-$level.gguf.part" "$G/$name-$level.gguf"; }
done
{
  echo "{\"model\": \"$name\", \"source\": \"$model_dir\", \"llama_cpp_commit\": \"$(cat $W/llama_commit.txt)\", \"files\": {"
  first=1
  for level in F16 Q8_0 Q5_K_M Q4_K_M Q4_0; do
    f="$G/$name-$level.gguf"
    [ $first = 1 ] || echo ","
    first=0
    printf '  "%s": {"file": "%s", "bytes": %s, "sha256": "%s"}' "$level" "$(basename "$f")" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
  echo; echo "}}"
} > "$R/${name}_files.json"
cat "$R/${name}_files.json"
