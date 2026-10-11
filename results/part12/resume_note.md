The fine-tuned model, quantized with tool-call calibration text and decoded under a JSON
  grammar, loses 2.5 points (65.0% to 62.5%), and compiles and runs entirely on the same
  NPU (an estimated 22 tokens/s: about 44 ms per token step, the sum of three AI Hub profile
  estimates, not a measured end-to-end run). For the
  previous fine-tune the full comparison was run: 3.5 points lost with Qualcomm's default
  calibration, 1.0 with calibration text and constrained decoding.
