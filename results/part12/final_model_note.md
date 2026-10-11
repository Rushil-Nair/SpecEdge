Control 65.0% (130 of 200, the same as llama.cpp on these tasks); w4a16 with tool-call
  calibration and forced JSON **62.5% (125): 2.5 points lower**, 10 tasks lost and 5 gained,
  on call tasks (100 to 93 of 121). No reply at the token limit. The calibration text is the
  first 2,000 examples of its own training file; the quantizer's log shows it and the
  encodings file differs from both of the second fine-tune's. Quantizer: 1 h 15 min 8 s, peak memory
  221.1 GB. On the hosted Galaxy S25 every operation runs on the NPU; AI Hub's estimates sum
  to about 43.8 ms per token step (18.9 ms and 24.9 ms for the two transformer parts) and
  73.7 ms for a 128-token prompt step. What the two fixes recover for this model is not
  measured, because its default-calibration row was not run; the remaining 2.5 points
  against the second fine-tune's 1.0 are five tasks apart on 200.
