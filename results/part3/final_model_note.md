**For the final model (`mine_recovery`)**, all 591 under the token-limit rule: F16 65.7%,
Q8_0 65.7%, Q5_K_M 66.5%, Q4_K_M 65.0%, Q4_0 65.5%; forced JSON costs two tasks at each level
(two or three of the 17 decimal tasks). Same picture: no task type breaks down to 4-bit, and
gains and losses against F16 are of the same size at every level (at Q4_K_M 5 gained, 9
lost). No reply ran to the token limit at any level. GPU speed on an RTX 5090: 216.8
tokens/s at F16, 337.8 at Q8_0, 412.6 at Q5_K_M, 438.8 at Q4_K_M (2.02x), 477.1 at Q4_0;
GPU memory 7.55 GB down to 3.30 GB. These rows were run on the second 5090, after its own
plain-speed check (Qwen3-8B: 239.6 / 242.7 / 240.6 tokens/s, all 180 outputs identical to
round 1). CPU speed on the first 5090's Ryzen 7 7800X3D, 7 threads, 15 tasks: 7.7 tokens/s at
F16, 13.8 at Q8_0, 19.6 at Q5_K_M, 21.8 at Q4_K_M (2.84x), 23.1 at Q4_0. Peak memory 8.89 GB
at F16, 5.94 GB at Q8_0, 4.88 GB at Q5_K_M, then higher again at Q4_K_M (5.83 GB) and Q4_0
(6.00 GB): llama.cpp's CPU backend copies 4-bit tensors into a rearranged layout (a repack
buffer of 1,299 MiB and 1,512 MiB) while the file stays mapped, so on the CPU the two 4-bit
levels hold those weights twice (`results/part3/cpu_repack_check/`). The GGUF files used there were rebuilt from the add-on and have
the same checksums as the files scored for accuracy.
