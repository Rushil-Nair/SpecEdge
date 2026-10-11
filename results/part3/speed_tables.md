**`mine_recovery`, GPU (NVIDIA GeForce RTX 5090, 8 CPUs visible), 60 tasks, three passes, median (lowest-highest)**

| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | GPU memory in use | Replies same as the GPU accuracy run |
|---|---|---|---|---|---|---|---|---|
| F16 | 6.43 GB | off | 216.8 (216.7-216.9) | 1.00x | 26374 (26198-26407) | 33 (33-33) | 7.55 GB | 60 of 60 |
| F16 | 6.43 GB | on | 215.5 (215.4-215.6) | 0.99x | 26440 (26361-26443) | 33 (33-33) | 7.55 GB | 60 of 60 |
| Q8_0 | 3.42 GB | off | 337.8 (337.6-337.9) | 1.56x | 25336 (25303-25342) | 33 (33-33) | 4.71 GB | 60 of 60 |
| Q8_0 | 3.42 GB | on | 334.4 (334.4-334.8) | 1.54x | 25323 (25309-25325) | 33 (33-33) | 4.71 GB | 60 of 60 |
| Q5_K_M | 2.32 GB | off | 412.6 (412.3-413.1) | 1.90x | 23759 (23738-23782) | 35 (35-35) | 3.68 GB | 60 of 60 |
| Q5_K_M | 2.32 GB | on | 405.9 (405.7-406.0) | 1.87x | 23740 (23731-23761) | 35 (35-35) | 3.68 GB | 60 of 60 |
| Q4_K_M | 2.02 GB | off | 438.8 (438.5-439.6) | 2.02x | 24268 (24228-24272) | 34 (34-35) | 3.40 GB | 60 of 60 |
| Q4_K_M | 2.02 GB | on | 433.9 (433.8-434.0) | 2.00x | 24284 (24257-24290) | 34 (34-34) | 3.40 GB | 60 of 60 |
| Q4_0 | 1.92 GB | off | 477.1 (476.8-477.3) | 2.20x | 26251 (26122-26252) | 32 (32-32) | 3.30 GB | 60 of 60 |
| Q4_0 | 1.92 GB | on | 469.3 (469.1-469.8) | 2.16x | 26250 (26232-26278) | 32 (32-32) | 3.30 GB | 60 of 60 |

"First token" is prompt time plus one token, calculated, not measured. GPU memory is read once at the end of the run, not a peak.

**`mine_recovery`, CPU (7 threads, GPU hidden), 15 tasks, three passes, median (lowest-highest)**

| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | Peak server memory | Replies same as the GPU accuracy run |
|---|---|---|---|---|---|---|---|---|
| F16 | 6.43 GB | off | 7.7 (7.7-7.7) | 1.00x | 137 (137-137) | 5839 (5836-5846) | 8.89 GB | 15 of 15 |
| F16 | 6.43 GB | on | 7.7 (7.6-7.7) | 1.00x | 137 (137-137) | 5844 (5840-5869) | 8.90 GB | 15 of 15 |
| Q8_0 | 3.42 GB | off | 13.8 (13.8-13.8) | 1.80x | 157 (157-157) | 5014 (5005-5018) | 5.94 GB | 15 of 15 |
| Q8_0 | 3.42 GB | on | 13.8 (13.8-13.8) | 1.80x | 157 (157-157) | 5030 (4988-5047) | 5.96 GB | 15 of 15 |
| Q5_K_M | 2.32 GB | off | 19.6 (19.6-19.6) | 2.55x | 303 (303-304) | 2549 (2547-2551) | 4.88 GB | 14 of 15 |
| Q5_K_M | 2.32 GB | on | 19.4 (19.4-19.5) | 2.54x | 304 (303-304) | 2549 (2546-2553) | 4.90 GB | 14 of 15 |
| Q4_K_M | 2.02 GB | off | 21.8 (21.8-21.8) | 2.84x | 197 (197-197) | 3893 (3888-3895) | 5.83 GB | 13 of 15 |
| Q4_K_M | 2.02 GB | on | 21.8 (21.8-21.8) | 2.84x | 198 (198-198) | 3883 (3881-3890) | 5.91 GB | 13 of 15 |
| Q4_0 | 1.92 GB | off | 23.1 (23.1-23.1) | 3.01x | 193 (193-193) | 4018 (4012-4021) | 6.00 GB | 12 of 15 |
| Q4_0 | 1.92 GB | on | 23.1 (23.1-23.1) | 3.01x | 193 (193-193) | 4016 (4014-4018) | 6.01 GB | 12 of 15 |

Peak memory is higher at Q4_K_M and Q4_0 than at Q5_K_M because llama.cpp's CPU backend copies 4-bit tensors into a rearranged layout (the CPU_REPACK buffer: 1,299 MiB at Q4_K_M, 1,512 MiB at Q4_0) while the file stays mapped, so those weights are held twice; the other levels are used straight from the mapped file (`results/part3/cpu_repack_check/`). "First token" is prompt time plus one token, calculated, not measured.

**`mine_multistep`, GPU (NVIDIA GeForce RTX 5090, 16 CPUs visible), 60 tasks, three passes, median (lowest-highest)**

| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | GPU memory in use | Replies same as the GPU accuracy run |
|---|---|---|---|---|---|---|---|---|
| F16 | 6.43 GB | off | 217.7 (217.7-217.9) | 1.00x | 26662 (26334-26669) | 33 (33-34) | 7.55 GB | 60 of 60 |
| F16 | 6.43 GB | on | 216.7 (216.6-216.7) | 1.00x | 26584 (26580-26593) | 33 (33-33) | 7.55 GB | 60 of 60 |
| Q8_0 | 3.42 GB | off | 338.2 (338.1-338.7) | 1.55x | 25446 (25399-25446) | 33 (33-33) | 4.71 GB | 60 of 60 |
| Q8_0 | 3.42 GB | on | 336.4 (336.1-336.5) | 1.55x | 25417 (25414-25436) | 33 (33-33) | 4.71 GB | 60 of 60 |
| Q5_K_M | 2.32 GB | off | 414.0 (413.9-414.1) | 1.90x | 23850 (23827-23852) | 35 (35-35) | 3.68 GB | 60 of 60 |
| Q5_K_M | 2.32 GB | on | 409.9 (409.7-410.2) | 1.88x | 23842 (23819-23860) | 35 (35-35) | 3.68 GB | 60 of 60 |
| Q4_K_M | 2.02 GB | off | 442.2 (442.0-442.9) | 2.03x | 24403 (24364-24411) | 34 (34-34) | 3.40 GB | 60 of 60 |
| Q4_K_M | 2.02 GB | on | 438.2 (438.0-438.4) | 2.01x | 24423 (24399-24449) | 34 (34-34) | 3.40 GB | 60 of 60 |
| Q4_0 | 1.92 GB | off | 482.3 (482.0-482.5) | 2.22x | 26334 (26309-26363) | 32 (32-32) | 3.30 GB | 60 of 60 |
| Q4_0 | 1.92 GB | on | 476.3 (476.0-476.4) | 2.19x | 26323 (26317-26338) | 32 (32-32) | 3.30 GB | 60 of 60 |

"First token" is prompt time plus one token, calculated, not measured. GPU memory is read once at the end of the run, not a peak.

**`mine_multistep`, CPU (7 threads, GPU hidden), 15 tasks, three passes, median (lowest-highest)**

| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | Peak server memory | Replies same as the GPU accuracy run |
|---|---|---|---|---|---|---|---|---|
| F16 | 6.43 GB | off | 7.7 (7.7-7.7) | 1.00x | 136 (136-136) | 5852 (5844-5872) | 8.89 GB | 14 of 15 |
| F16 | 6.43 GB | on | 7.7 (7.6-7.7) | 1.00x | 137 (137-137) | 5844 (5840-5869) | 8.89 GB | 14 of 15 |
| Q8_0 | 3.42 GB | off | 13.8 (13.8-13.8) | 1.80x | 157 (157-157) | 5021 (5010-5039) | 5.94 GB | 15 of 15 |
| Q8_0 | 3.42 GB | on | 13.8 (13.8-13.8) | 1.80x | 157 (157-157) | 5011 (4993-5049) | 5.94 GB | 15 of 15 |
| Q5_K_M | 2.32 GB | off | 19.5 (19.5-19.5) | 2.54x | 304 (303-304) | 2554 (2550-2565) | 4.87 GB | 14 of 15 |
| Q5_K_M | 2.32 GB | on | 19.5 (19.5-19.5) | 2.54x | 303 (303-304) | 2552 (2549-2562) | 4.89 GB | 14 of 15 |
| Q4_K_M | 2.02 GB | off | 21.9 (21.9-21.9) | 2.85x | 197 (197-197) | 3892 (3885-3897) | 5.89 GB | 12 of 15 |
| Q4_K_M | 2.02 GB | on | 21.9 (21.9-21.9) | 2.86x | 195 (195-195) | 3937 (3935-3947) | 5.90 GB | 13 of 15 |
| Q4_0 | 1.92 GB | off | 23.3 (23.2-23.3) | 3.04x | 193 (193-193) | 4023 (4017-4027) | 6.00 GB | 13 of 15 |
| Q4_0 | 1.92 GB | on | 23.3 (23.3-23.3) | 3.03x | 193 (193-193) | 4020 (4014-4024) | 5.94 GB | 13 of 15 |

Peak memory is higher at Q4_K_M and Q4_0 than at Q5_K_M because llama.cpp's CPU backend copies 4-bit tensors into a rearranged layout (the CPU_REPACK buffer: 1,299 MiB at Q4_K_M, 1,512 MiB at Q4_0) while the file stays mapped, so those weights are held twice; the other levels are used straight from the mapped file (`results/part3/cpu_repack_check/`). "First token" is prompt time plus one token, calculated, not measured.

**`mine`, GPU (NVIDIA GeForce RTX 5090, 16 CPUs visible), 60 tasks, three passes, median (lowest-highest)**

| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | GPU memory in use | Replies same as the GPU accuracy run |
|---|---|---|---|---|---|---|---|---|
| F16 | 6.43 GB | off | 217.3 (216.7-217.5) | 1.00x | 26604 (24122-26630) | 33 (33-35) | 7.55 GB | 60 of 60 |
| F16 | 6.43 GB | on | 216.2 (216.1-216.2) | 0.99x | 26528 (26456-26591) | 33 (33-34) | 7.55 GB | 60 of 60 |
| Q8_0 | 3.42 GB | off | 337.9 (337.9-338.4) | 1.56x | 25389 (25290-25392) | 33 (33-33) | 4.71 GB | 60 of 60 |
| Q8_0 | 3.42 GB | on | 334.5 (334.3-334.8) | 1.54x | 25427 (25419-25436) | 33 (33-33) | 4.71 GB | 60 of 60 |
| Q5_K_M | 2.32 GB | off | 413.2 (413.0-413.4) | 1.90x | 23870 (23848-23872) | 35 (35-35) | 3.68 GB | 60 of 60 |
| Q5_K_M | 2.32 GB | on | 409.7 (409.5-409.9) | 1.89x | 23838 (23821-23853) | 35 (35-35) | 3.68 GB | 60 of 60 |
| Q4_K_M | 2.02 GB | off | 442.4 (442.1-442.4) | 2.04x | 24355 (24340-24378) | 34 (34-34) | 3.40 GB | 60 of 60 |
| Q4_K_M | 2.02 GB | on | 438.4 (438.2-438.4) | 2.02x | 24396 (24378-24397) | 34 (34-34) | 3.40 GB | 60 of 60 |
| Q4_0 | 1.92 GB | off | 481.5 (481.3-482.0) | 2.22x | 26312 (26300-26313) | 32 (32-32) | 3.30 GB | 60 of 60 |
| Q4_0 | 1.92 GB | on | 471.8 (471.4-472.2) | 2.17x | 26347 (26340-26355) | 32 (32-32) | 3.30 GB | 60 of 60 |

"First token" is prompt time plus one token, calculated, not measured. GPU memory is read once at the end of the run, not a peak.

**`mine`, CPU (7 threads, GPU hidden), 15 tasks, three passes, median (lowest-highest)**

| Level | File size | Forced JSON | Generation, tokens/s | Against F16 | Prompt, tokens/s | First token, ms | Peak server memory | Replies same as the GPU accuracy run |
|---|---|---|---|---|---|---|---|---|
| F16 | 6.43 GB | off | 7.6 (7.6-7.6) | 1.00x | 133 (133-133) | 5939 (5933-5944) | 8.82 GB | 15 of 15 |
| F16 | 6.43 GB | on | 7.6 (7.6-7.6) | 1.00x | 133 (133-133) | 5937 (5936-5942) | 8.89 GB | 15 of 15 |
| Q8_0 | 3.42 GB | off | 13.8 (13.8-13.8) | 1.80x | 157 (157-157) | 5012 (5010-5017) | 5.94 GB | 15 of 15 |
| Q8_0 | 3.42 GB | on | 13.8 (13.8-13.8) | 1.80x | 157 (157-157) | 5016 (4999-5036) | 5.94 GB | 15 of 15 |
| Q5_K_M | 2.32 GB | off | 19.4 (19.4-19.4) | 2.54x | 304 (304-304) | 2547 (2545-2561) | 4.87 GB | 13 of 15 |
| Q5_K_M | 2.32 GB | on | 19.4 (19.4-19.4) | 2.54x | 303 (303-303) | 2555 (2551-2555) | 4.88 GB | 13 of 15 |
| Q4_K_M | 2.02 GB | off | 21.8 (21.8-21.9) | 2.86x | 198 (198-198) | 3882 (3879-3887) | 5.90 GB | 14 of 15 |
| Q4_K_M | 2.02 GB | on | 21.8 (21.8-21.8) | 2.85x | 206 (206-206) | 3798 (3790-3801) | 5.92 GB | 14 of 15 |
| Q4_0 | 1.92 GB | off | 23.1 (23.1-23.1) | 3.02x | 188 (188-188) | 4090 (4084-4093) | 6.02 GB | 13 of 15 |
| Q4_0 | 1.92 GB | on | 23.1 (23.1-23.1) | 3.02x | 187 (187-188) | 4100 (4092-4105) | 6.00 GB | 13 of 15 |

Peak memory is higher at Q4_K_M and Q4_0 than at Q5_K_M because llama.cpp's CPU backend copies 4-bit tensors into a rearranged layout (the CPU_REPACK buffer: 1,299 MiB at Q4_K_M, 1,512 MiB at Q4_0) while the file stays mapped, so those weights are held twice; the other levels are used straight from the mapped file (`results/part3/cpu_repack_check/`). "First token" is prompt time plus one token, calculated, not measured.

