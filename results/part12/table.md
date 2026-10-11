| Row | Engine | All 200, old rule | All 200, token-limit rule | Call (121) | Should ask: made no call (45) | No tool fits (34) | Against its control: lost / gained |
|---|---|---|---|---|---|---|---|
| Stock 3B, full precision | llama.cpp F16 | 46.0% (92) | 45.0% (90) | 69.4% (84) | 0.0% (0) | 17.6% (6) |  |
| Stock 3B, full precision (control) | Qualcomm's engine | 42.0% (84) | 42.0% (84) | 69.4% (84) | 0.0% (0) | 0.0% (0) |  |
| Stock 3B, w4a16 | Qualcomm's simulation | 34.0% (68) | 34.0% (68) | 56.2% (68) | 0.0% (0) | 0.0% (0) | 21 / 5 |
| Second fine-tune (`mine_multistep`), full precision | llama.cpp F16 | 66.5% (133) | 66.5% (133) | 83.5% (101) | 24.4% (11) | 61.8% (21) |  |
| Second fine-tune, full precision (control) | Qualcomm's engine | 67.0% (134) | 67.0% (134) | 84.3% (102) | 24.4% (11) | 61.8% (21) |  |
| Second fine-tune, w4a16 | Qualcomm's simulation | 63.5% (127) | 63.5% (127) | 76.0% (92) | 28.9% (13) | 64.7% (22) | 14 / 7 |
| Second fine-tune, w4a16, forced JSON | Qualcomm's simulation | 64.5% (129) | 64.5% (129) | 77.7% (94) | 28.9% (13) | 64.7% (22) | 12 / 7 |
| Second fine-tune, w4a16, tool-call calibration | Qualcomm's simulation | 65.5% (131) | 65.5% (131) | 80.2% (97) | 26.7% (12) | 64.7% (22) | 10 / 7 |
| Second fine-tune, w4a16, tool-call calibration, forced JSON | Qualcomm's simulation | 66.0% (132) | 66.0% (132) | 81.0% (98) | 26.7% (12) | 64.7% (22) | 9 / 7 |
| **Final model (`mine_recovery`)**, full precision | llama.cpp F16 | 65.0% (130) | 65.0% (130) | 82.6% (100) | 24.4% (11) | 55.9% (19) |  |
| **Final model**, full precision (control) | Qualcomm's engine | 65.0% (130) | 65.0% (130) | 82.6% (100) | 24.4% (11) | 55.9% (19) |  |
| **Final model**, w4a16, tool-call calibration, forced JSON | Qualcomm's simulation | 62.5% (125) | 62.5% (125) | 76.9% (93) | 28.9% (13) | 55.9% (19) | 10 / 5 |
