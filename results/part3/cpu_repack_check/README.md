# Why Q4_K_M and Q4_0 use more CPU memory than Q5_K_M

Each level of `mine_recovery` loaded once on the first 5090 with the GPU hidden, 7 threads,
`llama-server -lv 5`, on 2026-10-08 at about 21:06 UTC, machine otherwise idle. Memory read
from `/proc/<pid>/status` right after "model loaded" (no request sent). The timing runs' own
server logs were written at the default verbosity and do not show how the weights are loaded.

| Level | Tensors repacked | CPU_REPACK buffer | Anonymous memory | File-mapped memory |
|---|---|---|---|---|
| F16 | none | none | 1.09 GB | 6.36 GB |
| Q8_0 | none | none | 1.09 GB | 3.42 GB |
| Q5_K_M | none | none | 1.09 GB | 2.34 GB |
| Q4_K_M | 168 | 1,299 MiB | 2.42 GB | 2.03 GB |
| Q4_0 | 196 | 1,512 MiB | 2.64 GB | 1.93 GB |

llama.cpp's CPU backend (build flag `REPACK = 1`) copies 4-bit tensors into a rearranged
layout for its AVX-512 kernels, in ordinary memory, while the model file stays mapped. Those
weights are then held twice. The other levels' tensor types cannot use that buffer (the log
says so) and are used straight from the mapped file. The extra anonymous memory at Q4_K_M
and Q4_0 (1.33 GB and 1.55 GB over the other levels) is the size of the repack buffer.
