# FlashRNN2 at large head dimension D: the maximum, and speed against FlashRNN1

Status: pre-registration, written 2026-10-17 before any large-D GPU run. Single GPU (H100), H = 1.

## 1. What limits D (login-node compile records, no GPU)

FlashRNN2 keeps R resident in shared memory, split by output columns: CTA s holds R[:, :, :, p_s] for S columns.
Per CTA that is 4·D·S·2 bytes of R plus the h operand tile (16·D·2 bytes) and small buffers; the H100 opt-in limit
is 232,448 bytes per block. S ≥ 16 (smallest MMA N), and every CTA of the cooperative grid must be resident
(k = D/S ≤ 132).

| path | shared memory per CTA at S = 16 | largest D that fits | other constraint |
| --- | --- | --- | --- |
| Triton R tiling, forward | 160·D (205 KB at D = 1280, 246 KB at D = 1536) | 1280 | K split into two powers of two (1344–1456 pad to 2048 and do not fit) |
| Triton R tiling, backward | 224 KB at D = 1280, 262 KB at D = 1536 | 1280 | same |
| TileLang grid, forward | 160·D + 4 KB (209 KB at D = 1280, 234 KB at D = 1440) | 1424 | D multiple of 16 |

So FlashRNN2's resident-R maximum is D = 1424 for the forward (TileLang grid) and D = 1280 for training (the
backward is the Triton kernel). The library defaults above D = 768 chose S = 32 (about 290–330 KB per CTA at
D = 1024), which cannot launch; they now choose S = 16 above D = 768 (forward `default_config`, backward
`backward_slice_width`). Going further needs part of R outside shared memory (registers, or streamed from L2/HBM
each step), as FlashRNN1's register tiles do; that is not built.

## 2. Runs

* **Qualification** (`tools/heads_qualify.py`, H = 1, pinned configuration, FP64 forward budget, FP64 gradient budget
  for training): TileLang grid S = 16 at D ∈ {1024, 1280, 1424} (training at D ≤ 1280: TileLang forward, Triton
  backward S = 16) and Triton R tiling S = 16, eight warps at D ∈ {1024, 1280} (forward and training); LSTM/sLSTM,
  FP16/BF16. A path is used for timing only if every one of its cases passes.
* **Feasibility** (`tools/heads_bench.py sweep`, medians, report only): FlashRNN1 `cuda_fused`, FlashRNN1 `cuda`
  (alternating: a recurrent GEMM and a pointwise kernel per step) and FlashRNN2 (TileLang grid S = 16) at
  D ∈ {1024, 1280, 1424, 1536, 1792, 2048, 2560, 3072, 4096}, LSTM/sLSTM, FP16, B = 16, T = 256, forward and training
  step. A build or run failure is the result for that D.
* **Paired timing** (3 process starts × 20 ABBA/BAAB blocks, 3 calls per entry, outputs bitwise stable): FlashRNN2
  (TileLang grid S = 16 forward; for training the same forward and the Triton backward S = 16) against FlashRNN1
  `cuda_fused` and against FlashRNN1 `cuda`, forward at D ∈ {1024, 1280, 1424} and training at D ∈ {1024, 1280};
  B ∈ {16, 64}, LSTM/sLSTM, FP16/BF16, T = 256.

## 3. Checks

* **W1**: qualification per path (every case passes).
* **W2** (reported): the largest D at which each implementation built and ran, forward and training.
* **W3**: FlashRNN1 `cuda_fused` / FlashRNN2 per-call time, 95% hierarchical bootstrap interval (10,000 draws over
  starts, then blocks); holds for a shape when the lower bound is ≥ 1.
* **W4**: the same against FlashRNN1 `cuda` (alternating).
* **W5** (reported): above FlashRNN2's maximum, FlashRNN1 `cuda_fused` against `cuda` from the feasibility medians.
