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

## 4. Register + shared-memory residency (addendum, 2026-10-17, before any run of it)

The user chose to go beyond the shared-memory limit with a hybrid residency (`build_grid_hybrid`, `KernelConfig.reg_k`):
CTA s keeps rows [0, reg_k) of its R slice [4S, D] in a register fragment and feeds it to WGMMA as the A operand
from registers; rows [reg_k, D) stay in shared memory; h^T is staged in two shared tiles of the same split; each
step runs two WGMMAs into one accumulator. Exchange (double-buffered global buffer, one grid barrier per step),
epilogue and outputs are unchanged from the grid kernel. One warpgroup, S = 16, BM = 16; reg_k is the smallest
multiple of 64 for which shared memory fits.

| D | CTAs | reg_k | registers per thread holding R | shared memory | ptxas (FP16 and BF16, both cells) |
| --- | --- | --- | --- | --- | --- |
| 1536 | 96 | 192 | 48 | 220 KB | 255 registers, 0 spill |
| 1792 | 112 | 512 | 128 | 220 KB | 255 registers, 0 spill |
| 1920 | 120 | 640 | 160 | 224 KB | 255 registers, 0 spill |
| 1984 | 124 | 704 | 176 | 226 KB | 255 registers, 0 spill |

D = 2048 needs more registers than one warpgroup has (a K split over two warpgroups with a shared-memory reduction
would reach it; not built). The hybrid path is forward only: the backward is the Triton kernel, which fits up to
D = 1280.

Runs (separate batch "maxd hybrid"):
* **Qualification** (W1-hybrid): forward at D ∈ {1536, 1792, 1920, 1984}, plus D = 1280 with reg_k = 256 as a
  control against the shared-memory-only grid kernel at the same D (both within the FP64 budget; bitwise equality is
  not expected, the K summation is split differently); LSTM/sLSTM, FP16/BF16, B = 16, T = 24; FP64 forward budget of
  `tools/bw_qualify.py`.
* **Paired timing** (W3/W4-hybrid): forward at D ∈ {1536, 1792, 1920, 1984}, B ∈ {16, 64}, LSTM/sLSTM, FP16/BF16,
  T = 256, FlashRNN2 hybrid against FlashRNN1 `cuda_fused` and `cuda`, same protocol and rule as section 3 (a
  FlashRNN1 build or run failure at a D is reported as such).

## 5. Two warpgroups for D = 2048 (addendum, 2026-10-17, before any run of it)

`build_grid_hybrid2` (`KernelConfig(family="tilelang", sync="grid", slice_width=16, warps=8, reg_k=896)`): 256
threads per CTA; warpgroup w owns K rows [wD/2, (w+1)D/2) of the CTA's R slice, the first reg_k/2 = 448 in its
register fragment and 576 in shared memory, and runs two WGMMAs (A from registers, A from shared memory) into its own
accumulator; each warpgroup writes its partial gates to its own shared stage, and after a block barrier all 256
threads sum the two stages in the epilogue. Exchange and epilogue as in the grid kernel. D = 2048: 128 CTAs, 216 KB
shared memory; ptxas (FP16 and BF16, LSTM and sLSTM): 242 registers, 0 spill. Login-node inspection of the
generated code: both branches issue `wgmma_rs`, and the second warpgroup's addressing is offset by its 128 threads.
Larger D: D = 2176 would need 136 CTAs (more than 132 SMs) at S = 16.

Runs (batch "maxd 2048"): qualification at D = 2048 (forward, FP64 budget; LSTM/sLSTM, FP16/BF16, B = 16, T = 24)
and paired forward timing at D = 2048, B ∈ {16, 64}, LSTM/sLSTM, FP16/BF16 against FlashRNN1 `cuda_fused` and
`cuda`, with the protocol and rule of section 3.

## 6. Backward to D = 2048 (addendum, 2026-10-17, before any run of it)

The Triton backward keeps R[:, :, p] (4 gates × D × S) as loop-invariant shared memory and fits only up to D = 1280.
Its dh reduction has a 16 × 16 output per CTA (batch tile × S columns), below WGMMA's M = 64, so the large-D
backward is a CUDA kernel with mma.sync m16n8k16 (`flashrnn/flashrnn2/hybrid_bw.cu`, loaded by `hybrid_bw.py`
through ctypes, CUDA 12.8, static cudart):

* 256 threads per CTA; the K = 4D reduction is split evenly over the 8 warps (D/2 rows each); warp w keeps the first
  KRW rows of its range of R[:, :, p] as mma B fragments in registers and the rest in shared memory in fragment
  order (32 bytes per row); the rounded dG_{t+1} operand is read from the exchange buffer straight into A fragments
  with L2-only loads; the 8 partial sums meet in shared memory.
* Pointwise backward, stores, exchange and the initial-state gradient follow the Triton kernel's plain persistent
  variant line by line (FlashRNN1 gradient convention); one cooperative grid barrier per step; the preactivation
  recompute and the dR / dbias GEMMs are the Triton backward's.
* Register rows per warp (fewest that let shared memory fit): D = 1024, 1280, 1536: 0; 1792: 32; 2048: 160.
  ptxas (FP16/BF16, LSTM/sLSTM): 64 registers up to D = 1536, 96 at 1792, 181-185 at 2048, 0 spill.
* The training path uses it above D = 1280 (or with `backward_options={"backend": "hybrid"}`); the forward above
  D = 1280 is the TileLang path of sections 4-5 (`default_config` now selects it).

Runs (batch "bw 2048"):
* **Qualification** (W1-train): training step (forward + backward) within the FP64 gradient budget of
  `tools/bw_qualify.py` and the FP64 forward budget: D = 1024 with the hybrid backward forced (and the Triton
  backward at the same D as a control), and D = 1536, 1792, 2048 (library choice); LSTM/sLSTM, FP16/BF16, B = 16,
  T = 24.
* **Paired timing** (W3/W4-train): training step at D ∈ {1536, 1792, 2048}, B ∈ {16, 64}, LSTM/sLSTM, FP16/BF16,
  T = 256, against FlashRNN1 `cuda_fused` and `cuda`, protocol and rule of section 3 (a FlashRNN1 build or run
  failure is reported as such).
* Not covered by this batch: race verification of the CUDA kernel's exchange (delay injection and broken-variant
  controls, as done for the Triton and TileLang kernels).
