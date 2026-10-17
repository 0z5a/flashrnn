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

## 7. Results of the shared-memory-only batch (job "maxd v1", 2026-10-17)

**W1**: 20/20 qualification cases pass (TileLang grid S = 16 at D = 1024/1280/1424, Triton R tiling S = 16 at D = 1024/1280; forward and training; FP64 budgets).

**W2** (largest D that built and ran; B = 16, T = 256): FlashRNN1 `cuda_fused` D = 2560 (D = 3072 and 4096: "No working solution of constraints"), FlashRNN1 `cuda` up to D = 4096 (all tested), FlashRNN2 in this batch D = 1424 forward and 1280 training, as the compile records predicted (the register-residency batches cover 1536-2048).

**W3 / W4** (ratio = FlashRNN1 time / FlashRNN2 time, 95% interval; holds when the lower bound is >= 1). W3 holds in 32/40 shapes, W4 in 24/40.

| kind | cell | dtype | B | D | vs cuda_fused (W3) | vs cuda (W4) |
| --- | --- | --- | --- | --- | --- | --- |
| forward | lstm | bflo | 16 | 1024 | 2.16 [1.86, 3.06] | 3.47 [3.45, 3.49] |
| forward | lstm | bflo | 16 | 1280 | 2.30 [2.30, 2.31] | 3.33 [3.25, 3.38] |
| forward | lstm | bflo | 16 | 1424 | 8.18 [8.15, 8.21] | 4.11 [4.04, 4.22] |
| forward | lstm | bflo | 64 | 1024 | 4.22 [4.15, 4.41] | 1.54 [1.52, 1.55] |
| forward | lstm | bflo | 64 | 1280 | 2.29 [2.23, 2.47] | 0.76 [0.75, 0.77] (no) |
| forward | lstm | bflo | 64 | 1424 | 3.00 [3.00, 3.01] | 0.53 [0.53, 0.53] (no) |
| forward | lstm | floa | 16 | 1024 | 1.81 [1.81, 1.82] | 3.22 [3.19, 3.26] |
| forward | lstm | floa | 16 | 1280 | 2.21 [2.21, 2.22] | 2.91 [2.90, 2.91] |
| forward | lstm | floa | 16 | 1424 | 5.53 [5.51, 5.55] | 2.90 [2.88, 2.91] |
| forward | lstm | floa | 64 | 1024 | 4.22 [4.07, 4.65] | 1.50 [1.50, 1.51] |
| forward | lstm | floa | 64 | 1280 | 2.35 [2.35, 2.36] | 0.93 [0.93, 0.93] (no) |
| forward | lstm | floa | 64 | 1424 | 1.59 [1.59, 1.59] | 0.58 [0.57, 0.58] (no) |
| forward | slstm | bflo | 16 | 1024 | 2.23 [1.90, 3.21] | 3.59 [3.57, 3.61] |
| forward | slstm | bflo | 16 | 1280 | 3.14 [3.13, 3.15] | 3.46 [3.43, 3.49] |
| forward | slstm | bflo | 16 | 1424 | 6.78 [6.73, 6.83] | 4.31 [4.26, 4.35] |
| forward | slstm | bflo | 64 | 1024 | 2.86 [2.86, 2.87] | 1.53 [1.52, 1.54] |
| forward | slstm | bflo | 64 | 1280 | 3.33 [3.32, 3.33] | 0.80 [0.79, 0.81] (no) |
| forward | slstm | bflo | 64 | 1424 | 1.60 [1.60, 1.61] | 0.54 [0.54, 0.55] (no) |
| forward | slstm | floa | 16 | 1024 | 1.89 [1.85, 1.91] | 3.31 [3.24, 3.38] |
| forward | slstm | floa | 16 | 1280 | 2.30 [2.27, 2.38] | 3.01 [2.98, 3.03] |
| forward | slstm | floa | 16 | 1424 | 5.58 [5.55, 5.60] | 3.04 [3.02, 3.05] |
| forward | slstm | floa | 64 | 1024 | 2.85 [2.84, 2.85] | 1.50 [1.47, 1.52] |
| forward | slstm | floa | 64 | 1280 | 2.48 [2.47, 2.48] | 0.97 [0.97, 0.97] (no) |
| forward | slstm | floa | 64 | 1424 | 1.62 [1.62, 1.62] | 0.59 [0.59, 0.60] (no) |
| train | lstm | bflo | 16 | 1024 | 0.83 [0.83, 0.83] (no) | 1.31 [1.29, 1.33] |
| train | lstm | bflo | 16 | 1280 | 0.86 [0.86, 0.86] (no) | 1.23 [1.21, 1.24] |
| train | lstm | bflo | 64 | 1024 | 1.35 [1.34, 1.36] | 0.75 [0.74, 0.76] (no) |
| train | lstm | bflo | 64 | 1280 | 1.03 [0.99, 1.04] (no) | 0.41 [0.41, 0.42] (no) |
| train | lstm | floa | 16 | 1024 | 0.78 [0.72, 0.80] (no) | 1.31 [1.29, 1.33] |
| train | lstm | floa | 16 | 1280 | 0.83 [0.83, 0.83] (no) | 1.20 [1.19, 1.20] |
| train | lstm | floa | 64 | 1024 | 1.35 [1.20, 1.41] | 0.71 [0.70, 0.72] (no) |
| train | lstm | floa | 64 | 1280 | 1.05 [1.05, 1.05] | 0.43 [0.43, 0.44] (no) |
| train | slstm | bflo | 16 | 1024 | 0.85 [0.85, 0.85] (no) | 1.31 [1.29, 1.33] |
| train | slstm | bflo | 16 | 1280 | 1.00 [1.00, 1.01] | 1.26 [1.23, 1.27] |
| train | slstm | bflo | 64 | 1024 | 1.12 [1.02, 1.16] | 0.75 [0.74, 0.76] (no) |
| train | slstm | bflo | 64 | 1280 | 1.23 [1.23, 1.23] | 0.43 [0.43, 0.44] (no) |
| train | slstm | floa | 16 | 1024 | 0.80 [0.80, 0.81] (no) | 1.31 [1.29, 1.34] |
| train | slstm | floa | 16 | 1280 | 0.83 [0.83, 0.83] (no) | 1.21 [1.21, 1.21] |
| train | slstm | floa | 64 | 1024 | 1.12 [1.11, 1.12] | 0.71 [0.69, 0.72] (no) |
| train | slstm | floa | 64 | 1280 | 1.07 [1.07, 1.07] | 0.45 [0.44, 0.45] (no) |

Reading (post hoc, not a check): the forward at B = 16 is 1.8-8.2x faster than both FlashRNN1 backends. At B = 64 it
stays faster than `cuda_fused` but loses to `cuda` from D = 1280 (0.53-0.97x): a group needs k = D/16 = 64-89 CTAs,
so at most one or two groups are resident and the four 16-row batch tiles run one after another, while the alternating
backend does all 64 rows in one GEMM per step. Training loses to `cuda_fused` at B = 16 (0.78-0.86x, except sLSTM BF16
D = 1280) and to `cuda` at B = 64 (0.41-0.75x), for the same serialization and the Triton backward's cost at large D.


## 8. Results of the register + shared-memory residency batch (job "maxd hybrid", 2026-10-17)

**W1-hybrid**: 20/20 forward cases pass the FP64 budget at D = 1280 (reg_k 256), 1536, 1792, 1920, 1984 (LSTM/sLSTM, FP16/BF16); the shared-memory-only control at D = 1280 passes 4/4.

**W3-hybrid** (against FlashRNN1 `cuda_fused`) holds in 32/32 shapes; **W4-hybrid** (against `cuda`) in 26/32. Forward, T = 256; ratio = FlashRNN1 time / FlashRNN2 time, 95% interval.

| cell | dtype | B | D | vs cuda_fused | vs cuda |
| --- | --- | --- | --- | --- | --- |
| lstm | bflo | 16 | 1536 | 2.61 [2.51, 2.90] | 4.14 [4.09, 4.17] |
| lstm | bflo | 16 | 1792 | 5.71 [5.69, 5.73] | 4.65 [4.57, 4.71] |
| lstm | bflo | 16 | 1920 | 6.53 [6.51, 6.55] | 3.88 [3.83, 3.93] |
| lstm | bflo | 16 | 1984 | 6.73 [6.70, 6.75] | 4.24 [4.21, 4.26] |
| lstm | bflo | 64 | 1536 | 2.92 [2.85, 3.14] | 0.78 [0.77, 0.79] (no) |
| lstm | bflo | 64 | 1792 | 3.31 [3.30, 3.31] | 0.79 [0.79, 0.80] (no) |
| lstm | bflo | 64 | 1920 | 5.38 [5.37, 5.38] | 1.05 [1.04, 1.05] |
| lstm | bflo | 64 | 1984 | 3.74 [3.74, 3.75] | 1.27 [1.26, 1.28] |
| lstm | floa | 16 | 1536 | 2.31 [2.30, 2.31] | 2.94 [2.90, 2.96] |
| lstm | floa | 16 | 1792 | 3.23 [3.22, 3.24] | 3.06 [3.04, 3.08] |
| lstm | floa | 16 | 1920 | 3.51 [3.26, 4.20] | 3.13 [3.09, 3.16] |
| lstm | floa | 16 | 1984 | 3.57 [3.56, 3.58] | 3.75 [3.69, 3.80] |
| lstm | floa | 64 | 1536 | 3.07 [3.04, 3.10] | 1.05 [1.02, 1.11] |
| lstm | floa | 64 | 1792 | 6.08 [5.93, 6.48] | 0.91 [0.91, 0.92] (no) |
| lstm | floa | 64 | 1920 | 6.28 [6.27, 6.29] | 1.10 [1.09, 1.11] |
| lstm | floa | 64 | 1984 | 6.53 [6.49, 6.62] | 1.35 [1.34, 1.36] |
| slstm | bflo | 16 | 1536 | 3.04 [2.57, 3.85] | 4.34 [4.31, 4.37] |
| slstm | bflo | 16 | 1792 | 5.66 [5.64, 5.68] | 4.85 [4.82, 4.89] |
| slstm | bflo | 16 | 1920 | 6.53 [6.51, 6.55] | 4.01 [4.00, 4.03] |
| slstm | bflo | 16 | 1984 | 6.68 [6.65, 6.70] | 4.37 [4.33, 4.42] |
| slstm | bflo | 64 | 1536 | 3.85 [3.81, 3.96] | 0.82 [0.82, 0.82] (no) |
| slstm | bflo | 64 | 1792 | 4.35 [4.34, 4.35] | 0.82 [0.81, 0.82] (no) |
| slstm | bflo | 64 | 1920 | 6.43 [6.30, 6.80] | 1.15 [1.14, 1.16] |
| slstm | bflo | 64 | 1984 | 3.79 [3.78, 3.79] | 1.37 [1.36, 1.38] |
| slstm | floa | 16 | 1536 | 2.51 [2.47, 2.60] | 3.12 [3.07, 3.15] |
| slstm | floa | 16 | 1792 | 3.36 [3.35, 3.37] | 3.11 [3.10, 3.13] |
| slstm | floa | 16 | 1920 | 4.12 [3.46, 5.35] | 3.23 [3.20, 3.25] |
| slstm | floa | 16 | 1984 | 3.72 [3.70, 3.73] | 3.95 [3.91, 3.99] |
| slstm | floa | 64 | 1536 | 2.86 [2.84, 2.87] | 1.07 [1.07, 1.07] |
| slstm | floa | 64 | 1792 | 5.48 [5.47, 5.48] | 0.94 [0.93, 0.95] (no) |
| slstm | floa | 64 | 1920 | 6.13 [6.11, 6.14] | 1.17 [1.17, 1.18] |
| slstm | floa | 64 | 1984 | 6.60 [6.45, 7.03] | 1.45 [1.44, 1.46] |

The six W4 misses are all B = 64 at D = 1536 or 1792 (0.78-0.94x): the same batch-tile serialization as in section 7
(one resident group of 96-112 CTAs, four 16-row tiles one after another). FlashRNN1 `cuda_fused` itself runs to
D = 2560 (section 7) but is 2.3-6.7x slower than the hybrid kernel at D = 1536-1984.


## 9. Results at D = 2048 with two warpgroups (job "maxd 2048", 2026-10-17)

**W1-2048**: 4/4 forward cases pass (LSTM/sLSTM, FP16/BF16). **W3-2048** holds 8/8, **W4-2048** holds 8/8 (forward, T = 256).

| cell | dtype | B | vs cuda_fused | vs cuda |
| --- | --- | --- | --- | --- |
| lstm | bflo | 16 | 7.61 [7.59, 7.63] | 5.60 [5.57, 5.64] |
| lstm | bflo | 64 | 3.56 [3.43, 3.93] | 1.24 [1.24, 1.25] |
| lstm | floa | 16 | 7.66 [7.25, 8.83] | 5.18 [5.15, 5.20] |
| lstm | floa | 64 | 5.35 [5.21, 5.76] | 1.38 [1.38, 1.39] |
| slstm | bflo | 16 | 7.51 [7.49, 7.54] | 5.95 [5.91, 5.98] |
| slstm | bflo | 64 | 3.46 [3.46, 3.46] | 1.36 [1.36, 1.37] |
| slstm | floa | 16 | 7.38 [7.35, 7.40] | 5.43 [5.40, 5.46] |
| slstm | floa | 64 | 5.12 [5.12, 5.13] | 1.44 [1.43, 1.44] |

## 10. Wide batch tiles with streamed h for B = 64 (addendum, 2026-10-17, before any run of it)

At B = 64 and D ≥ 1280 the forward lost to FlashRNN1 `cuda` (sections 7-8) because one batch-tile group fills the
GPU and the four 16-row tiles run one after another. `flashrnn/flashrnn2/tilelang_stream.py`
(`KernelConfig(stream_h=True, batch_tile=64, reg_k=...)`) carries 64 batch rows per CTA and step (WGMMA N = 64): the
first reg_k rows of the CTA's R slice sit in register fragments of 64 rows each, the rest in a shared buffer of 64-row
chunks, and h^T is no longer staged whole but streamed from the exchange buffer in 64-row chunks (two alternating
shared tiles for the register chunks, a two-stage pipelined copy for the shared chunks). Exchange, epilogue and
outputs follow the grid kernel. The kernel source is generated per reg_k (one fragment name per chunk) and imported
from a cache. Login-node compile (FP16, LSTM/sLSTM): BM = 64 at D = 1024, 1280 (reg_k 0), 1536 (128), 1792 (384) with
0 spill; ptxas reports serialized WGMMAs (each chunk's GEMM waits before the next copy). D must be a multiple of 64.

Runs (batch "bm64"):
* **Qualification** (W1-bm64): forward within the FP64 budget at D ∈ {1024, 1280, 1536, 1792}, BM = 64, B ∈ {64, 40}
  (40: one partial tile), LSTM/sLSTM, FP16/BF16, T = 24.
* **Paired timing**, forward, B = 64, T = 256, D ∈ {1024, 1280, 1536, 1792}, LSTM/sLSTM, FP16/BF16, protocol of
  section 3: **W6** the BM = 64 kernel against the previous FlashRNN2 configuration at the same D (TileLang grid BM = 16
  at D ≤ 1280, register residency BM = 16 above); **W4-bm64** against FlashRNN1 `cuda`; **W3-bm64** against `cuda_fused`.
  Ratio = other time / BM = 64 time; a check holds for a shape when the lower 95% bound is ≥ 1.

## 11. First training run (job "bw 2048", 2026-10-17) and reruns

W1-train as pre-registered: **FAIL**, 8/16 on the large-D path (control: Triton backward at D = 1024, 4/4). Every
D = 1536 and D = 1792 case died at kernel trace with `NameError: HYBRID_THREADS` — the two-warpgroup change had removed
a constant the one-warpgroup forward still used (a code defect of this session, not a numerical failure). The cases
that ran pass: the CUDA backward at D = 1024 (forced) has the same FP64 gradient error as the Triton backward on the
same inputs (FP16 rel. 3.0e-4 / 3.8e-4 for d_wx / dR, BF16 2.4e-3 / 2.7e-3), and the full training step at D = 2048
passes in all four (cell, dtype) cases (FP16 3.0e-4, BF16 2.4e-3). Because the path did not qualify, no paired timing
ran. The B = 64 batch (section 10) carried the same defect in its comparison arm and was cancelled before it started.

Fix: the constant restored; and every batch now runs `tools/tl_trace_check.py` on the login node, which traces and
compiles every TileLang kernel of the plan (both FlashRNN2 arms, forward and training modes), and the seal requires it
to pass. Both batches are rerun unchanged otherwise ("bw 2048 v2", "bm64 v2").

## 12. Race verification of the CUDA backward (pre-registration, 2026-10-17, before any run of it)

`hybrid_bw.cu` gains a verification build (`FRNN_VERIFY`, compiled out when 0; 181 registers either way at D = 2048,
194 with the checks): a content check of every A-fragment element read from the exchange buffer against the sender's
stored FP32 dG rounded the same way, once after the read and once after the MMAs; seeded 0-4 us delays before the
publish and after the grid barrier; and two broken variants as negative controls (single buffer; no grid barrier
except before the tail). The uninstrumented build differs from the earlier one only by the error-buffer kernel
argument (the SASS is therefore not byte-identical to the build that ran in section 11).

`tools/hybrid_bw_race.py`, per task (D ∈ {1024, 1536, 2048}; LSTM/sLSTM; FP16; B = 40, i.e. three batch tiles and, at
D ≥ 1536, three groups serialized in slots; T = 48), 21 runs per variant (no delay, then 20 seeds), outputs (dG and
the initial-state gradient) compared bitwise with the uninstrumented kernel. Rule: the correct kernel passes when every
run counts zero violations and its outputs are bitwise equal to the uninstrumented reference; a broken variant is
detected when violations are counted in at least 80% of its runs and no run changes the outputs without counting a
violation.

## 13. B = 64 rerun ("bm64 v2", 2026-10-17): deadlock, and the third run

The first qualification case hung for its full 1800 s timeout (the batch stopped at the canary, as designed). Cause,
found in the generated CUDA: TileLang lowered the pipelined h-chunk loop with automatic warp specialization — block
size 256 instead of the configured 128, a TMA producer warpgroup (threads 0-127) and the 128 consumer threads — and
only the consumer branch reaches the per-step cooperative grid barrier, so the grid never completes a step. The
login-node trace check compiled the kernel but did not compare the block size. Fixes: `tilelang_stream` compiles with
`tl.disable_warp_specialized`; `tl_trace_check.py` now fails any kernel whose `__launch_bounds__` differs from the
configured thread count. The batch is rerun unchanged otherwise ("bm64 v3").

## 14. Training to D = 2048 (job "bw 2048 v2", 2026-10-17)

**W1-train**: 16/16 large-D training cases pass (D = 1024 with the CUDA backward forced, D = 1536, 1792, 2048 with the library choice; LSTM/sLSTM, FP16/BF16); Triton-backward control 4/4. **W3-train** holds 18/24, **W4-train** 10/24 (training step, T = 256).

| cell | dtype | B | D | vs cuda_fused | vs cuda |
| --- | --- | --- | --- | --- | --- |
| lstm | bflo | 16 | 1536 | 0.85 [0.85, 0.85] (no) | 1.15 [1.13, 1.16] |
| lstm | bflo | 16 | 1792 | 1.17 [1.17, 1.17] | 1.08 [1.07, 1.09] |
| lstm | bflo | 16 | 2048 | 1.59 [1.59, 1.59] | 1.51 [1.50, 1.52] |
| lstm | bflo | 64 | 1536 | 1.06 [1.06, 1.06] | 0.36 [0.36, 0.37] (no) |
| lstm | bflo | 64 | 1792 | 1.06 [1.06, 1.06] | 0.31 [0.31, 0.32] (no) |
| lstm | bflo | 64 | 2048 | 1.25 [1.22, 1.26] | 0.56 [0.56, 0.56] (no) |
| lstm | floa | 16 | 1536 | 0.73 [0.61, 0.79] (no) | 1.07 [1.07, 1.08] |
| lstm | floa | 16 | 1792 | 0.86 [0.86, 0.86] (no) | 0.85 [0.84, 0.86] (no) |
| lstm | floa | 16 | 2048 | 1.35 [1.27, 1.38] | 1.45 [1.44, 1.46] |
| lstm | floa | 64 | 1536 | 1.08 [1.08, 1.08] | 0.40 [0.39, 0.40] (no) |
| lstm | floa | 64 | 1792 | 1.30 [1.30, 1.30] | 0.34 [0.34, 0.35] (no) |
| lstm | floa | 64 | 2048 | 1.49 [1.49, 1.50] | 0.57 [0.56, 0.57] (no) |
| slstm | bflo | 16 | 1536 | 0.94 [0.94, 0.94] (no) | 1.27 [1.26, 1.27] |
| slstm | bflo | 16 | 1792 | 1.18 [1.18, 1.18] | 1.15 [1.12, 1.17] |
| slstm | bflo | 16 | 2048 | 1.79 [1.79, 1.79] | 1.74 [1.73, 1.75] |
| slstm | bflo | 64 | 1536 | 1.24 [1.24, 1.25] | 0.40 [0.39, 0.40] (no) |
| slstm | bflo | 64 | 1792 | 1.19 [1.19, 1.19] | 0.36 [0.36, 0.37] (no) |
| slstm | bflo | 64 | 2048 | 1.25 [1.25, 1.25] | 0.57 [0.56, 0.57] (no) |
| slstm | floa | 16 | 1536 | 0.84 [0.84, 0.84] (no) | 1.16 [1.14, 1.17] |
| slstm | floa | 16 | 1792 | 0.87 [0.81, 0.90] (no) | 0.91 [0.90, 0.92] (no) |
| slstm | floa | 16 | 2048 | 1.57 [1.57, 1.58] | 1.73 [1.71, 1.73] |
| slstm | floa | 64 | 1536 | 1.12 [1.12, 1.13] | 0.44 [0.44, 0.44] (no) |
| slstm | floa | 64 | 1792 | 1.45 [1.45, 1.45] | 0.39 [0.39, 0.39] (no) |
| slstm | floa | 64 | 2048 | 1.51 [1.51, 1.51] | 0.57 [0.57, 0.58] (no) |

Reading (post hoc): at B = 16 the step is faster than both FlashRNN1 backends at D = 2048 (1.35-1.79x) and mixed at
D = 1536-1792; at B = 64 it beats `cuda_fused` (1.06-1.49x) but loses to `cuda` (0.31-0.57x): forward and backward
both run the four 16-row batch tiles one after another. The B = 64 forward fix is section 10/13; the backward needs the
same widening.


## 15. Wide batch tiles in the CUDA backward, B = 64 training (pre-registration, 2026-10-17, before any run of it)

`hybrid_bw.cu` now carries BM ∈ {16, 32, 64} batch rows per CTA (`FRNN_BM`; `hybrid_bw.batch_tile`: 16 up to B = 16,
32 up to 32, else 64): every warp runs BM/16 mma row tiles against each R fragment, every thread PE = BM·16/256 pointwise
elements, the partial-sum buffer is [8, BM, 16]. At BM = 16 the arithmetic is unchanged. ptxas (FP16, LSTM/sLSTM,
sm_90a): BM = 64 at D = 1536 113-128 registers, D = 1792 252-254, D = 2048 254-255 (LSTM: 4 bytes spill), BM = 32 at
D = 2048 250, all others 0 spill. The kernel body changed, so the race verification of section 12 (job "bw race", not
yet started) was cancelled and is folded into this batch for the new kernel.

Runs (batch "train64"):
* **Race verification** (rule of section 12): D ∈ {1536, 2048} × LSTM/sLSTM × BM ∈ {16 (B = 40, three tiles), 64
  (B = 100, two tiles, one partial)}, T = 48, 21 runs per variant.
* **Qualification** (W1-train64): training step within the FP64 budgets with the B = 64 configuration — forward: the
  streamed-h BM = 64 kernel (section 10) at D ∈ {1024, 1536, 1792}, the two-warpgroup kernel at D = 2048; backward:
  BM = 64 — at B ∈ {64, 40}, LSTM/sLSTM, FP16/BF16, T = 24.
* **Paired timing**, training step, B = 64, T = 256, D ∈ {1536, 1792, 2048}, LSTM/sLSTM, FP16/BF16: **W6-train64** against
  the previous configuration (BM = 16 forward and backward), **W3-train64** against FlashRNN1 `cuda_fused`,
  **W4-train64** against `cuda`; protocol and rule of section 3.

## 16. Results of the B = 64 wide-tile forward (job "bm64 v3", 2026-10-17)

**W1-bm64**: 32/32 (D = 1024-1792, B = 64 and 40, LSTM/sLSTM, FP16/BF16). **W6** (previous 16-row-tile configuration / BM = 64) holds 12/16, **W4-bm64** (FlashRNN1 `cuda`) 16/16, **W3-bm64** (`cuda_fused`) 16/16. Forward, B = 64, T = 256.

| cell | dtype | D | vs previous BM = 16 | vs cuda | vs cuda_fused |
| --- | --- | --- | --- | --- | --- |
| lstm | bflo | 1024 | 0.93 [0.93, 0.93] (no) | 1.42 [1.40, 1.44] | 3.89 [3.88, 3.89] |
| lstm | bflo | 1280 | 1.87 [1.87, 1.88] | 1.43 [1.41, 1.44] | 4.18 [4.18, 4.19] |
| lstm | bflo | 1536 | 1.70 [1.70, 1.70] | 1.34 [1.34, 1.35] | 4.90 [4.89, 4.91] |
| lstm | bflo | 1792 | 1.62 [1.62, 1.62] | 1.30 [1.29, 1.31] | 5.40 [5.40, 5.41] |
| lstm | floa | 1024 | 0.93 [0.93, 0.94] (no) | 1.39 [1.37, 1.41] | 3.83 [3.82, 3.85] |
| lstm | floa | 1280 | 1.86 [1.84, 1.86] | 1.73 [1.71, 1.75] | 4.49 [4.38, 4.78] |
| lstm | floa | 1536 | 1.70 [1.70, 1.70] | 1.73 [1.71, 1.74] | 5.34 [5.32, 5.36] |
| lstm | floa | 1792 | 1.62 [1.61, 1.62] | 1.46 [1.45, 1.47] | 8.61 [8.59, 8.63] |
| slstm | bflo | 1024 | 1.00 [1.00, 1.01] (no) | 1.55 [1.55, 1.56] | 2.88 [2.87, 2.90] |
| slstm | bflo | 1280 | 1.91 [1.90, 1.92] | 1.54 [1.54, 1.55] | 6.34 [6.31, 6.37] |
| slstm | bflo | 1536 | 1.72 [1.72, 1.72] | 1.41 [1.40, 1.43] | 6.63 [6.63, 6.64] |
| slstm | bflo | 1792 | 1.68 [1.68, 1.69] | 1.38 [1.36, 1.39] | 7.32 [7.31, 7.33] |
| slstm | floa | 1024 | 1.00 [1.00, 1.01] (no) | 1.49 [1.48, 1.50] | 3.04 [2.88, 3.52] |
| slstm | floa | 1280 | 1.92 [1.92, 1.92] | 1.85 [1.83, 1.86] | 4.75 [4.72, 4.76] |
| slstm | floa | 1536 | 1.72 [1.72, 1.72] | 1.84 [1.82, 1.85] | 4.96 [4.95, 4.97] |
| slstm | floa | 1792 | 1.68 [1.68, 1.68] | 1.54 [1.52, 1.56] | 9.26 [9.23, 9.29] |

The four W6 misses are D = 1024 (0.93-1.00x), where two 16-row groups were already resident. The losses to FlashRNN1
`cuda` at B = 64 (sections 7-8) are gone: 1.30-1.85x. The library now selects this kernel by default for B > 32 at
D = 1280-1792 (`execution_policy.wide_batch_config`); D = 1024 and D = 2048 keep their previous defaults.


## 17. Results of the B = 64 training batch (job "train64", 2026-10-17)

**Race verification (section 12 rule, new kernel)**: 8/8 tasks pass — D ∈ {1536, 2048} × LSTM/sLSTM × BM ∈ {16, 64}: the correct kernel counts zero violations in 21/21 runs with outputs bitwise equal to the uninstrumented kernel; the single-buffer variant is flagged in 21/21 runs, the no-barrier variant in 21/21 (outputs changed in all 21, never silently).

**W1-train64**: 32/32. **W6-train64** (previous 16-row tiles / new) holds 12/12, **W3-train64** (`cuda_fused`) 12/12, **W4-train64** (`cuda`) 0/12. Training step, B = 64, T = 256.

| cell | dtype | D | vs previous | vs cuda_fused | vs cuda |
| --- | --- | --- | --- | --- | --- |
| lstm | bflo | 1536 | 1.28 [1.28, 1.28] | 1.37 [1.37, 1.38] | 0.47 [0.47, 0.47] (no) |
| lstm | bflo | 1792 | 1.35 [1.34, 1.35] | 1.51 [1.51, 1.51] | 0.43 [0.43, 0.43] (no) |
| lstm | bflo | 2048 | 1.11 [1.10, 1.16] | 1.47 [1.46, 1.47] | 0.64 [0.63, 0.64] (no) |
| lstm | floa | 1536 | 1.27 [1.27, 1.27] | 1.40 [1.40, 1.40] | 0.51 [0.50, 0.52] (no) |
| lstm | floa | 1792 | 1.35 [1.35, 1.35] | 1.83 [1.79, 1.89] | 0.48 [0.48, 0.48] (no) |
| lstm | floa | 2048 | 1.10 [1.10, 1.10] | 1.71 [1.71, 1.71] | 0.66 [0.65, 0.66] (no) |
| slstm | bflo | 1536 | 1.26 [1.26, 1.26] | 1.60 [1.60, 1.60] | 0.51 [0.51, 0.52] (no) |
| slstm | bflo | 1792 | 1.33 [1.32, 1.34] | 1.61 [1.61, 1.61] | 0.49 [0.48, 0.50] (no) |
| slstm | bflo | 2048 | 1.22 [1.22, 1.22] | 1.57 [1.57, 1.57] | 0.71 [0.70, 0.71] (no) |
| slstm | floa | 1536 | 1.26 [1.26, 1.26] | 1.48 [1.47, 1.48] | 0.56 [0.56, 0.57] (no) |
| slstm | floa | 1792 | 1.34 [1.33, 1.34] | 1.91 [1.88, 1.92] | 0.55 [0.54, 0.55] (no) |
| slstm | floa | 2048 | 1.22 [1.22, 1.22] | 1.88 [1.88, 1.88] | 0.71 [0.71, 0.71] (no) |

Reading (post hoc): the wide tiles help training by 1.10-1.35x, but the step still loses to the alternating backend at
B = 64 (0.43-0.71x). Bytes moved per step explain it: with R resident and split over k = D/16 CTAs, every CTA reads the
whole exchanged operand, k·B·D elements in the forward (h) and k·B·4D in the backward (dG), while the alternating
backend streams R once, 4D² elements. The ratio is B/64 in the forward and B/16 in the backward, so at B = 64 the
backward's broadcast moves four times what streaming R does. Reducing it needs the exchange to be loaded once per
cluster (TMA multicast within thread-block clusters) or a backward that reduces partial dh instead of broadcasting dG;
neither is built.

