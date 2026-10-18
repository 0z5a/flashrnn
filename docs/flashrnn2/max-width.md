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


## 18. Distributed-shared-memory exchange in the CUDA backward, and a diagnosis of the B = 64 loss (pre-registration, 2026-10-17, before any run of it)

**Change.** `hybrid_bw.cu` gains `FRNN_CLUSTER = 8` (`hybrid_bw.backward(cluster=8)`, training option `cluster`; the
default stays 0). Eight neighbouring CTAs (column slices) form a thread-block cluster; all of them need the same
exchanged dG operand of their batch tile, and warp w of every CTA reads the same K range. Per round of 64 rows of each
warp's K range, cluster rank q loads warp q's [BM × 64] chunk from L2 with 128-bit loads into its shared stage (same
offset in every CTA, double buffered, aliasing the partial-sum buffer, row stride 72 elements so the fragment reads are
conflict-free), the cluster synchronizes, and warp w of every CTA reads its A fragments from rank w's stage through
distributed shared memory. L2 reads of the exchange drop eightfold; the price is one cluster barrier per round (KW/64 =
16 per step at D = 2048, 12 at 1536) and remote shared loads. The MMAs run in the same order as before (register rows
first, then shared rows, ascending), so the outputs must be bitwise equal to the plain kernel's. The launch is
`cudaLaunchKernelEx` with the cluster-dimension and cooperative attributes; co-resident groups are capped by
`cudaOccupancyMaxActiveClusters`. ptxas (FP16, sm_90a): BM = 64 at D = 1536 152-172 registers, D = 1792 228-250, D = 2048
255 with 4-36 bytes of spill (the verification builds 350-430 bytes; they are not timed), BM = 16 at D = 2048 192, no
other spill. This changes two things at once, the load width (32-bit fragment loads from L2 become 128-bit staged
loads) and the volume (one L2 read per cluster instead of per CTA); the profile below is there to separate them.

Also in this source: phase timers (`prof`, verification bit 16: globaltimer around the reduction, the pointwise update
plus publish, and the grid barrier, thread 0 of each CTA), used only in the report-only profile runs; paired timings run
without them.

**Runs (batch "dsmem")**, single GPU, B = 64, T = 256 unless stated:
* **C0, feasibility** (first, report only): per D ∈ {1024, 1280, 1536, 1792, 2048} and BM ∈ {16, 64}, the number of
  co-resident clusters the GPU accepts and whether a cluster launch (cooperative + cluster attributes) runs; a one-call
  backward (T = 4) bitwise against the plain kernel. A shape where the cluster launch does not fit or is rejected is a
  reported outcome, not an error: its cluster cases fail qualification and are not timed; the plain-kernel runs below
  are unaffected.
* **R-dsmem, race verification** (rule of section 12) of the cluster kernel: D ∈ {1536, 2048} × LSTM/sLSTM at BM = 64
  (B = 100) and D = 2048 × LSTM/sLSTM at BM = 16 (B = 40), T = 48, 21 runs per variant. Two broken variants are added for
  the new synchronization, a single stage buffer and no cluster barrier between rounds (the last round's stays); the
  content check also covers the A operands as read through distributed shared memory. Pass also needs the
  uninstrumented outputs bitwise equal to the plain kernel's.
* **W1-dsmem, qualification**: training step within the FP64 budgets with the cluster backward, D ∈ {1536, 1792, 2048},
  B ∈ {64, 40}, LSTM/sLSTM, FP16/BF16, T = 24 (24 cases; forward as in section 15). The plain BM = 64 backward is
  re-qualified first on this source (D ∈ {1536, 2048}, B = 64, 8 cases; the first of them is the batch's canary).
* **Paired timing**, training step, D ∈ {1536, 1792, 2048} × LSTM/sLSTM × FP16/BF16, cluster backward against
  **W6-dsmem** the plain BM = 64 backward (same forward) and **W4-dsmem** FlashRNN1 `cuda`; protocol and rule of
  section 3 (holds when the lower bound of other / cluster is ≥ 1). A slowdown is a legitimate outcome: it would mean the
  cluster barriers and remote loads cost more than the L2 traffic they remove.
* **Profile, report only.** (a) Step breakdown at D ∈ {1536, 2048}, LSTM FP16 and sLSTM BF16: training forward, and per
  backward variant (plain, cluster; each with and without the phase timers) the preactivation recompute, the kernel and
  the dR/dbias GEMMs (CUDA events), the in-kernel phases per step, next to FlashRNN1 `cuda`'s step. (b) Nsight Systems
  timelines at D ∈ {1536, 2048}, LSTM FP16: plain, cluster, FlashRNN1 `cuda`. (c) Nsight Compute (kernel replay) of the
  first backward launch, plain and cluster at D ∈ {1536, 2048}, LSTM FP16, T = 32: SpeedOfLight,
  MemoryWorkloadAnalysis (+ tables), WarpStateStats, SchedulerStats, SourceCounters, plus L2 read sectors / requests. If
  Nsight Compute is not permitted on the node, the in-kernel phase timers of (a) are the fallback.
* **H > 1 follow-up** (heads.md section 8.1): qualification of the R-tiled path at D = 768 with the corrected H > 1
  default (H ∈ {2, 3, 16}, LSTM/sLSTM, FP16/BF16, 12 cases); step breakdown with the new default at D = 64, H = 256 and
  D = 128, H = 1024 (LSTM FP16, against FlashRNN1), and Nsight Systems of both at D = 64, H = 1024 (report only).

**Reading rule for the profile (fixed now).** For the plain kernel: (i) L2 throughput ≥ 70% of peak with ≥ 80% of the
requested sectors used → the exchange broadcast is bandwidth-bound, and the cluster's volume cut is the lever; (ii)
sector use near 50% and long-scoreboard the largest stall → the load width is the limiter (the 128-bit staging alone
would help); (iii) non-kernel phases > 30% of the step → the FP32 passes around the kernel are the lever. Attribution of
any cluster gain: the ratio of L2 read sectors (plain / cluster; about 8 for the exchange part if the volume cut acts),
sectors per request, and the stall mix (long scoreboard against barrier / membar) of both kernels at the same shape. A
gain with an L2 read-sector ratio near 1 would contradict the volume explanation.

## 19. Results of the DSMEM batch (job "dsmem", 2026-10-17)

Evidence: `max-width-evidence/dsmem-analysis.json` and the archive (SHA-256 verified).

**C0, feasibility.** The GPU holds at most 15 co-resident clusters of 8 CTAs (`cudaOccupancyMaxActiveClusters`, at one
CTA per SM), i.e. 120 CTAs. D ≤ 1792 needs at most 14 clusters per group and runs (cooperative + cluster launch
accepted; output bitwise equal to the plain kernel at every D ∈ {1024, 1280, 1536, 1792}, BM ∈ {16, 64}). D = 2048
needs 16 clusters and does not fit: every D = 2048 cluster case (qualification, timing, race, profile) reports "no
group of clusters fits on the GPU", the outcome pre-registered as reportable.

**R-dsmem**: 2/2 runnable tasks pass (D = 1536, LSTM and sLSTM, BM = 64): the correct kernel 0 violations in 21/21
runs, bitwise equal to the uninstrumented kernel and to the plain kernel; all four broken variants (single exchange
buffer, no grid barrier, single stage buffer, no cluster barrier between rounds) flagged in 21/21 runs, never silently.
The four D = 2048 tasks did not run (C0).

**W1-dsmem**: 16/16 at D ∈ {1536, 1792}, 0/8 at D = 2048 (C0). Plain BM = 64 re-qualification 8/8. H > 1 follow-up
(heads.md section 8.1): the R-tiled path at D = 768 with the corrected default 12/12.

**W6-dsmem 0/12, W4-dsmem 0/12** (8 measured, D = 2048 not run). Training step, B = 64, T = 256:

| cell | dtype | D | plain / cluster | cuda / cluster |
| --- | --- | --- | --- | --- |
| lstm | floa | 1536 | 0.71 [0.71, 0.71] | 0.38 [0.37, 0.38] |
| lstm | floa | 1792 | 0.75 [0.75, 0.75] | 0.37 [0.36, 0.38] |
| lstm | bflo | 1536 | 0.69 [0.69, 0.70] | 0.36 [0.35, 0.37] |
| lstm | bflo | 1792 | 0.76 [0.76, 0.76] | 0.36 [0.36, 0.36] |
| slstm | floa | 1536 | 0.69 [0.69, 0.70] | 0.40 [0.40, 0.40] |
| slstm | floa | 1792 | 0.74 [0.74, 0.74] | 0.41 [0.41, 0.42] |
| slstm | bflo | 1536 | 0.70 [0.70, 0.70] | 0.39 [0.38, 0.40] |
| slstm | bflo | 1792 | 0.75 [0.74, 0.75] | 0.38 [0.38, 0.38] |

The cluster exchange makes the training step 1.3-1.45x slower than the plain kernel; the slowdown pre-registered as
a legitimate outcome happened.

**Profile.** Nsight Compute did not run: the command named a section that this Nsight Compute version does not have
(`LaunchStatistics`), a harness error of this batch, so the reading rule's L2-sector and stall counters are missing.
What was measured (CUDA events, in-kernel phase timers, Nsight Systems):

| D = 1536, LSTM FP16, B = 64, per call | plain | cluster | FlashRNN1 `cuda` |
| --- | --- | --- | --- |
| training step | 26.1 ms | 37.0 ms | 13.4 ms |
| backward recurrent kernel(s) | 16.6 ms | 27.6 ms | ~5 ms (cuBLAS dh = dG·Rᵀ per step) + ~0.9 ms pointwise |
| per step in the kernel: reduction / pointwise + publish / grid barrier | 47.5 / 12.1 / 4.8 µs | 89.0 / 11.8 / 4.0 µs | — |
| training forward / preactivation recompute / dR + dbias | 3.3 / 2.7 / 1.7 ms | same | — |

At D = 2048 (plain): step 37.2 ms against `cuda`'s 26.4 ms; kernel 21.2 ms, reduction 64 µs of a 81 µs step.

Reading (post hoc; the pre-registered counters are missing, so the rule's cases (i) and (ii) cannot be decided):
* The reduction phase is three quarters of the plain kernel's step, and the cluster version doubles it, while
  pointwise and barrier stay the same. So the cluster cost lies in the reduction: 12 cluster barriers per step and
  32-bit remote shared loads. Cutting the L2 reads eightfold did not make the reduction faster, so L2 bandwidth was not
  its limit at this shape. This is the opposite of what case (i) needs.
* Non-kernel phases (forward, recompute, weight gradients, copies) are about 30% of the plain step, at the border of
  case (iii).
* Against `cuda` the gap is the kernel itself. cuBLAS reads R from L2 once per step (4·D² · 2 B = 19 MB at D = 1536,
  about 12 µs per step). The resident-R kernel reads the whole exchanged dG in every CTA (96 · 64 · 6144 · 2 B = 75 MB
  per step) through 32-bit fragment loads, at an effective 1.6 TB/s. Neither shared staging nor a smaller operand
  volume per CTA helped while every CTA still reduces the full K = 4D. What would change the volume is a 2-D split of R
  (CTA (i, j) holds a [4D/q × D/p] block, reads only its dG slice, writes an FP32 partial dh reduced over q CTAs). Per
  step that moves 8 · (p + q) · B · D bytes (FP16 dG slices: 8 · p · B · D; FP32 partials written and read:
  8 · q · B · D) against the 1-D kernel's 8 · k · B · D (k = D / 16): at D = 1536 with p = 8, q = 16 about 19 MB
  against 75 MB (4x; at most about 4.9x for p ≈ q ≈ √(p q)), the same order as the 19 MB of R that cuBLAS reads per step.
  The price is a second grid barrier per step. (An earlier draft of this paragraph said 7x; it mixed FP16 and FP32
  element sizes.)

The cluster variant stays opt-in (`cluster=8`) and is not a default.

## 20. R split in two dimensions in the CUDA backward (pre-registration, 2026-10-18, before any run of it)

**Change.** `hybrid_bw2d.cu` / `hybrid_bw2d.py`, training option `backend="hybrid2d"` (opt-in, not a default). The
PC × Q CTAs of a group form a grid; CTA (i, j) keeps the R block [K block j (4D / Q rows) × column block i (D / PC
columns)] resident and per step: pointwise update of its D / PC / Q owned columns (dh → dG_t, into dg and the
exchange buffer) → grid barrier B2 → GEMM of its staged dG K block (128-bit loads, double-buffered 64-row chunks) with
the resident R block → FP32 partial dh → grid barrier B1 → owners sum the Q partials of their columns in fixed order.
The first step takes dh from d_final, the GEMM of step 0 gives the initial-state gradient; single exchange and partial
buffers (the two barriers separate every write from every read; argument in the source). Grids (CTAs = PC · Q):
D = 1024 8 × 16, D = 1280 5 × 16, D = 1536 8 × 16 (R block in shared memory), D = 2048 8 × 16 with 8 of 32 k-steps per
warp in registers; D = 1792 has no fitting grid and is not supported. ptxas (sm_90a, FP16, BM = 64): 116 / 184 / 156 /
238 registers at D = 1024 / 1280 / 1536 / 2048, no spill in any build including the verification builds (≤ 255);
shared memory 82 / 182 / 166 / 215 KB (limit 227 KB).

**Bytes per step** (section 19, corrected): 8 · (PC + Q) · B · D against 8 · (D / 16) · B · D for the 1-D kernel:
at D = 1536, B = 64 about 19 MB against 75 MB, at D = 2048 about 25 MB against 134 MB — the same order as the R bytes
cuBLAS reads per step (19 / 34 MB). The win, if any, comes from staying inside one kernel; the price is a second grid
barrier per step (the 1-D kernel's barrier phase was 4-5 µs per step).

**Runs (batch "bw2d"):**
* **R-2d, race verification** (rule of section 12): D ∈ {1536, 2048} × LSTM/sLSTM at BM = 64 (B = 100, two tiles) and
  D = 2048 × LSTM at BM = 16 (B = 40), T = 48, 21 runs per variant. Detectors: content check of every staged dG
  operand against the stored FP32 dG (barrier B2), and a step stamp per K block written after each partial (release)
  and checked (acquire) by every partial reader (barrier B1). Broken variants: no B1, no B2. Also recorded: the largest
  difference to the 1-D kernel (not bitwise: different summation order).
* **W1-2d, qualification**: training step within the FP64 budgets of the earlier W1 runs with `backend="hybrid2d"`,
  D ∈ {1024, 1536, 2048}, B ∈ {64, 40}, LSTM/sLSTM, FP16/BF16, T = 24 (24 cases; forward as in sections 15/18).
* **Paired timing**, training step, B = 64, T = 256, D ∈ {1024, 1536, 2048} × LSTM/sLSTM × FP16/BF16, 2-D backward
  against **W6-2d** the 1-D BM = 64 backward (same forward) and **W4-2d** FlashRNN1 `cuda`; protocol and rule of
  section 3. Prediction band (stated now): against the 1-D kernel 1.0-2.0x; against `cuda` anywhere between 0.6x and
  1.2x — a loss to `cuda` remains a legitimate outcome.
* **Profile, report only**: step breakdown with in-kernel phases (pointwise, B2, GEMM, B1, partial sum) at
  D ∈ {1536, 2048}, LSTM FP16 and sLSTM BF16; Nsight Systems of the 2-D step at D ∈ {1536, 2048}; Nsight Compute of the
  first 2-D and 1-D backward launch at D ∈ {1536, 2048}, T = 32 (sections checked against `ncu --list-sections` on
  the login node before sealing).

## 21. Results of the 2-D split backward (job "bw2d", 2026-10-18)

Evidence: `max-width-evidence/bw2d-analysis.json` and the archive (SHA-256 verified).

**R-2d**: 5/5 tasks pass (D ∈ {1536, 2048} × LSTM/sLSTM at BM = 64, B = 100; D = 2048 LSTM at BM = 16, B = 40). The
correct kernel counts zero violations in 21/21 runs per task and its outputs are bitwise equal to the uninstrumented
kernel. Both broken variants are flagged in 21/21 runs, never silently: without B1 the step stamps go stale, and
without B2 the staged operands do not match the stored dG. Difference to the 1-D kernel: dG at most 7.1e-6 relative
to its largest value, d_initial at most 4.8e-5.

**W1-2d**: 24/24 (D ∈ {1024, 1536, 2048}, B ∈ {64, 40}, LSTM/sLSTM, FP16/BF16); 1-D canary 4/4.

**W6-2d** (1-D / 2-D) holds 12/12, **W4-2d** (`cuda` / 2-D) 7/12. Training step, B = 64, T = 256:

| cell | dtype | D | vs 1-D backward | vs cuda |
| --- | --- | --- | --- | --- |
| lstm | bflo | 1024 | 1.28 [1.28, 1.28] | 1.02 [1.01, 1.03] |
| lstm | bflo | 1536 | 1.58 [1.58, 1.58] | 0.84 [0.84, 0.85] (no) |
| lstm | bflo | 2048 | 1.42 [1.41, 1.42] | 1.03 [1.02, 1.04] |
| lstm | floa | 1024 | 1.28 [1.28, 1.28] | 1.04 [1.02, 1.05] |
| lstm | floa | 1536 | 1.59 [1.59, 1.59] | 0.89 [0.89, 0.90] (no) |
| lstm | floa | 2048 | 1.42 [1.42, 1.43] | 1.06 [1.05, 1.07] |
| slstm | bflo | 1024 | 1.39 [1.38, 1.39] | 1.07 [1.06, 1.08] |
| slstm | bflo | 1536 | 1.61 [1.45, 1.68] | 0.95 [0.89, 0.99] (no) |
| slstm | bflo | 2048 | 1.50 [1.50, 1.50] | 1.15 [1.14, 1.16] |
| slstm | floa | 1024 | 1.39 [1.38, 1.39] | 1.10 [0.96, 1.17] (no) |
| slstm | floa | 1536 | 1.67 [1.67, 1.67] | 0.99 [0.97, 1.01] (no) |
| slstm | floa | 2048 | 1.51 [1.50, 1.51] | 1.16 [1.15, 1.17] |

All rows are inside the pre-registered bands: 1.28-1.67x against the 1-D kernel, 0.84-1.16x against `cuda`. The B = 64
training step now beats FlashRNN1's alternating backend at D = 1024 and D = 2048. At D = 1536 it is still 0.84-0.99x;
the section 17 rows had been 0.47-0.71x.

**Profile.**

| per call, B = 64, LSTM FP16 | D = 1536 1-D | D = 1536 2-D | D = 2048 1-D | D = 2048 2-D |
| --- | --- | --- | --- | --- |
| training step | 24.2 ms | 15.2 ms | 33.9 ms | 23.9 ms |
| backward kernel | 14.8 ms | 6.0 ms | 17.8 ms | 7.9 ms |
| forward / recompute / dR + dbias | 3.3 / 2.7 / 1.7 | same | 7.2 / 3.9 / 2.7 | same |

In-kernel phases of the 2-D kernel, per step (2-D numbers in µs; 1-D shown as its reduction + pointwise/publish +
barrier):
* D = 1536: pointwise 9.6, B2 1.7, GEMM 7.7, B1 2.3, partial sum 1.8. The 1-D kernel spent 47.3 + 6.0 + 4.1.
* D = 2048: pointwise 12.5, B2 2.1, GEMM 11.5, B1 2.3, partial sum 1.4. The 1-D kernel spent 57.7 + 6.3 + 4.6.

Nsight Compute (first backward launch, T = 32, LSTM FP16, B = 64):

| | 1-D D = 1536 | 2-D D = 1536 | 1-D D = 2048 | 2-D D = 2048 |
| --- | --- | --- | --- | --- |
| duration | 1.89 ms | 0.76 ms | 2.27 ms | 0.98 ms |
| L2 read sectors / requests | 119 M / 117 M | 20.0 M / 11.2 M | 152 M / 146 M | 22.1 M / 9.2 M |
| L2 throughput (% of peak) | 59 | 33 | 61 | 28 |
| SM throughput (% of peak) | 5.3 | 10.8 | 7.5 | 11.1 |

Reading against the rule of section 18, now that the counters exist:
* The 1-D kernel issues about one L2 sector per request. Its 32-bit fragment loads take 8 rows × 4 bytes per warp
  instruction.
* At 59-61% of L2 peak it is bound by the L2 request rate, not by bytes. That is case (ii): load width and request
  count.
* The 2-D kernel cuts L2 sectors 6-7x and requests 10-16x, the 128-bit staging plus the smaller volume. That matches
  the 4-5x byte estimate of section 20 plus the staging.
* What remains in the 2-D step is the pointwise phase (FP32 history and preactivation reads) and the GEMM. The two
  barriers cost 4-4.5 µs per step together, about what the single 1-D barrier phase cost.

Also observed: in one profile sweep (`s-prof-lstm-floa-d2048`) FlashRNN1 `cuda` took 169 ms for one call against
about 25 ms elsewhere. The paired claim rows at that shape are unaffected (3 starts × 20 blocks). It is reported, not
explained.

**Default (after the run).** For B > 32 at D ∈ {1536, 2048}, the training backward now uses the 2-D kernel when no
backend is named (`training.py`). D = 1792 keeps the 1-D kernel. B ≤ 32 keeps the 1-D kernel, untimed for the 2-D
one. D ≤ 1280 keeps the Triton backward; the D = 1024 rows above compare against the 1-D CUDA kernel, not against
that default.

## 22. Row ownership of the 2-D backward's pointwise update (pre-registration, 2026-10-18, before any run of it)

**What changes.**
* *The phase being targeted.* In section 21's profile the pointwise phase is the largest per step (9.6 µs at
  D = 1536, 12.5 µs at D = 2048, LSTM FP16).
* *The old layout.* CTA (i, j) owned all BM rows of NC / Q = 12-16 columns, so every row's FP32 loads (preactivations,
  history, history gradient, partials) and its dG stores touched only 48-64 contiguous bytes.
* *The new layout.* `FRNN_ROWOWN = 1`, now the default, gives CTA (i, j) the BM / Q rows j · BM / Q .. of all NC
  columns of block i: 768-1024 contiguous bytes per row. The elements per CTA, the arithmetic and the partial-sum order
  are unchanged, so dG and d_initial must be bitwise equal to the first version (`rowown=False`).
* *ptxas.* No spill at BM = 64 (168 / 232 registers at D = 1536 / 2048).

**Runs (batch "rowown"):**
* **R1, bitwise against the first version**: D ∈ {1024, 1280, 1536, 2048} × LSTM/sLSTM × FP16/BF16 × B ∈ {40, 64,
  100}, T = 24 (48 cases).
* **R2, race verification** (rule of section 12) of the new layout: D ∈ {1536, 2048} × LSTM/sLSTM at BM = 64, B = 100
  (4 tasks).
* **Paired timing**, training step, B = 64, T = 256, D ∈ {1024, 1536, 2048} × LSTM/sLSTM × FP16/BF16:
  * **W6-rowown**: first version / new;
  * **W4-rowown**: `cuda` / new;
  * protocol and rule of section 3.
* **Prediction.** The pointwise phase shrinks and the step gets 1.05-1.3x faster. Against `cuda`, D = 1536 moves toward
  or past 1.
* **Profile, report only**: in-kernel phases of both layouts at D ∈ {1536, 2048}, LSTM FP16.

## 23. Results of the row-ownership batch (job "rowown", 2026-10-18)

**R1**: 48/48. dG and d_initial are bitwise equal to the first version (D ∈ {1024, 1280, 1536, 2048} × LSTM/sLSTM ×
FP16/BF16 × B ∈ {40, 64, 100}).

**R2**: 4/4. The correct kernel counts 0 violations; no B1 and no B2 are each flagged 21/21, never silently.

**W6-rowown** (old / new) holds 9/12, at 1.01-1.03x. The three misses are D = 2048 LSTM BF16 and D = 1024 sLSTM, whose
intervals include 1. **W4-rowown** (`cuda` / new) holds 7/12:
* D = 1024: 1.10-1.17x.
* D = 2048: 1.13-1.15x.
* D = 1536: 0.91-0.99x, still below.

The prediction (1.05-1.3x) was too high. In this run's profile the old layout's pointwise phase is 5.5 µs, not the
9.6 µs of section 21, so its share was smaller than assumed. The new layout takes it to 4.7 µs at D = 1536 and from
6.7 to 6.2 µs at D = 2048. The partial-sum phase drops from 1.8 to 1.1 µs at D = 1536.

Where a D = 1536 step goes now (LSTM FP16, B = 64, per call):
* step 14.1 ms;
* backward kernel 4.6 ms. Per step that is GEMM 7.6 µs, pointwise 4.7, B1 2.3, B2 1.7, partial sum 1.1;
* preactivation recompute 2.7 ms;
* dR + dbias 1.7 ms;
* training forward about 3.3 ms;
* the rest is copies.

Two next steps follow from this:
1. *Overlapping the dG staging.* The staging load of the next chunk stalls each warp until the data arrives (load,
   then store to shared memory), so it does not overlap the MMAs. An asynchronous copy (`cp.async`) would overlap it.
2. *TF32 for the preactivation GEMM.* Its two operands are input-dtype values (FP16: 10-bit mantissa, BF16: 7-bit),
   so TF32 represents them exactly and their products are exact in TF32 tensor cores, as the
   `preactivations` docstring notes. Only the summation order changes. The current FP32 path runs as a three-pass
   TF32 emulation kernel (`f32f32_tf32f32`).

## 24. Overlapped dG staging and TF32 preactivations (pre-registration, 2026-10-18, before any run of it)

Two changes, each switchable for the A/B:
1. **`FRNN_ASYNC = 1`** (default), `asyncload=False` turns it off. The 2-D kernel stages its dG chunks with `cp.async`,
   so the copy of chunk r + 1 overlaps the MMAs of chunk r. The staged values and the arithmetic are unchanged, so the
   outputs must be bitwise equal to synchronous staging. ptxas: no spill; 156 / 231 registers at D = 1536 / 2048.
2. **`pre_tf32=True`** (opt-in for now, training option `pre_tf32`). The 2-D backward's preactivation GEMM runs on
   TF32 tensor cores. Both operands are input-dtype values and exact in TF32, so only the summation order of the
   K = D products changes. Not bitwise; qualified against the FP64 budgets.

**Runs (batch "tune"):**
* **A1**: async against synchronous staging, bitwise. D ∈ {1024, 1280, 1536, 2048} × LSTM/sLSTM × FP16/BF16 ×
  B ∈ {40, 64, 100}.
* **A2**: race verification of the async kernel (section 12 rule), D ∈ {1536, 2048} × LSTM/sLSTM, BM = 64, B = 100.
* **W1-tf32**: training step within the FP64 budgets with `pre_tf32=True`, D ∈ {1024, 1536, 2048}, B ∈ {64, 40},
  LSTM/sLSTM, FP16/BF16 (24 cases).
* **Paired timing**, training step, B = 64, T = 256, D ∈ {1024, 1536, 2048} × LSTM/sLSTM × FP16/BF16. The new 2-D
  path is async + TF32 preactivations. **W6-tune** compares it against the section 23 kernel (synchronous staging,
  FP32 preactivations); **W4-tune** against `cuda`.
* **Profile, report only**: the four combinations at D ∈ {1536, 2048}, LSTM FP16, step breakdown and in-kernel
  phases.

**Prediction.** The GEMM phase shrinks by 1-3 µs per step. The preactivation drops from 2.7 to about 1 ms at
D = 1536. W4-tune holds at every D, including 1536.

## 25. Results of the "tune" batch (job "tune", 2026-10-18); the next runs (pre-registration)

**Results.**
* **A1**: 48/48. cp.async staging is bitwise equal to synchronous staging.
* **A2**: 4/4. The correct kernel counts 0 violations; no B1 and no B2 are each flagged 21/21, never silently.
* **W1-tf32**: 24/24 within the FP64 budgets.
* **W6-tune**, the section 23 kernel against async staging with TF32 preactivations: holds 8/12, at 1.02-1.04x. The
  four misses have intervals that include 1.
* **W4-tune**, `cuda` against the new path: holds 9/12.
  * D = 1024: 1.11-1.20x.
  * D = 2048: 1.14-1.19x.
  * D = 1536: 0.98 [0.92, 1.02] (LSTM BF16), 1.01 [1.00, 1.02] (LSTM FP16), 1.01 [0.97, 1.04] (sLSTM BF16),
    1.03 [1.02, 1.06] (sLSTM FP16, holds).

At D = 1536 the 2-D path now matches `cuda` within about ±3%; section 21 had it at 0.84-0.99x.

Profile, LSTM FP16, B = 64:
* cp.async cuts the GEMM phase from 7.7 to 6.0 µs per step at D = 1536 and from 11.2 to 9.4 µs at D = 2048.
* The backward kernel goes from 4.62 to 4.20 ms and from 6.06 to 5.49 ms.
* TF32 for the preactivation GEMM changed nothing measurable: 2.69 ms with and without, and the same at D = 2048. So
  the preactivation phase is not bound by that GEMM; the elementwise passes around it are a likely candidate, not
  measured. `pre_tf32` stays off, and the claimed gain comes from cp.async alone, which stays the default.

**Next (batch "tune2").** The 2-D backward with its defaults: section 22's row ownership plus cp.async; FP32
preactivations.
* **W1-1792**: D = 1792 on the new 7 × 16 grid (4 register k-steps; ptxas 226 registers, 215 KB shared memory, no
  spill), training within the FP64 budgets, B ∈ {64, 40} (8 cases).
* **R-1792**: race verification at D = 1792, LSTM/sLSTM, BM = 64, B = 100.
* **W1-small**: the 2-D backward at B ∈ {16, 32}, D ∈ {1536, 2048} (16 cases).
* **Paired timing**:
  * D = 1792, B = 64, four (cell, dtype) shapes. **W6-2d1792** compares against the 1-D kernel (the current default
    there), **W4-2d1792** against `cuda`.
  * B ∈ {16, 32} × D ∈ {1536, 1792, 2048} × (LSTM FP16, sLSTM BF16). **W6-small** compares against the 1-D kernel,
    **W4-small** against `cuda`.
  * Protocol of section 3.
* **Prediction.** D = 1792 behaves like its neighbours: about 1.4-1.6x against 1-D, and about 1.0-1.15x against
  `cuda`. At B ≤ 32 the 2-D kernel is at least as fast as 1-D, since its bytes per step scale with B as well.

## 26. Results of the "tune2" batch (2026-10-18)

Evidence: `max-width-evidence/tune2-analysis.json`, archive verified by SHA-256. Everything ran with the 2-D defaults:
row ownership, cp.async staging, FP32 preactivations.

**W1-1792: 8/8; R-1792: 2/2.** The correct kernel counts 0 violations; no B1 and no B2 are each flagged 21/21.
Difference to the 1-D kernel: dG ≤ 7.5e-6, d_initial ≤ 4.0e-5. **W1-small: 16/16** (B ∈ {16, 32}, D ∈ {1536, 2048}).

**Paired training step:**

| case | vs 1-D backward (W6) | vs `cuda` (W4) |
| --- | --- | --- |
| D = 1792, B = 64, LSTM FP16 | 1.81 [1.62, 1.90] | 0.99 [0.96, 1.02] (no) |
| D = 1792, B = 64, LSTM BF16 | 1.91 | 0.89 [0.87, 0.92] (no) |
| D = 1792, B = 64, sLSTM FP16 | 1.81 | 1.02 [1.01, 1.03] |
| D = 1792, B = 64, sLSTM BF16 | 1.86 | 0.96 [0.94, 0.98] (no) |
| D = 1536, B = 16 / 32, LSTM FP16 | 1.56 / 1.68 | 1.90 / 1.11 |
| D = 1792, B = 16 / 32, LSTM FP16 | 1.75 / 1.89 | 1.67 / 1.09 |
| D = 2048, B = 16 / 32, LSTM FP16 | 1.66 / 1.61 | 2.65 / 1.68 |
| D = 1536, B = 16 / 32, sLSTM BF16 | 1.47 / 1.59 | 1.98 / 1.20 |
| D = 1792, B = 16 / 32, sLSTM BF16 | 1.78 / 1.83 | 2.16 / 1.26 |
| D = 2048, B = 16 / 32, sLSTM BF16 | 1.58 / 1.58 | 2.91 / 1.74 |

* **W6-2d1792** holds 4/4 and **W4-2d1792** 1/4: D = 1792 behaves like D = 1536, about at parity with `cuda`.
* **W6-small** holds 12/12 and **W4-small** 12/12. At B ≤ 32 the 2-D backward is 1.5-1.9x faster than the 1-D kernel,
  and the training step beats `cuda` by 1.09-2.91x.

**Default (after the run).** The training backward now uses the 2-D kernel at D ∈ {1536, 1792, 2048} for every batch
size (`hybrid_bw2d.DEFAULT_WIDTHS`). D = 1280 keeps the Triton backward.

## 27. The 2-D backward at D = 1024 / 1280 against the Triton default (pre-registration, 2026-10-18, before any run of it)

At D ≤ 1280 the training backward is the Triton R-tiled kernel. Section 21's D = 1024 rows compared the 2-D kernel
with the 1-D CUDA kernel, not with that default.

**Runs (batch "tune3"):**
* **W1-low**: the 2-D backward within the FP64 budgets at D = 1280 (B ∈ {64, 40, 16, 32}) and at D = 1024
  (B ∈ {16, 32}), LSTM/sLSTM, FP16/BF16 (24 cases).
* **Paired training step** at D ∈ {1024, 1280} × B ∈ {16, 64} × (LSTM FP16, sLSTM BF16). **W6-low** compares the
  Triton default against the 2-D kernel, **W4-low** `cuda` against the 2-D kernel. Library forward, protocol of
  section 3.

**Rule for the default.** The 2-D kernel becomes the default at a (D, B range) only where W6-low holds in every
measured row at that D and batch size.

## 28. The forward default at 768 < D ≤ 1280 (pre-registration, 2026-10-18, before any run of it)

`execution_policy.default_config` gives D ∈ (768, 1280] the conservative, unmeasured Triton R-tiled kernel (16-column
slices), except at B > 32 for D = 1280, where it gives the wide-tile TileLang kernel. scale-out-plan.md section 24
measured that default at 18.6 µs per step at D = 1024, B = 16, and 47 µs per step at B = 64. The TileLang grid kernel
(16-column slices, 16-row tiles) is qualified at these widths (section 7: W1 20/20) and was timed only against
FlashRNN1, never against this default.

**Runs (batch "fwd1024"):**
* **Q**: forward qualification of the TileLang grid kernel at D ∈ {1024, 1280}, B ∈ {16, 64}, LSTM/sLSTM, FP16
  (8 cases).
* **Paired forward timing**, T = 256: D = 1024 at B ∈ {16, 64} and D = 1280 at B = 16, LSTM/sLSTM, FP16/BF16
  (12 shapes). **W6-fwd** compares the Triton default against the TileLang grid kernel, **W4-fwd** FlashRNN1 `cuda`
  against it. Protocol of section 3.

**Rule.** The TileLang grid kernel becomes the default at a (D, B range) where W6-fwd holds in every measured row there.

**Prediction.** It holds everywhere, at 2-4x.

## 29. Results of the "tune3" batch (2026-10-18)

Evidence: `max-width-evidence/tune3-analysis.json`, archive verified by SHA-256.

**W1-low: 24/24** (D = 1280 at B ∈ {16, 32, 40, 64}, D = 1024 at B ∈ {16, 32}; LSTM/sLSTM, FP16/BF16).

**W6-low** (Triton default / 2-D) holds 8/8. **W4-low** (`cuda` / 2-D) holds 5/8:

| cell | dtype | D | B | vs Triton default | vs `cuda` |
| --- | --- | --- | --- | --- | --- |
| lstm | floa | 1024 | 16 | 1.44 [1.30, 1.52] | 2.12 [2.07, 2.15] |
| lstm | floa | 1024 | 64 | 1.47 [1.41, 1.49] | 1.16 [1.09, 1.21] |
| lstm | floa | 1280 | 16 | 1.31 [1.10, 1.42] | 1.81 [1.79, 1.82] |
| lstm | floa | 1280 | 64 | 2.03 [2.03, 2.04] | 0.99 [0.97, 1.04] (no) |
| slstm | bflo | 1024 | 16 | 1.39 [1.16, 1.52] | 2.11 [2.07, 2.14] |
| slstm | bflo | 1024 | 64 | 1.50 [1.39, 1.60] | 1.05 [0.91, 1.15] (no) |
| slstm | bflo | 1280 | 16 | 1.41 [1.40, 1.41] | 1.90 [1.90, 1.91] |
| slstm | bflo | 1280 | 64 | 2.00 [2.00, 2.01] | 1.00 [0.96, 1.02] (no) |

**Default (by the section 27 rule).** At H = 1 the training backward now uses the 2-D kernel at D ∈ {1024, 1280, 1536,
1792, 2048} for every batch size. H > 1 keeps its earlier paths, since the 2-D kernel is not qualified there. D = 1280
at B = 64 sits at parity with `cuda`. The step there also contains the forward and the FP32 preactivation and weight
gradient passes.

## 30. The 2-D backward at D = 512 / 768 (pre-registration, 2026-10-18, before any run of it)

New grids:
* D = 512: 4 × 16 (64 CTAs);
* D = 768: 6 × 16 (96 CTAs).

R sits in shared memory. ptxas: 136-138 registers at BM = 64, no spill. These are opt-in; at these widths the training
default is the tuned Triton backward (gate-batched at B ≤ 16, section 26 of the SPLITG batch).

**Runs (batch "tune4"):**
* **W1-mid**: FP64 budgets at D ∈ {512, 768} × B ∈ {16, 64} × LSTM/sLSTM × FP16/BF16 (16 cases).
* **R-mid**: race verification at D = 768, LSTM/sLSTM, BM = 64, B = 100.
* **Paired training step** at D ∈ {512, 768} × B ∈ {16, 64} × (LSTM FP16, sLSTM BF16) with the library's forward.
  **W6-mid** compares the current default backward against 2-D, **W4-mid** `cuda` against 2-D.
* **Rule.** As in section 27: the default changes at a (D, B) only where W6-mid holds in every row.

## 31. Results of the forward-default batch "fwd1024" (2026-10-18)

Evidence: `max-width-evidence/fwd1024-analysis.json`, archive verified by SHA-256.

**Q: 8/8.** **W6-fwd** (Triton default / TileLang grid) holds 12/12. **W4-fwd** (`cuda` / TileLang grid) holds 12/12.
Forward, T = 256:

| shape | vs Triton default | vs `cuda` |
| --- | --- | --- |
| D = 1024, B = 16 (four cell/dtype rows) | 4.75-4.86x | 3.34-3.50x |
| D = 1024, B = 64 | 5.40-5.54x | 1.47-1.60x |
| D = 1280, B = 16 | 2.81-2.89x | 2.83-3.42x |

**Default (by the section 28 rule).** For 1024 ≤ D ≤ 1280 (the measured range; 768 < D < 1024 is not measured and
keeps the Triton default), `execution_policy.default_config` now gives the TileLang grid
kernel (16-column slices) when TileLang is installed. The exception stays: B > 32 at D = 1280 keeps the wide-tile
kernel of section 16. Without TileLang the Triton default stays. This also changes the single-GPU baseline of
scale-out-plan.md section 24 at D = 1024: that baseline was the old default.

## 32. The 2-D backward at H > 1 (pre-registration, 2026-10-18, before any run of it)

The 2-D kernel loops over (head, batch tile) groups like the 1-D kernel; per head it runs the H = 1 arithmetic. The
training default uses it at H = 1 only (section 29).

**Runs (batch "heads2d"):**
* **H1-2d**: the heads qualification (FP64 budgets plus per-head bitwise against the H = 1 runs through the test
  hook) with `backend="hybrid2d"` passed to the H-head and the H = 1 calls. D ∈ {1024, 1536, 2048} × H ∈ {2, 3} ×
  LSTM/sLSTM, FP16, B = 16, T = 24 (12 cases).
* **Paired training step**, B = 16, T = 256, D ∈ {1024, 2048} × H ∈ {4, 16} × (LSTM FP16, sLSTM BF16) with the library
  forward. **W6-h2d** compares the current H > 1 backward (Triton at D = 1024, the 1-D CUDA kernel at D = 2048) against
  2-D, **W4-h2d** FlashRNN1 `cuda` against 2-D.
* **Rule.** As in section 27.

## 33. Operator profile of the large-D training step (report only, batch "opprof", 2026-10-18)

`tools/heads_opprof.py` (torch.profiler, one step after two warm-ups) at H = 1, B = 64, T = 256, D ∈ {1024, 1536, 2048},
LSTM FP16 / sLSTM BF16, library defaults, plus D = 1536 with `pre_tf32`. Purpose: attribute the preactivation and
weight-gradient phases (about 30% of the step, section 23) to operators, and find out why TF32 for the preactivation
GEMM changed nothing (section 25). No checks.

## 34. Results of the "tune4" batch (2026-10-18)

Evidence: `max-width-evidence/tune4-analysis.json`, archive verified by SHA-256.

**W1-mid: 16/16. R-mid: 2/2** at D = 768 (0 violations; no B1 / no B2 flagged 21/21).

**W6-mid** (current default / 2-D) holds 1/8: 0.91-1.02x. At D = 512 / 768 the tuned Triton backward (gate-batched,
table-selected) is as fast as the 2-D kernel or faster. Under the section 30 rule the default does not change; the 2-D
grids stay opt-in. **W4-mid** (`cuda` / 2-D) holds 8/8, at 1.14-2.26x.

| cell | dtype | D | B | default / 2-D | `cuda` / 2-D |
| --- | --- | --- | --- | --- | --- |
| lstm | floa | 512 | 16 | 0.91 [0.81, 0.95] | 1.90 |
| lstm | floa | 512 | 64 | 1.00 [0.97, 1.01] | 1.49 |
| lstm | floa | 768 | 16 | 0.99 | 2.24 |
| lstm | floa | 768 | 64 | 0.98 | 1.14 |
| slstm | bflo | 512 | 16 | 0.94 | 1.83 |
| slstm | bflo | 512 | 64 | 0.95 [0.84, 0.99] | 1.52 |
| slstm | bflo | 768 | 16 | 1.02 [1.02, 1.03] | 2.26 |
| slstm | bflo | 768 | 64 | 0.98 | 1.19 |

## 35. Fused input-dtype preactivation GEMM (pre-registration, 2026-10-18, before any run of it)

**What section 33's profile showed** (D = 1536, LSTM FP16, B = 64, one step):
* The training backward's options include `tf32=True` (`training.BW_DEFAULTS`), so the preactivation and dR GEMMs
  already ran on TF32 tensor cores. That is why `pre_tf32` changed nothing (section 25).
* The preactivation costs addmm 1.73 ms + `pre += Wx` 0.66 ms + copies.
* dR costs an mm of 1.32 ms.

**Change.** `pre_fused=True`, opt-in. `preact.preactivations_fused` is one Triton GEMM (128 × 128 × 64 tiles) on
input-dtype tensor cores with FP32 accumulation and FP32 output, with the Wx and bias additions in its epilogue. Both
operands are input-dtype values, so the products are exact; the summation order differs from cuBLAS TF32. CPU
interpreter check at FP16: 1.0e-7 relative to FP64. (The interpreter cannot run BF16.)

**Runs (batch "pref"):**
* **P1, accuracy.** At D ∈ {1024, 1536, 1792, 2048} × B ∈ {16, 64} × FP16/BF16, T = 64: the fused preactivation's
  largest error relative to FP64 must be at most the TF32 path's (+1e-7), and its h_{t-1} must equal the library's.
* **W1-pref.** The training step with `pre_fused=True` within the FP64 budgets at D ∈ {1024, 1536, 2048} × B ∈ {16, 64}
  × LSTM/sLSTM × FP16/BF16 (24 cases).
* **Paired training step.** B = 64, T = 256, D ∈ {1024, 1536, 1792, 2048} × (LSTM FP16, sLSTM BF16). **W6-pref**
  compares the default against `pre_fused`, **W4-pref** `cuda` against `pre_fused`.

**Prediction.** P1 and W1-pref pass. The step gains 1.5-2.5 ms at D = 1536 (10-18%), and W4-pref holds at every D.
**Rule.** `pre_fused` becomes the default at H = 1 where W6-pref holds in every row.

## 36. dG stored in the input dtype, input-dtype dR GEMM, H = 1 (pre-registration, 2026-10-18, before any run of it)

**What remains outside the kernel after section 35.** The 2-D kernel stores dG in FP32 (B · T · 4D · 4 bytes, 403 MB
at D = 1536, B = 64, T = 256). The training wrapper then casts it to the input dtype (another read and a half-size
write). dR is a TF32 GEMM over the FP32 dG (1.32 ms at D = 1536, section 35).

**Change.** `dg_dtype="input"` with `wgrad="bmm_lp"`, opt-in, single GPU, the arrangement H > 1 already uses (heads
§8.5-§8.6).
* Build flag `FRNN_DGLP`: the kernel stores to_elem(dG) — the same rounded value it publishes to the exchange buffer.
  The recurrence never reads the stored dG, so dG, d_initial and the exchange are unchanged.
* dR runs as an input-dtype GEMM with FP32 accumulation on the stored tensor, with no conversion pass.
* dbias becomes an FP32 sum of the rounded dG.

**Runs (batch "dglp"):**
* **B1, bitwise.** `bw2d_rowown_check.py --option dg_dtype` compares two configurations, both using wgrad `bmm_lp`:
  the dG_LP store, and the FP32 store. The stored dG must equal the FP32 dG rounded to the input dtype, and d_initial
  and dR must be bitwise equal. Cases: D ∈ {1024, 1536, 1792, 2048} × B ∈ {16, 64} × (LSTM FP16, sLSTM BF16).
* **R1, race.** The section 12 rule on the dG_LP build at D = 1024 and D = 2048 (LSTM FP16 B = 64, sLSTM BF16 B = 16),
  21 runs per variant.
* **W1-dglp.** The training step with `dg_dtype="input", wgrad="bmm_lp"` must stay within the FP64 budgets: D ∈ {1024,
  1536, 2048} × B ∈ {16, 64} × LSTM/sLSTM × FP16/BF16 (24 cases).
* **Paired training step.** B = 64, T = 256, D ∈ {1024, 1536, 1792, 2048} × (LSTM FP16, sLSTM BF16).
  * **W6-dglp** compares the default against dG_LP.
  * **W4-dglp** compares `cuda` against dG_LP.
  * `pre_fused` is off in both arms, so this measures the change apart from section 35.

**Prediction.** B1, R1 and W1-dglp pass. The step gains 0.6-1.0 ms at D = 1536 (4-7%), and W6-dglp holds at every D.
**Rule.** `dg_dtype="input", wgrad="bmm_lp"` becomes the default at H = 1 on the 2-D path where W6-dglp holds in every
row and B1, R1 and W1-dglp pass.

## 37. Results of the "heads2d" batch (section 32, 2026-10-18)

Evidence: `max-width-evidence/heads2d-analysis.json`, archive verified by SHA-256.

**H1-2d: 12/12.** Every case is within the FP64 budgets and bitwise per head against the H = 1 runs, at D ∈ {1024, 1536,
2048} × H ∈ {2, 3}.

**W6-h2d holds 8/8 and W4-h2d holds 8/8.** All starts were stable. Training step, B = 16, T = 256, point ratio with the
95% interval:

| D | H | cell / dtype | current H > 1 path / 2-D | `cuda` / 2-D |
| --- | --- | --- | --- | --- |
| 1024 | 4 | LSTM FP16 | 1.051 (1.033-1.058) | 1.377 |
| 1024 | 4 | sLSTM BF16 | 1.047 (1.033-1.055) | 1.381 |
| 1024 | 16 | LSTM FP16 | 1.062 (1.061-1.064) | 1.215 |
| 1024 | 16 | sLSTM BF16 | 1.051 (1.049-1.057) | 1.194 |
| 2048 | 4 | LSTM FP16 | 1.454 (1.452-1.455) | 2.451 |
| 2048 | 4 | sLSTM BF16 | 1.473 (1.471-1.474) | 2.503 |
| 2048 | 16 | LSTM FP16 | 1.486 (1.485-1.490) | 2.222 |
| 2048 | 16 | sLSTM BF16 | 1.468 (1.468-1.468) | 3.255 |

The current H > 1 path is Triton at D = 1024 and the 1-D CUDA kernel at D = 2048.

**Default (by the section 27 rule).** At H > 1 the training backward now defaults to the 2-D kernel, but only at the
measured widths and batch: D ∈ {1024, 2048} with B ≤ 16 (`hybrid_bw2d.HEADS_DEFAULT_MAX_BATCH`). D = 1536 was qualified
but not timed. It keeps the earlier path, as do larger batches, until they are measured.

## 38. Results of the "pref" batch (section 35, 2026-10-18)

Evidence: `max-width-evidence/pref-analysis.json` and `pref-p1.json`. Every archived file was checked against its
SHA-256 manifest (216 files).

**P1: 16/16.** In every case the fused preactivation is more accurate against FP64 than the cuBLAS TF32 path, and its
h_{t-1} equals the library's.

| dtype | fused, largest relative error | TF32 path, largest relative error |
| --- | --- | --- |
| FP16 | 1.56e-6 | 2.66e-6 |
| BF16 | 1.31e-6 | 2.63e-6 |

The largest fused / TF32 ratio over the cases is 0.63.

**W1-pref: 24/24.** All cases are within the FP64 budgets.

**W6-pref holds 8/8; W4-pref holds 8/8.** All starts were stable. Training step, B = 64, T = 256, point ratio with the
95% interval:

| D | cell / dtype | default / `pre_fused` | `cuda` / `pre_fused` |
| --- | --- | --- | --- |
| 1024 | LSTM FP16 | 1.150 (1.147-1.152) | 1.396 (1.353-1.422) |
| 1024 | sLSTM BF16 | 1.142 (1.140-1.143) | 1.344 (1.294-1.432) |
| 1536 | LSTM FP16 | 1.142 (1.141-1.144) | 1.222 (1.213-1.230) |
| 1536 | sLSTM BF16 | 1.134 (1.133-1.136) | 1.166 (1.143-1.181) |
| 1792 | LSTM FP16 | 1.139 (1.127-1.143) | 1.138 (1.111-1.178) |
| 1792 | sLSTM BF16 | 1.146 (1.130-1.188) | 1.077 (1.046-1.096) |
| 2048 | LSTM FP16 | 1.143 (1.121-1.205) | 1.296 (1.270-1.313) |
| 2048 | sLSTM BF16 | 1.121 (1.120-1.123) | 1.305 (1.298-1.313) |

The gain is 12-15%, within the 10-18% predicted. With it, D = 1536 / 1792 at B = 64 — previously at parity with `cuda`
(section 26) — is now 1.08-1.22x faster than `cuda`.

**Default (by the section 35 rule).** `pre_fused` is now the default at H = 1 at the timed widths D ∈ {1024, 1536, 1792,
2048}, for B ≥ 64 (`hybrid_bw2d.PRE_FUSED_WIDTHS`, `PRE_FUSED_MIN_BATCH`). It was qualified at B = 16 but timed only at
B = 64, so B < 64 and D = 1280 keep the TF32 preactivation until measured.

## 39. Filling the gaps of sections 37 and 38 (pre-registration, 2026-10-18, before any run of it)

Sections 37 and 38 set defaults only at the timed shapes. This batch ("ext") times the neighbouring shapes. In every
pair the other arm is named explicitly, so the defaults of section 37 and 38 do not leak into it.

**Runs:**
* **W1-pref2.** `pre_fused=True` must stay within the FP64 budgets at D = 1280: B ∈ {16, 64} × LSTM/sLSTM × FP16/BF16
  (8 cases).
* **H1-h2d2.** The heads qualification with `backend="hybrid2d"` at B = 64: D ∈ {1024, 2048}, H = 2, (LSTM FP16, sLSTM
  BF16), T = 24 (4 cases).
* **W6-pref2 and W4-pref2.** H = 1, T = 256, (LSTM FP16, sLSTM BF16). W6-pref2 compares `pre_fused=False` against
  `True`; W4-pref2 compares `cuda` against `pre_fused=True`. Shapes:
  * B = 16 at D ∈ {1024, 1536, 1792, 2048};
  * D = 1280 at B ∈ {16, 64}.
* **W6-h2d2 and W4-h2d2.** T = 256, (LSTM FP16, sLSTM BF16). W6-h2d2 compares the earlier H > 1 path against the 2-D
  backward. The earlier path is named explicitly: `backend="triton"` at D ≤ 1280, `"hybrid"` above. W4-h2d2 compares
  `cuda` against the 2-D backward. Shapes:
  * H = 4, B = 64 at D ∈ {1024, 1536, 2048};
  * D = 1536, B = 16 at H ∈ {4, 16}.

**Prediction.** All qualifications pass.
* W6-pref2 holds at B = 64, D = 1280. At B = 16 the gain is smaller (the GEMM has a quarter of the rows) but holds.
* W6-h2d2 holds in every row, by about 1.05x at D = 1024 and 1.3-1.5x at D ≥ 1536.

**Rules.**
* `pre_fused` extends to every (D, B) row where W6-pref2 holds. At a D where both B = 16 and B = 64 hold, it applies at
  every B.
* At H > 1 the 2-D default extends, at each D, up to the largest timed B for which every timed row at that D with
  B' ≤ B holds.

## 40. Results of the "dglp" batch (section 36), and the h-only history gradient (pre-registration, 2026-10-18)

### 40.1 Results of "dglp"

Evidence: `max-width-evidence/dglp-analysis.json` and `dglp-b1-r1.json`. Every archived file was checked against its
SHA-256 manifest (224 files).

**W1-dglp: 24/24.** All cases are within the FP64 budgets.

**B1 and R1: incomplete (a plan defect).** The tool tasks ran without TileLang. Above D = 1280 the forward that builds
their history needs TileLang, so every row at D ≥ 1536 stopped with `ModuleNotFoundError: tilelang`. The runs at
D = 1024 pass:
* B1 4/4: the stored dG equals the FP32 dG rounded to the input dtype, and d_initial and dR are bitwise equal. The
  dbias relative difference is 1.1e-4 to 1.1e-3, from summing the rounded dG.
* R1 2/2: the correct kernel has 0 violations in 21 runs; no_b1 and no_b2 are each flagged 21/21 with none silent.

**W6-dglp holds 6/8; W4-dglp holds 8/8.** Training step, B = 64, T = 256, point ratio with the 95% interval:

| D | cell / dtype | default / dG_LP | `cuda` / dG_LP |
| --- | --- | --- | --- |
| 1024 | LSTM FP16 | 1.172 (0.954-1.654), not held | 1.237 |
| 1024 | sLSTM BF16 | 1.059 (0.962-1.102), not held | 1.294 |
| 1536 | LSTM FP16 | 1.122 (1.120-1.124) | 1.150 |
| 1536 | sLSTM BF16 | 1.111 (1.110-1.112) | 1.113 |
| 1792 | LSTM FP16 | 1.128 (1.118-1.132) | 1.128 |
| 1792 | sLSTM BF16 | 1.136 (1.132-1.147) | 1.063 |
| 2048 | LSTM FP16 | 1.123 (1.121-1.124) | 1.325 |
| 2048 | sLSTM BF16 | 1.122 (1.121-1.123) | 1.319 |

The two D = 1024 rows fail on a few outlier blocks, not on the typical block:
* the per-start median ratio is 1.10 in all six starts;
* single blocks reach 9.3 and 0.44, where one arm's mean took a 9x spike.

The pre-registered rule uses the bootstrap of block means, so these rows count as not held. The measured gain (11-14%
at D ≥ 1536) is larger than predicted (4-7%).

**Default: none yet.** The section 36 rule also needs B1 and R1, which do not cover D ≥ 1536. Section 40.2 reruns them.

### 40.2 Pre-registration: h-only history gradient in the 2-D kernel, plus the dG_LP gaps (before any run of it)

**Change.** Option `honly=True`, opt-in. It applies to the 2-D path when the output is hidden and d_output is in the
input dtype.
* The kernel (build flag `FRNN_HONLY`) reads dh from d_output directly ([B, T, H, D], input dtype). It never reads the
  zero c (and n) history gradients.
* The training wrapper no longer builds the zero-filled [S, B, T, H, D] FP32 history gradient and its add, which is
  800 MB for sLSTM at D = 1536, B = 64.
* The FP32 values the kernel adds are those of the old tensor (exact conversion), so outputs are bitwise equal up to
  the sign of zero.

**Runs (batch "honly"):**
* **B1-dglp and R1-dglp, completed.** Section 36's B1 at D ∈ {1536, 1792, 2048} × B ∈ {16, 64} × (LSTM FP16, sLSTM
  BF16), and R1 at D = 2048 (LSTM FP16 B = 64, sLSTM BF16 B = 16), now with TileLang.
* **W6-dglp at D = 1024, rerun.** The same two claims, same arms (`pre_fused=False` in both). The rerun replaces the
  D = 1024 rows of 40.1.
* **B2, bitwise.** `bw2d_rowown_check.py --option honly` compares d_hidden against the zero-filled FP32 tensor built
  from the same rounded values: dG, d_initial, dR and dbias must be bitwise equal. D ∈ {1024, 1536, 1792, 2048} × B ∈
  {16, 64} × (LSTM FP16, sLSTM BF16).
* **R2, race.** The section 12 rule on the FRNN_HONLY build: D ∈ {1024, 2048} × (LSTM FP16 B = 64, sLSTM BF16 B = 16).
* **W1-honly.** The training step with `honly=True` within the FP64 budgets: D ∈ {1024, 1536, 2048} × B ∈ {16, 64} ×
  LSTM/sLSTM × FP16/BF16 (24 cases).
* **W6-honly and W4-honly.** B = 64, T = 256, D ∈ {1024, 1536, 1792, 2048} × (LSTM FP16, sLSTM BF16). Both arms set
  `pre_fused=True` explicitly. W6-honly compares `honly=False` against `True`; W4-honly compares `cuda` against
  `honly=True`.

**Prediction.** Every check passes. The step gains 1-3% for LSTM and 3-6% for sLSTM.

**Rules.**
* dG_LP: section 36's rule over the union of 40.1 and this batch, with the D = 1024 rows taken from the rerun.
* honly becomes the default for hidden output at H = 1 on the 2-D path where W6-honly holds in every row and B2, R2
  and W1-honly pass.

## 41. Column-group flags in place of grid barrier B1 (pre-registration, 2026-10-18, before any run of it)

**Why.** The advisor named SM synchronization as the bottleneck and suggested manual semaphores. In the 2-D kernel each
step has two grid barriers. B1 (partials complete) cost 2.3 µs per step at D = 1536 (section 23), yet each owner needs
the partials of only its own column block's Q = 16 CTAs.

**Change.** `flags=True`, opt-in, build flag `FRNN_FLAGS`.
* After its partial, each CTA writes a step flag with release semantics at GPU scope.
* The owner's first Q threads spin with acquire loads on the Q flags of its column block, then the CTA synchronizes.
* B2 stays a grid barrier, and so does the last step's B1.

The hazards B1 used to cover:
* A partial written in the next step is still separated from the sum by the next B2.
* The exchange buffer is double buffered by step parity. The pointwise update of step s + 1 writes the buffer last read
  by the GEMMs of step s - 1, and all of those precede B2 of step s.

The arithmetic and summation order are unchanged, so the outputs must be bitwise equal. ptxas reports no spill (158
registers at D = 1536, 233 at D = 2048).

**Runs (batch "flags"):**
* **F1, bitwise.** `bw2d_rowown_check.py --option flags`: dG and d_initial with flags against without. D ∈ {1024,
  1280, 1536, 1792, 2048} × B ∈ {16, 64, 100} × (LSTM FP16, sLSTM BF16). B = 100 runs two slots, which crosses the
  slot boundary.
* **F2, race.** The section 12 rule on the flags build. Here no_b1 drops the flag wait. Cases:
  * D ∈ {1536, 2048} × (LSTM FP16 B = 64, sLSTM BF16 B = 16);
  * D = 1024, LSTM FP16, B = 100.
* **W1-flags.** The training step with `flags=True` within the FP64 budgets: D ∈ {1024, 1536, 2048} × B ∈ {16, 64} ×
  (LSTM FP16, sLSTM BF16) (12 cases).
* **W6-flags and W4-flags.** T = 256, D ∈ {1024, 1536, 1792, 2048} × B ∈ {16, 64} × (LSTM FP16, sLSTM BF16). The arms
  differ only in `flags`; everything else is the library default. W6-flags compares `flags=False` against `True`;
  W4-flags compares `cuda` against `flags=True`.

**Prediction.** F1, F2 and W1-flags pass. The B1 wait drops to about 1 µs per step. The step gains 2-5%, more at
D = 1024 and at B = 16, where the barriers are a larger share.
**Rule.** `flags` becomes the default at H = 1 on the 2-D path in the (D, B) rows where W6-flags holds, provided F1, F2
and W1-flags pass.

## 42. Results of the "ext" (section 39) and "honly" (section 40.2) batches (2026-10-18)

Evidence: `max-width-evidence/ext-analysis.json`, `honly-analysis.json` and `honly-bitwise-race.json`. Every archived
file was checked against its SHA-256 manifest (340 and 260 files).

### 42.1 ext

**W1-pref2 8/8, H1-h2d2 4/4.**

**W6-h2d2 holds 10/10.** Earlier H > 1 path / 2-D:

| shape | ratio |
| --- | --- |
| H = 4, B = 64, D = 1024 | 1.50 |
| H = 4, B = 64, D = 1536 | 1.71-1.73 |
| H = 4, B = 64, D = 2048 | 1.48-1.52 |
| D = 1536, B = 16, H ∈ {4, 16} | 1.49-1.52 |

**W4-h2d2 holds 5/10.** At H = 4, B = 64, FlashRNN1 `cuda` is faster than the 2-D path at D = 1024 (0.78-0.84x) and at
D = 2048 (0.95-0.98x). D = 1536 is at parity (0.99-1.03x). At B = 16 the 2-D path is 1.88-2.27x faster than `cuda`.

**W6-pref2 holds 9/12.** At B = 16, `pre_fused` gains 6-10%:
* D ∈ {1024, 1792, 2048}: both cells hold (1.08-1.10x);
* D = 1536: LSTM 1.080, sLSTM not held (0.961, interval 0.727-1.080);
* D = 1280: B = 16 LSTM 1.061, sLSTM not held; B = 64 LSTM 1.127, sLSTM not held (1.079, 0.973-1.122).

**W4-pref2 holds 11/12.**

### 42.2 honly

**dG_LP completion.** B1 at D ≥ 1536: 12/12 bitwise (stored dG = rounded FP32 dG; d_initial and dR equal). R1 at
D = 2048: 2/2. The D = 1024 rerun of W6-dglp holds 0/2 again: 1.042 (0.939-1.105) and 1.064 (0.972-1.099).

**B2 16/16, bitwise.** dG, d_initial, dR and dbias are equal with and without honly. **R2: 4/4.** **W1-honly: 24/24.**

**W6-honly holds 7/8**, 1.03-1.07x. D = 1792 sLSTM does not hold (1.017, 0.931-1.051). **W4-honly holds 8/8**, 1.11-1.45x.

### 42.3 The rows that did not hold, and a harness defect

Every W6 row that missed in sections 40.1 and 42 has per-start median ratios of 1.04-1.12. Each was lost to a single
slow block.

The batches ext, dglp, pref and heads2d have 20 blocks slower than 1.5x their start's median. Of these, 19 are block
0, the first timed block. Before that block, `heads_bench.py paired` had warmed up each arm with a single call.

This is a harness defect. From the next batch on, `--warmup-blocks 1` runs one untimed block of every arm before the
timed blocks. The rules above are applied as pre-registered, with no post hoc trimming.

### 42.4 Defaults (by the rules of sections 36, 39 and 40.2)

**H > 1.** The 2-D backward is the default up to B = 64 at D ∈ {1024, 1536, 2048}
(`hybrid_bw2d.HEADS_DEFAULT_MAX_BATCH`). At H = 4, B = 64 it beats the earlier path by 1.5-1.7x, but FlashRNN1
`cuda` stays faster at D = 1024 and D = 2048.

**`pre_fused`.** Applies at every B for D ∈ {1024, 1792, 2048}, and for B ≥ 64 at D = 1536 (`PRE_FUSED_MIN_BATCH`).
D = 1280 keeps the TF32 preactivation.

**dG_LP.** The input-dtype dG store with input-dtype dR (`dg_dtype="input", wgrad="bmm_lp"`) is the default for B ≥ 64
at D ∈ {1536, 1792, 2048} (`DGLP_MIN_BATCH`). Every row there held, and B1, R1 and W1 pass. D = 1024 does not get it.

**honly.** The default for hidden output for B ≥ 64 at D ∈ {1024, 1536, 2048} (`HONLY_MIN_BATCH`).

**Untested combination.** These defaults were measured one at a time. The integration batch of section 43 checks
them together.

## 43. Integration check of the 2026-10-18 defaults, and reruns (pre-registration, 2026-10-18, before any run of it)

Batch "final". It uses the paired harness with the warm-up block of section 42.3. Arms without options are the
library's own selection, forward included.

**Runs:**
* **D2, qualification of the combined defaults.** No options at all, within the FP64 budgets:
  * H = 1: D ∈ {1024, 1280, 1536, 1792, 2048} × LSTM/sLSTM × FP16/BF16 × B ∈ {16, 64} (40 cases);
  * H = 2: D ∈ {1024, 1536, 2048} × (LSTM FP16, sLSTM BF16) × B ∈ {16, 64} (12 cases).
* **W6-final and W4-final.** H = 1, T = 256, D ∈ {1024, 1280, 1536, 1792, 2048} × B ∈ {16, 64} × (LSTM FP16, sLSTM
  BF16), 20 shapes.
  * The new arm is the library default.
  * The old arm is the 2-D backward with today's later options all off: `pre_fused=False, honly=False, wgrad="bmm"`.
  * W6-final compares old against new; W4-final compares `cuda` against new.
* **Reruns of the rows lost in sections 40.1 and 42**, with the warm-up block. Both cells (LSTM FP16, sLSTM BF16) at
  each (D, B), with the original arms and every other later option pinned off:
  * W6-dglp3: D = 1024, B = 64;
  * W6-honly3: D = 1792, B = 64;
  * W6-pref3: D = 1536, B = 16; D = 1280, B = 16; D = 1280, B = 64.

**Prediction.** D2 passes 52/52. W6-final holds in every row where at least one later default applies, at 1.15-1.35x
at B = 64, D ≥ 1536. W4-final holds in every row. The reruns hold.

**Rules.**
* A D2 failure removes the defaults involved at that shape.
* A rerun applies the rule of its original section to its (D, B): both cells must hold.
* W6-final and W4-final are reported; they change nothing.

## 44. Results of the "flags" batch (section 41, 2026-10-18): correct, but slower

Evidence: `max-width-evidence/flags-analysis.json` and `flags-f1-f2.json`. Every archived file was checked against its
SHA-256 manifest (286 files).

**Correctness passes:**
* **F1, 30/30.** dG and d_initial are bitwise equal with and without flags, across D = 1024-2048 and B ∈ {16, 64, 100}.
  The B = 100 cases cross the slot boundary.
* **F2, 5/5.** The correct kernel has 0 violations. Without the flag wait, B1 is flagged in 21/21 runs in every task;
  no run changed silently.
* **W1-flags, 12/12.**

**W6-flags holds 0/16.** The flags build is slower in every row:

| B | grid barrier / flags |
| --- | --- |
| 16 | 0.91-0.98 |
| 64 | 0.95-0.99 |

The largest loss is 9%, for sLSTM at D = 1024, B = 16. W4-flags still holds 15/16, because the 2-D path's lead over
`cuda` absorbs the loss.

**Reading (not measured).** The design saves the wait on 112-128 CTAs, but it adds per-step costs to every CTA:
* a `__syncthreads`;
* a `__threadfence`;
* a release store after the partial;
* acquire spins by Q threads, then a second `__syncthreads`.

The next B2 is still a full grid barrier, so a step still ends with every CTA in sync. Whatever B1's wait cost, these
added costs exceed it. A cheaper variant would drop the `__threadfence`, which the release store may already cover, and
time the B1 phase with the in-kernel phase counters before the step. Neither was done here.

**Default: none.** `flags` stays opt-in, as a verified negative result.

## 45. Where the step goes now, and why flags lost (report only, batch "prof2", 2026-10-18, before any run of it)

No checks and no rules.
* **Kernel phases** (`heads_bench.py` sweep with profile variants, five calls each, median): LSTM FP16 at D ∈ {1024,
  1536, 2048} × B ∈ {16, 64}, plus sLSTM BF16 at D = 1536, B = 64. Each case runs four variants:
  * the library default (CUDA-event phases: preactivation / kernel / weight gradients);
  * the default with the in-kernel phase timers (pointwise, B2, GEMM, B1, partial sum, in µs per step);
  * `flags=True` with the timers;
  * today's later options off (`pre_fused=False, honly=False, wgrad="bmm"`) with the timers.
* **Operator profile** (`heads_opprof.py`, torch.profiler, library default): D = 1536, B = 64, LSTM FP16 and sLSTM
  BF16. It is the counterpart of section 33's table after today's defaults.

## 46. Results of "prof2" (section 45, report only, 2026-10-18)

Evidence: `max-width-evidence/prof2.json`. Every archived file was checked against its SHA-256 manifest (65 files).
Medians of five calls. Step times are one training step: forward + backward, `heads_bench` loss included.

**Where the step goes with the library defaults** (H = 1, T = 256), in ms:

| shape | step | step, later options off | training forward (full history) | preactivation | 2-D kernel | weight gradients |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM FP16, D = 1024, B = 16 | 3.53 | 3.75 | 1.18 | 0.18 | 1.68 | 0.25 |
| LSTM FP16, D = 1024, B = 64 | 7.24 | 8.70 | 2.34 | 0.56 | 2.75 | 0.83 |
| LSTM FP16, D = 1536, B = 16 | 5.15 | 5.19 | 1.35 | 0.72 | 2.17 | 0.53 |
| LSTM FP16, D = 1536, B = 64 | 10.05 | 13.60 | 3.36 | 1.01 | 4.05 | 0.74 |
| sLSTM BF16, D = 1536, B = 64 | 10.43 | 14.17 | 3.60 | 1.01 | 4.24 | 0.72 |
| LSTM FP16, D = 2048, B = 16 | 6.00 | 6.53 | 1.53 | 0.53 | 2.62 | 0.74 |
| LSTM FP16, D = 2048, B = 64 | 16.36 | 21.46 | 7.22 | 1.61 | 5.34 | 1.04 |

"Later options off" means `pre_fused=False, honly=False, wgrad="bmm"`. At B = 64 these options together make the step
1.20x (D = 1024), 1.35-1.36x (D = 1536) and 1.31x (D = 2048) faster. At D = 1536, B = 16 none of them is a default,
and the two columns agree.

The operator profile at D = 1536, B = 64 (LSTM FP16) adds up as follows. The two recurrent kernels take 71% of the
step: backward 4.05 ms, forward 3.10 ms. The rest:
* the fused preactivation GEMM, 0.80 ms;
* the input-dtype dR GEMM, 0.52 ms (it was 1.32 ms in TF32, section 33);
* copies, 0.34 ms;
* the benchmark's loss multiply, 0.33 ms;
* small elementwise kernels and the dbias sum, about 0.6 ms.

**2-D kernel phases** (µs per step, default, LSTM FP16):

| D, B | pointwise | B2 | GEMM | B1 | partial sum |
| --- | --- | --- | --- | --- | --- |
| 1024, 16 | 1.42 | 1.29 | 1.76 | 1.37 | 0.28 |
| 1024, 64 | 2.98 | 1.52 | 3.29 | 1.90 | 0.74 |
| 1536, 16 | 1.49 | 1.34 | 3.02 | 1.91 | 0.31 |
| 1536, 64 | 4.35 | 1.60 | 5.89 | 2.41 | 1.11 |
| 2048, 16 | 1.50 | 1.44 | 4.49 | 1.87 | 0.36 |
| 2048, 64 | 5.82 | 1.77 | 9.26 | 1.93 | 1.42 |

The two barriers take 3-4.3 µs per step: 40% of the kernel at B = 16, 21-26% at B = 64.

**Why flags lost (section 44).** With `flags=True` the B1 wait falls to 0.70-0.85 µs per step, as intended. The GEMM
phase grows by 1.2-2.1 µs, because the phase now ends with the partials' publication (`__syncthreads`,
`__threadfence`, release store). The time moved from the barrier to making the partials visible at GPU scope; it did
not disappear. Any B1 replacement that keeps the partials in global memory pays this.

## 47. Results of the "final" batch (section 43, 2026-10-18)

Evidence: `max-width-evidence/final-analysis.json` and `final-d2.json`. Every archived file was checked against its
SHA-256 manifest (556 files).

### D2: 52/52 by its pre-registered criterion

Section 43 defines D2 as "within the FP64 budgets". Every case meets it:
* the 40 H = 1 cases;
* the 12 H = 2 cases, forward and training.

The qualification tool also applies its own per-head bitwise check: the H-head training call against H = 1 calls on
each head. Its overall flag therefore fails 10 of the 12 H = 2 cases. D2 is scored here by the section 43 wording, not
by that flag. The flag still exposes a real change in behaviour:
* *Why it fails.* The H = 1 calls now take the H = 1-only defaults (`pre_fused`, dG_LP, `honly`); the H-head path
  never uses them. The only two H = 2 cases that pass are D = 1536, B = 16, the one H = 2 shape where none of those
  defaults applies at H = 1 (under that batch's tables).
* *What changed.* With library defaults, an H-head training call is no longer bitwise equal, head by head, to H = 1
  calls wherever an H = 1-only default applies. Both stay within the FP64 budgets.
* *Where it was checked before.* This document qualified the per-head bitwise property with matched options in both
  calls: heads §8.1 through the test hook, and here §37 (H1-2d) and §42 (H1-h2d2). The library defaults never
  promised it across different H.
* The forward is unaffected: it is bitwise per head in all 12 cases.

### Reruns with the warm-up block: 10/10 rows hold

| rerun | shape | ratios (point, 95% interval) |
| --- | --- | --- |
| W6-dglp3 | D = 1024, B = 64 | LSTM 1.192 (1.101-1.454), sLSTM 1.098 (1.097-1.100) |
| W6-honly3 | D = 1792, B = 64 | LSTM 1.042, sLSTM 1.083 (1.047-1.188) |
| W6-pref3 | D = 1536, B = 16 | 1.079 and 1.130 |
| W6-pref3 | D = 1280, B = 16 | 1.072 and 1.066 |
| W6-pref3 | D = 1280, B = 64 | 1.128 and 1.122 |

Every W4 rerun row holds as well.

### W6-final and W4-final (report only): 12 of 20 shapes measured

Today's later options together (library default / options off) give:

| D | B = 16 | B = 64 |
| --- | --- | --- |
| 1024 | 1.08, 1.15 | 1.23, 1.23 |
| 1280 | 1.00-1.03 (not held) | 1.00-1.02 (not held) |
| 1536 | 1.00-1.07 | lost |
| 1792 | lost | 1.31 |

The D = 1280 rows were measured under that batch's tables, where no later option applied at D = 1280. Those rows
therefore compare two identical arms.

Against FlashRNN1 `cuda`, the library default is:
* at B = 16: 1.64-2.30x (D ≤ 1536);
* at B = 64: 1.42-1.45x (D = 1024) and 1.28-1.32x (D = 1792);
* at D = 1280, B = 64: 0.95-0.98x (`cuda` ahead).

**Lost: 8 shapes** (D = 1536 B = 64, D = 1792 B = 16, D = 2048 at both B). There are two defects:
* *Plan defect.* The claims named no forward configuration, so their FlashRNN1 prebuild had to call the library's
  selection, which needs TileLang. The prebuild tasks ran without TileLang and failed. The claim processes then built
  FlashRNN1 in the TileLang environment, where nvcc 12.8 does not match PyTorch's CUDA 11.8.
* *Harness defect.* An arm that fails its first call stayed in the timing loop. The whole start was then marked ERROR,
  which lost the W6 arm too. Fixed: a failed arm is now dropped (`heads_bench.py`).

### Defaults (by the rules of sections 39, 40.2 and 43)

* `pre_fused`: every B at D = 1280 and D = 1536 (both now held at B = 16 and B = 64). `PRE_FUSED_MIN_BATCH` now covers
  every B at D = 1024-2048.
* dG_LP: D = 1024 is added for B ≥ 64. Its B1 and R1 (section 40.1) and W1-dglp passed.
* `honly`: D = 1792 is added for B ≥ 64.
* The H > 1 2-D default stays as in section 42.4.

D2 ran before these additions. The new combinations, all three options together at D = 1024 and D = 1792 for B = 64,
and `pre_fused` at D = 1280 / 1536 for B < 64, are qualified in section 48.

## 48. Qualification of the section 47 additions, and the lost integration rows (pre-registration, 2026-10-18)

Batch "final2". It uses the paired harness with the warm-up block and the failed-arm fix. The forward configuration is
pinned to the one used in sections 35-43 at each shape, so the FlashRNN1 prebuild needs no TileLang.

**Runs:**
* **D3, qualification of the new combinations.** No options, within the FP64 budgets:
  * H = 1, D ∈ {1024, 1792}, B = 64: all three later options together;
  * H = 1, D ∈ {1280, 1536}, B = 16: `pre_fused` newly on;
  * each × LSTM/sLSTM × FP16/BF16 (16 cases).
* **D3-H2, per-head bitwise with matched options.** The heads qualification at H = 2 with `pre_fused=False,
  honly=False, wgrad="bmm"` in both the H-head and the H = 1 calls. D ∈ {1024, 2048} × B ∈ {16, 64} × (LSTM FP16, sLSTM
  BF16), 8 cases. This confirms that the H > 1 path is still bitwise per head.
* **W6-final2 and W4-final2 (report only).** The arms of section 43; only the forward configuration is now pinned.
  T = 256, (LSTM FP16, sLSTM BF16), at:
  * the eight lost shapes: D = 1536 B = 64; D = 1792 B = 16; D = 2048 B ∈ {16, 64};
  * D = 1280, B ∈ {16, 64}, and D ∈ {1024, 1792}, B = 64, whose defaults changed in section 47.

**Prediction.** D3 16/16, D3-H2 8/8 (FP64 and per-head bitwise).
**Rule.** A D3 failure removes the section 47 additions at that shape. W6-final2 and W4-final2 change nothing.

## 49. Results of the "final2" batch (section 48, 2026-10-18)

Evidence: `max-width-evidence/final2-analysis.json`. Every archived file was checked against its SHA-256 manifest (304
files). All 48 claim starts ran.

**D3: 16/16.** The section 47 additions are within the FP64 budgets with library defaults:
* all three later options together at D ∈ {1024, 1792}, B = 64;
* `pre_fused` at D ∈ {1280, 1536}, B = 16.

The defaults stand.

**D3-H2: 8/8, FP64 and per-head bitwise.** With matched options (`pre_fused=False, honly=False, wgrad="bmm"`) the H > 1
2-D path is bitwise equal per head to H = 1 calls. Section 47's failures were therefore only the difference in options
between H = 1 and H > 1.

**W6-final2 holds 16/16 and W4-final2 16/16 (report only).** Training step, T = 256, library default against:
* "later options off" (`pre_fused=False, honly=False, wgrad="bmm"`, same 2-D kernel and forward);
* FlashRNN1 `cuda`.

| D | B | cell / dtype | options off / default | `cuda` / default |
| --- | --- | --- | --- | --- |
| 1024 | 64 | LSTM FP16 | 1.363 | 1.607 |
| 1024 | 64 | sLSTM BF16 | 1.555 (1.367-1.915) | 1.601 |
| 1280 | 16 | LSTM FP16 | 1.070 | 1.909 |
| 1280 | 16 | sLSTM BF16 | 1.069 | 2.041 |
| 1280 | 64 | LSTM FP16 | 1.142 | 1.150 |
| 1280 | 64 | sLSTM BF16 | 1.129 | 1.106 |
| 1536 | 64 | LSTM FP16 | 1.374 | 1.410 |
| 1536 | 64 | sLSTM BF16 | 1.373 | 1.353 |
| 1792 | 16 | LSTM FP16 | 1.137 | 1.869 |
| 1792 | 16 | sLSTM BF16 | 1.083 | 2.386 |
| 1792 | 64 | LSTM FP16 | 1.460 (1.379-1.699) | 1.355 |
| 1792 | 64 | sLSTM BF16 | 1.377 | 1.367 |
| 2048 | 16 | LSTM FP16 | 1.099 | 2.879 |
| 2048 | 16 | sLSTM BF16 | 1.130 | 3.069 |
| 2048 | 64 | LSTM FP16 | 1.357 | 1.571 |
| 2048 | 64 | sLSTM BF16 | 1.345 | 1.576 |

Together with section 47's measured rows, the library default is now faster than FlashRNN1 `cuda` at every measured
H = 1 shape from D = 1024 to 2048:
* 1.11-1.61x at B = 64;
* 1.64-3.07x at B = 16.

D = 1280, B = 64 was 0.95-0.98x in section 47, before `pre_fused` applied there; it is now 1.11-1.15x. The later
options alone make the step 1.13-1.56x faster at B = 64.

## 50. dG_LP and honly at B = 16 and at D = 1280 (pre-registration, 2026-10-18, before any run of it)

Both options were timed only at B = 64 and only at D ∈ {1024, 1536, 1792, 2048}. Batch "gaps2" uses the paired harness
with the warm-up block. In each pair the arms differ in that one option only; everything else is the library default.
* dG_LP: `dg_dtype="input", wgrad="bmm_lp"` against `wgrad="bmm"`.
* honly: `honly=True` against `False`.

**Runs:**
* **W6-dglp4 and W6-honly4, with W4 against `cuda`.** T = 256, (LSTM FP16, sLSTM BF16), at:
  * B = 16, D ∈ {1024, 1280, 1536, 1792, 2048};
  * B = 64, D = 1280.
* **D = 1280 checks**, both at B ∈ {16, 64} × (LSTM FP16, sLSTM BF16):
  * B1 (dG_LP bitwise, section 36) and B2 (honly bitwise, section 40.2);
  * R1 and R2 races (LSTM FP16 B = 64, sLSTM BF16 B = 16).
* **W1-gaps.** Library defaults with both options forced on, within the FP64 budgets: B = 16 at D ∈ {1024, 1280, 1536,
  2048}, and B = 64 at D = 1280. Each case × (LSTM FP16, sLSTM BF16), 10 cases.

**Prediction.** All checks pass. At B = 16 the gains are smaller than at B = 64: dG_LP 3-8%, honly 1-3%.
**Rule.** For each option, the tables (`DGLP_MIN_BATCH`, `HONLY_MIN_BATCH`) extend to every (D, B) row where its W6
holds for both cells, provided the D = 1280 checks (for D = 1280) and W1-gaps pass. Where B = 16 and B = 64 both hold,
the option applies at every B.

## 51. Results of the "gaps2" batch (section 50, 2026-10-18)

Evidence: `max-width-evidence/gaps2-analysis.json` and `gaps2-bitwise-race.json`. Every archived file was checked
against its SHA-256 manifest (370 files). All 72 claim starts ran.

**D = 1280 checks pass.**
* B1: 4/4. The stored dG is the rounded FP32 dG; d_initial and dR are equal.
* B2: 4/4. dG, d_initial, dR and dbias are equal.
* R1 2/2 and R2 2/2. The correct kernel has 0 violations; no_b1 and no_b2 are each flagged 21/21, never silently.

**W1-gaps: 10/10.** Both options forced on, with the other library defaults, stay within the FP64 budgets.

**W6-dglp4 holds 12/12; W6-honly4 holds 12/12.** W4 against `cuda` holds 24/24. Point ratios, option off / on:

| D | B | dG_LP (LSTM, sLSTM) | honly (LSTM, sLSTM) |
| --- | --- | --- | --- |
| 1024 | 16 | 1.173, 1.129 | 1.125, 1.064 |
| 1280 | 16 | 1.085, 1.131 | 1.026, 1.034 |
| 1536 | 16 | 1.102, 1.098 | 1.130, 1.071 |
| 1792 | 16 | 1.131, 1.092 | 1.032, 1.047 |
| 2048 | 16 | 1.092, 1.104 | 1.033, 1.055 |
| 1280 | 64 | 1.161, 1.100 | 1.040, 1.062 |

At B = 16 both options gained more than predicted: dG_LP 9-17% (predicted 3-8%), honly 3-13% (predicted 1-3%).

**Defaults (by the section 50 rule).** B = 16 and B = 64 now hold for both options at every D from 1024 to 2048, so
dG_LP and honly apply at every B there (`DGLP_MIN_BATCH`, `HONLY_MIN_BATCH`). With `pre_fused`, all three later options
are now the H = 1 default on the whole 2-D range, D = 1024-2048.

## 52. Qualification of the section 51 defaults (pre-registration, 2026-10-18, before any run of it)

Batch "final3", qualification only. No options; the library's selection, forward included. Within the FP64 budgets:
* H = 1: D ∈ {1024, 1280, 1536, 1792, 2048} × LSTM/sLSTM × FP16/BF16 × B ∈ {16, 40} (40 cases). At B = 40 the 64-row
  tile is only partly filled.
* H = 2: D ∈ {1024, 2048} × (LSTM FP16, sLSTM BF16) × B ∈ {16, 64} (8 cases), FP64 only (section 47).

**Rule.** A failure removes the section 51 additions (dG_LP and honly at B < 64, and both at D = 1280) at that shape.

## 53. Results of the "final3" batch (section 52, 2026-10-18)

Evidence: `max-width-evidence/final3-d4.json`. Every archived file was checked against its SHA-256 manifest (184
files).

**D4: 48/48 within the FP64 budgets.** That is the 40 H = 1 cases (D = 1024-2048, both cells and dtypes, B ∈ {16, 40})
and the 8 H = 2 cases. The section 51 defaults stand.

As expected from section 47, the tool's per-head bitwise flag fails all 8 H = 2 cases. The H = 1 options now apply at
every B, so no H = 2 shape is left where H = 1 and H > 1 use the same options. The forward is bitwise per head in every
case.

## 54. The new defaults at B > 64 (pre-registration, 2026-10-18, before any run of it)

Every check of `pre_fused`, dG_LP and `honly` ran at B ≤ 64, and B = 40 is still a single 64-row tile. At B > 64 the
2-D kernel runs several tiles as slots, a path the FRNN_DGLP and FRNN_HONLY stores and loads have not run through.

**Batch "final4"** (qualification only): library defaults, H = 1, D ∈ {1024, 1536, 2048} × (LSTM FP16, sLSTM BF16) ×
B ∈ {100, 128}, T = 24 (12 cases), within the FP64 budgets.
**Rule.** A failure caps that D's entries in `PRE_FUSED_MIN_BATCH`, `DGLP_MIN_BATCH` and `HONLY_MIN_BATCH` to B ≤ 64.

## 55. Results of the "final4" batch (section 54, 2026-10-18)

Evidence: `max-width-evidence/final4-d5.json`. Every archived file was checked against its SHA-256 manifest (76 files).

**D5: 12/12 within the FP64 budgets.** Library defaults at H = 1, D ∈ {1024, 1536, 2048} × (LSTM FP16, sLSTM BF16) ×
B ∈ {100, 128}. These run the multi-tile, multi-slot path with `pre_fused`, dG_LP and `honly` all on. The tables stay
as in section 51.
