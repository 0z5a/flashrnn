# R tiling: ThunderKittens and TileLang cluster kernels (M5)

Two implementations of one algorithm, each also compared with the Triton L2-barrier
path from M0–M4:

* A cluster of k = D/S CTAs per (batch tile, head); one warpgroup per CTA.
* Per step, WGMMA computes the gate pre-activations as R_slice[4S × D] · hᵀ[D × 16]
  (weights as operand A, the hidden state as operand B in shared memory).
* Each CTA writes its rounded h slice into its own next-step hᵀ buffer and copies
  those bytes into every peer's buffer with
  `cp.async.bulk.shared::cluster.shared::cta`, completing on the peer's mbarrier.
  Double buffering plus that data dependency orders the step, so there is no
  per-step cluster barrier.

Sources: `flashrnn/flashrnn2/tk_rtiled.cu` (+ `tk_rtiled.py`, a standalone
shared object built with CUDA 12.8 and called through ctypes, because this
PyTorch is a CUDA 11.8 build) and `build_v2` / `recurrence_v2` in
`flashrnn/flashrnn2/tilelang_rtiled.py`. ThunderKittens is pinned at commit
`67845f5fa48d05343dbbc1ba1403f11061f08d2d` (tarball SHA256
`383868318e06a0b7730b3013c0f74bb61d86b23484f7f4a3b098f309ac92ab7a`); TileLang is
0.1.15. Both kernels implement the spike contract: LSTM/sLSTM, one full logical
call, hidden history and FP32 final carry; no chunk continuation or full-state
output yet. Everything here was run in FP16 (the ThunderKittens build is FP16
only; TileLang also compiles BF16, which is not qualified here).

Known implementation differences that remain:

* Barrier arming follows each framework's idiom. ThunderKittens arms the
  receiver's mbarrier once per step with the total byte count; TileLang's
  `copy_cluster` makes every sender do a remote `mbarrier.arrive.expect_tx`
  before its copy.
* TileLang emits two block barriers before the exchange copies (one explicit,
  one inserted); ThunderKittens emits one.
* Both use coalesced unit-fastest Wx/output access and 16-byte vector loads of
  the R slice.
* D = 768 needs more than 16 CTAs per cluster at the feasible slice widths, so
  neither cluster kernel covers it.

## Qualification

Phase 0 of each batch checks every (backend, cell, D, S) in
{(128,32), (128,64), (192,16), (192,32), (192,64), (256,16), (256,32), (256,64),
(384,32), (512,32)} against an FP64 GPU oracle under the PR #79
`CELL_BUDGET` (hidden history and FP32 carry; LSTM warm, sLSTM zero/mixed/warm;
(B,T,H) = (3,17,2) and (16,128,1)) and requires 20 bit-identical repeats.

* ThunderKittens: 20/20 configurations available (job 47301466).
* TileLang v2: the first check of job 47301466 failed (see below). After the fix,
  20/20 configurations available in job 47302393 and again in job 47303121.
  Job 47302393 then timed nothing: its selection phase ran without the TileLang
  environment (the driver keyed it on an arm name `tilelang`, the arms are
  `tilelang_v2`), so no comparison qualified. Job 47303121 repeated the batch
  with that fixed.

## ThunderKittens vs Triton R tiling (job 47301466, name 32)

One ACC node `as05r1b19`, 9 min 36 s; 200/200 starts passed; 40/40
comparisons complete; raw archive verified (1328 files). Selection on split 0,
frozen table, split-1 timing with 5 starts × 20 ABBA/BAAB blocks and the
10000-draw hierarchical bootstrap. Speedup = Triton time / ThunderKittens time.

### Kernel level (T=1024, H=1)

| Cell | D | B | Baseline | Candidate (k) | Baseline ms/call | Candidate ms/call | µs/step base → cand | Speedup [95% CI] |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| LSTM | 192 | 16 | Triton S=32, 4 warps, padded K | ThunderKittens S=16 (12) | 4.068 | 2.658 | 3.97 → 2.60 | 1.526 [1.524, 1.527] |
| LSTM | 192 | 64 | Triton S=32, 8 warps, padded K | ThunderKittens S=16 (12) | 4.223 | 2.829 | 4.12 → 2.76 | 1.490 [1.483, 1.496] |
| LSTM | 256 | 16 | Triton S=32, 8 warps | ThunderKittens S=16 (16) | 4.204 | 2.941 | 4.11 → 2.87 | 1.428 [1.427, 1.429] |
| LSTM | 256 | 64 | Triton S=64, 8 warps | ThunderKittens S=16 (16) | 4.480 | 3.058 | 4.37 → 2.99 | 1.465 [1.460, 1.470] |
| LSTM | 384 | 16 | Triton S=32, 4 warps | ThunderKittens S=32 (12) | 4.435 | 4.987 | 4.33 → 4.87 | 0.888 [0.887, 0.890] |
| LSTM | 384 | 64 | Triton S=32, 4 warps | ThunderKittens S=32 (12) | 4.644 | 5.020 | 4.54 → 4.90 | 0.924 [0.922, 0.926] |
| LSTM | 512 | 16 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.126 | 5.320 | 5.01 → 5.20 | 0.964 [0.963, 0.965] |
| LSTM | 512 | 64 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.479 | 5.430 | 5.35 → 5.30 | 1.009 [1.006, 1.011] |
| sLSTM | 192 | 16 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 3.685 | 2.391 | 3.60 → 2.33 | 1.540 [1.537, 1.544] |
| sLSTM | 192 | 64 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 3.928 | 2.575 | 3.84 → 2.51 | 1.522 [1.518, 1.524] |
| sLSTM | 256 | 16 | Triton S=64, 4 warps | ThunderKittens S=16 (16) | 4.094 | 2.718 | 4.00 → 2.65 | 1.505 [1.502, 1.508] |
| sLSTM | 256 | 64 | Triton S=16, 4 warps | ThunderKittens S=16 (16) | 4.386 | 2.835 | 4.28 → 2.77 | 1.547 [1.540, 1.554] |
| sLSTM | 384 | 16 | Triton S=32, 8 warps | ThunderKittens S=32 (12) | 5.148 | 5.348 | 5.03 → 5.22 | 0.964 [0.963, 0.965] |
| sLSTM | 384 | 64 | Triton S=32, 8 warps | ThunderKittens S=32 (12) | 5.307 | 5.407 | 5.18 → 5.28 | 0.981 [0.979, 0.983] |
| sLSTM | 512 | 16 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.290 | 5.549 | 5.17 → 5.42 | 0.952 [0.951, 0.954] |
| sLSTM | 512 | 64 | Triton S=32, 8 warps | ThunderKittens S=32 (16) | 5.576 | 5.694 | 5.45 → 5.56 | 0.978 [0.976, 0.980] |

### Classifier (model level, includes each launch path)

| Cell | D | B / C / T | Baseline | Candidate (k) | Baseline ms | Candidate ms | Speedup [95% CI] |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| LSTM | 192 | 1 / 8 / 32 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 4.486 | 3.373 | 1.338 [1.324, 1.352] |
| LSTM | 192 | 16 / 32 / 32 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 1.273 | 0.975 | 1.327 [1.310, 1.346] |
| LSTM | 192 | 32 / 64 / 128 | Triton S=32, 4 warps, padded K | ThunderKittens S=16 (12) | 3.654 | 2.707 | 1.347 [1.345, 1.349] |
| LSTM | 256 | 1 / 8 / 32 | Triton S=32, 4 warps | ThunderKittens S=16 (16) | 4.724 | 3.607 | 1.316 [1.307, 1.330] |
| LSTM | 256 | 16 / 32 / 32 | Triton S=32, 4 warps | ThunderKittens S=16 (16) | 1.334 | 1.061 | 1.259 [1.254, 1.263] |
| LSTM | 256 | 32 / 64 / 128 | Triton S=32, 4 warps | ThunderKittens S=16 (16) | 3.847 | 2.943 | 1.308 [1.306, 1.310] |
| LSTM | 384 | 1 / 8 / 32 | Triton S=32, 4 warps | ThunderKittens S=32 (12) | 4.719 | 4.959 | 0.951 [0.948, 0.955] |
| LSTM | 384 | 16 / 32 / 32 | Triton S=32, 4 warps | ThunderKittens S=32 (12) | 1.313 | 1.456 | 0.903 [0.899, 0.907] |
| LSTM | 384 | 32 / 64 / 128 | Triton S=32, 4 warps | ThunderKittens S=32 (12) | 4.099 | 4.666 | 0.876 [0.874, 0.877] |
| LSTM | 512 | 1 / 8 / 32 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.435 | 5.390 | 1.008 [1.005, 1.012] |
| LSTM | 512 | 16 / 32 / 32 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 1.529 | 1.597 | 0.957 [0.955, 0.960] |
| LSTM | 512 | 32 / 64 / 128 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.642 | 5.573 | 1.009 [1.003, 1.012] |
| sLSTM | 192 | 1 / 8 / 32 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 4.807 | 3.383 | 1.432 [1.422, 1.443] |
| sLSTM | 192 | 16 / 32 / 32 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 1.393 | 1.001 | 1.395 [1.382, 1.415] |
| sLSTM | 192 | 32 / 64 / 128 | Triton S=32, 4 warps | ThunderKittens S=16 (12) | 3.418 | 2.560 | 1.330 [1.327, 1.333] |
| sLSTM | 256 | 1 / 8 / 32 | Triton S=64, 4 warps | ThunderKittens S=16 (16) | 5.272 | 3.656 | 1.451 [1.442, 1.462] |
| sLSTM | 256 | 16 / 32 / 32 | Triton S=16, 4 warps | ThunderKittens S=16 (16) | 1.446 | 1.079 | 1.351 [1.327, 1.380] |
| sLSTM | 256 | 32 / 64 / 128 | Triton S=16, 4 warps | ThunderKittens S=16 (16) | 3.910 | 2.830 | 1.382 [1.371, 1.395] |
| sLSTM | 384 | 1 / 8 / 32 | Triton S=32, 8 warps | ThunderKittens S=32 (12) | 5.791 | 4.757 | 1.222 [1.216, 1.228] |
| sLSTM | 384 | 16 / 32 / 32 | Triton S=32, 8 warps | ThunderKittens S=32 (12) | 1.592 | 1.474 | 1.084 [1.081, 1.088] |
| sLSTM | 384 | 32 / 64 / 128 | Triton S=32, 8 warps | ThunderKittens S=32 (12) | 4.982 | 4.535 | 1.096 [1.094, 1.098] |
| sLSTM | 512 | 1 / 8 / 32 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.872 | 5.223 | 1.126 [1.115, 1.138] |
| sLSTM | 512 | 16 / 32 / 32 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 1.643 | 1.617 | 1.015 [1.009, 1.022] |
| sLSTM | 512 | 32 / 64 / 128 | Triton S=32, 4 warps | ThunderKittens S=32 (16) | 5.945 | 5.724 | 1.037 [1.036, 1.038] |

Summary: at D = 192 and 256, where ThunderKittens selects S = 16 (k = 12 and 16),
it is 1.43×–1.55× faster per call at the kernel level and 1.26×–1.45× faster in
the classifier. At D = 384 and 512, where only S = 32 fits, it is slower or
level at the kernel level (0.888×–1.009×); in the classifier LSTM is 0.876×–1.009×
and sLSTM 1.015×–1.222×.

## ThunderKittens vs TileLang (job 47303121, name 37)

One ACC node `as05r1b22`, 18 min 2 s; 400/400 starts passed; 80/80 comparisons
complete; raw archive verified (2505 files). Same protocol as job 47301466.
Speedup = baseline time / candidate time.

### ThunderKittens (candidate) vs TileLang v2 (baseline)

Kernel level (T=1024, H=1):

| Cell | D | B | Baseline | Candidate (k) | Baseline ms/call | Candidate ms/call | µs/step base → cand | Speedup [95% CI] |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| LSTM | 192 | 16 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 2.182 | 2.623 | 2.13 → 2.56 | 0.832 [0.831, 0.832] |
| LSTM | 192 | 64 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 2.416 | 2.952 | 2.36 → 2.88 | 0.818 [0.818, 0.819] |
| LSTM | 256 | 16 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 2.559 | 2.971 | 2.50 → 2.90 | 0.861 [0.861, 0.862] |
| LSTM | 256 | 64 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 2.574 | 3.039 | 2.51 → 2.97 | 0.849 [0.846, 0.852] |
| LSTM | 384 | 16 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 3.512 | 4.978 | 3.43 → 4.86 | 0.707 [0.706, 0.709] |
| LSTM | 384 | 64 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 3.719 | 5.213 | 3.63 → 5.09 | 0.713 [0.713, 0.713] |
| LSTM | 512 | 16 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 4.126 | 5.392 | 4.03 → 5.27 | 0.766 [0.764, 0.767] |
| LSTM | 512 | 64 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 4.082 | 5.383 | 3.99 → 5.26 | 0.758 [0.757, 0.759] |
| sLSTM | 192 | 16 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 2.024 | 2.356 | 1.98 → 2.30 | 0.858 [0.858, 0.859] |
| sLSTM | 192 | 64 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 2.252 | 2.676 | 2.20 → 2.61 | 0.840 [0.838, 0.843] |
| sLSTM | 256 | 16 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 2.361 | 2.747 | 2.31 → 2.68 | 0.860 [0.858, 0.861] |
| sLSTM | 256 | 64 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 2.379 | 2.819 | 2.32 → 2.75 | 0.845 [0.841, 0.848] |
| sLSTM | 384 | 16 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 3.149 | 5.334 | 3.08 → 5.21 | 0.591 [0.590, 0.591] |
| sLSTM | 384 | 64 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 3.338 | 5.557 | 3.26 → 5.43 | 0.601 [0.600, 0.601] |
| sLSTM | 512 | 16 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 3.784 | 5.653 | 3.70 → 5.52 | 0.670 [0.668, 0.671] |
| sLSTM | 512 | 64 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 3.734 | 5.662 | 3.65 → 5.53 | 0.660 [0.659, 0.662] |

Classifier (model level, includes each launch path):

| Cell | D | B / C / T | Baseline | Candidate (k) | Baseline ms | Candidate ms | Speedup [95% CI] |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| LSTM | 192 | 1 / 8 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 3.049 | 3.268 | 0.931 [0.929, 0.934] |
| LSTM | 192 | 16 / 32 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 0.893 | 0.975 | 0.914 [0.912, 0.917] |
| LSTM | 192 | 32 / 64 / 128 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 2.446 | 2.747 | 0.894 [0.891, 0.899] |
| LSTM | 256 | 1 / 8 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 3.323 | 3.556 | 0.937 [0.934, 0.940] |
| LSTM | 256 | 16 / 32 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 0.944 | 1.030 | 0.915 [0.912, 0.918] |
| LSTM | 256 | 32 / 64 / 128 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 2.848 | 3.098 | 0.920 [0.918, 0.921] |
| LSTM | 384 | 1 / 8 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 4.394 | 5.001 | 0.877 [0.875, 0.878] |
| LSTM | 384 | 16 / 32 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 1.222 | 1.451 | 0.843 [0.841, 0.844] |
| LSTM | 384 | 32 / 64 / 128 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 3.815 | 4.668 | 0.816 [0.815, 0.817] |
| LSTM | 512 | 1 / 8 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 5.092 | 5.557 | 0.917 [0.915, 0.919] |
| LSTM | 512 | 16 / 32 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 1.380 | 1.589 | 0.867 [0.865, 0.868] |
| LSTM | 512 | 32 / 64 / 128 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 4.536 | 5.536 | 0.820 [0.817, 0.823] |
| sLSTM | 192 | 1 / 8 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 3.143 | 3.316 | 0.953 [0.947, 0.959] |
| sLSTM | 192 | 16 / 32 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 0.931 | 1.001 | 0.934 [0.926, 0.945] |
| sLSTM | 192 | 32 / 64 / 128 | TileLang v2 S=16 | ThunderKittens S=16 (12) | 2.360 | 2.578 | 0.913 [0.912, 0.914] |
| sLSTM | 256 | 1 / 8 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 3.438 | 3.602 | 0.959 [0.954, 0.965] |
| sLSTM | 256 | 16 / 32 / 32 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 0.972 | 1.047 | 0.933 [0.929, 0.938] |
| sLSTM | 256 | 32 / 64 / 128 | TileLang v2 S=16 | ThunderKittens S=16 (16) | 2.725 | 2.949 | 0.924 [0.922, 0.925] |
| sLSTM | 384 | 1 / 8 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 4.302 | 4.781 | 0.897 [0.890, 0.902] |
| sLSTM | 384 | 16 / 32 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 1.190 | 1.455 | 0.817 [0.814, 0.821] |
| sLSTM | 384 | 32 / 64 / 128 | TileLang v2 S=32 | ThunderKittens S=32 (12) | 3.502 | 4.517 | 0.777 [0.774, 0.781] |
| sLSTM | 512 | 1 / 8 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 4.987 | 5.384 | 0.927 [0.926, 0.929] |
| sLSTM | 512 | 16 / 32 / 32 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 1.357 | 1.620 | 0.836 [0.834, 0.838] |
| sLSTM | 512 | 32 / 64 / 128 | TileLang v2 S=32 | ThunderKittens S=32 (16) | 4.281 | 5.698 | 0.752 [0.751, 0.753] |

### TileLang v2 (candidate) vs Triton R tiling (baseline)

Kernel level (T=1024, H=1):

| Cell | D | B | Baseline | Candidate (k) | Baseline ms/call | Candidate ms/call | µs/step base → cand | Speedup [95% CI] |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| LSTM | 192 | 16 | Triton S=32, 4 warps, padded K | TileLang v2 S=16 (12) | 3.890 | 2.238 | 3.80 → 2.19 | 1.735 [1.733, 1.737] |
| LSTM | 192 | 64 | Triton S=32, 8 warps, padded K | TileLang v2 S=16 (12) | 4.262 | 2.300 | 4.16 → 2.25 | 1.851 [1.843, 1.856] |
| LSTM | 256 | 16 | Triton S=32, 8 warps | TileLang v2 S=16 (16) | 4.259 | 2.510 | 4.16 → 2.45 | 1.693 [1.689, 1.696] |
| LSTM | 256 | 64 | Triton S=64, 8 warps | TileLang v2 S=16 (16) | 4.549 | 2.698 | 4.44 → 2.63 | 1.687 [1.685, 1.689] |
| LSTM | 384 | 16 | Triton S=32, 4 warps | TileLang v2 S=32 (12) | 4.421 | 3.574 | 4.32 → 3.49 | 1.236 [1.231, 1.240] |
| LSTM | 384 | 64 | Triton S=32, 4 warps | TileLang v2 S=32 (12) | 4.617 | 3.550 | 4.51 → 3.47 | 1.300 [1.297, 1.302] |
| LSTM | 512 | 16 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.120 | 4.057 | 5.00 → 3.96 | 1.262 [1.259, 1.265] |
| LSTM | 512 | 64 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.545 | 4.275 | 5.42 → 4.17 | 1.297 [1.294, 1.298] |
| sLSTM | 192 | 16 | Triton S=32, 4 warps | TileLang v2 S=16 (12) | 3.436 | 2.072 | 3.36 → 2.02 | 1.658 [1.655, 1.662] |
| sLSTM | 192 | 64 | Triton S=32, 4 warps | TileLang v2 S=16 (12) | 3.952 | 2.145 | 3.86 → 2.10 | 1.836 [1.829, 1.843] |
| sLSTM | 256 | 16 | Triton S=64, 4 warps | TileLang v2 S=16 (16) | 4.169 | 2.316 | 4.07 → 2.26 | 1.795 [1.791, 1.798] |
| sLSTM | 256 | 64 | Triton S=16, 4 warps | TileLang v2 S=16 (16) | 4.403 | 2.490 | 4.30 → 2.43 | 1.768 [1.766, 1.771] |
| sLSTM | 384 | 16 | Triton S=32, 8 warps | TileLang v2 S=32 (12) | 5.164 | 3.208 | 5.04 → 3.13 | 1.607 [1.603, 1.611] |
| sLSTM | 384 | 64 | Triton S=32, 8 warps | TileLang v2 S=32 (12) | 5.333 | 3.199 | 5.21 → 3.12 | 1.666 [1.661, 1.670] |
| sLSTM | 512 | 16 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.300 | 3.721 | 5.18 → 3.63 | 1.425 [1.421, 1.430] |
| sLSTM | 512 | 64 | Triton S=32, 8 warps | TileLang v2 S=32 (16) | 5.680 | 3.909 | 5.55 → 3.82 | 1.454 [1.451, 1.457] |

Classifier (model level, includes each launch path):

| Cell | D | B / C / T | Baseline | Candidate (k) | Baseline ms | Candidate ms | Speedup [95% CI] |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| LSTM | 192 | 1 / 8 / 32 | Triton S=32, 4 warps, padded K | TileLang v2 S=16 (12) | 4.450 | 3.108 | 1.435 [1.429, 1.442] |
| LSTM | 192 | 16 / 32 / 32 | Triton S=32, 4 warps, padded K | TileLang v2 S=16 (12) | 1.300 | 0.908 | 1.430 [1.421, 1.442] |
| LSTM | 192 | 32 / 64 / 128 | Triton S=32, 4 warps, padded K | TileLang v2 S=16 (12) | 3.681 | 2.416 | 1.517 [1.514, 1.520] |
| LSTM | 256 | 1 / 8 / 32 | Triton S=32, 4 warps | TileLang v2 S=16 (16) | 4.741 | 3.446 | 1.382 [1.377, 1.388] |
| LSTM | 256 | 16 / 32 / 32 | Triton S=32, 4 warps | TileLang v2 S=16 (16) | 1.322 | 0.971 | 1.365 [1.354, 1.383] |
| LSTM | 256 | 32 / 64 / 128 | Triton S=32, 4 warps | TileLang v2 S=16 (16) | 3.892 | 2.761 | 1.405 [1.403, 1.408] |
| LSTM | 384 | 1 / 8 / 32 | Triton S=32, 4 warps | TileLang v2 S=32 (12) | 4.722 | 4.350 | 1.098 [1.085, 1.118] |
| LSTM | 384 | 16 / 32 / 32 | Triton S=32, 4 warps | TileLang v2 S=32 (12) | 1.322 | 1.247 | 1.064 [1.044, 1.081] |
| LSTM | 384 | 32 / 64 / 128 | Triton S=32, 4 warps | TileLang v2 S=32 (12) | 4.102 | 3.907 | 1.053 [1.051, 1.054] |
| LSTM | 512 | 1 / 8 / 32 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.471 | 4.923 | 1.112 [1.110, 1.115] |
| LSTM | 512 | 16 / 32 / 32 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 1.525 | 1.387 | 1.102 [1.098, 1.107] |
| LSTM | 512 | 32 / 64 / 128 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.658 | 4.737 | 1.193 [1.192, 1.194] |
| sLSTM | 192 | 1 / 8 / 32 | Triton S=32, 4 warps | TileLang v2 S=16 (12) | 4.788 | 3.216 | 1.499 [1.480, 1.520] |
| sLSTM | 192 | 16 / 32 / 32 | Triton S=32, 4 warps | TileLang v2 S=16 (12) | 1.403 | 0.957 | 1.473 [1.464, 1.483] |
| sLSTM | 192 | 32 / 64 / 128 | Triton S=32, 4 warps | TileLang v2 S=16 (12) | 3.415 | 2.317 | 1.466 [1.463, 1.469] |
| sLSTM | 256 | 1 / 8 / 32 | Triton S=64, 4 warps | TileLang v2 S=16 (16) | 5.263 | 3.564 | 1.485 [1.473, 1.500] |
| sLSTM | 256 | 16 / 32 / 32 | Triton S=16, 4 warps | TileLang v2 S=16 (16) | 1.427 | 0.997 | 1.429 [1.419, 1.440] |
| sLSTM | 256 | 32 / 64 / 128 | Triton S=16, 4 warps | TileLang v2 S=16 (16) | 3.929 | 2.646 | 1.483 [1.481, 1.486] |
| sLSTM | 384 | 1 / 8 / 32 | Triton S=32, 8 warps | TileLang v2 S=32 (12) | 5.866 | 4.294 | 1.371 [1.362, 1.379] |
| sLSTM | 384 | 16 / 32 / 32 | Triton S=32, 8 warps | TileLang v2 S=32 (12) | 1.599 | 1.218 | 1.313 [1.310, 1.318] |
| sLSTM | 384 | 32 / 64 / 128 | Triton S=32, 8 warps | TileLang v2 S=32 (12) | 4.988 | 3.575 | 1.394 [1.392, 1.396] |
| sLSTM | 512 | 1 / 8 / 32 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.926 | 4.821 | 1.229 [1.225, 1.235] |
| sLSTM | 512 | 16 / 32 / 32 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 1.643 | 1.383 | 1.192 [1.190, 1.195] |
| sLSTM | 512 | 32 / 64 / 128 | Triton S=32, 4 warps | TileLang v2 S=32 (16) | 5.940 | 4.461 | 1.331 [1.329, 1.332] |

Summary:

* With the same algorithm, this TileLang implementation is faster than this
  ThunderKittens implementation in all 40 comparisons: ThunderKittens runs at
  0.591×–0.861× of TileLang's speed per kernel call and 0.752×–0.959× in the
  classifier. The gap is largest where both use S = 32 (D = 384 and 512).
* TileLang v2 is faster than Triton R tiling in all 40 comparisons: 1.236×–1.851×
  at the kernel level and 1.053×–1.517× in the classifier, including D = 384 and
  512 where ThunderKittens was not faster than Triton.
* These compare two implementations written for this iteration, not the ceiling
  of either framework. The ThunderKittens kernel is hand-written CUDA with
  ThunderKittens tiles and WGMMA; TileLang generates its own layouts, pipeline
  and synchronization from a higher-level description. The remaining recorded
  differences (barrier-arming idiom, two vs one block barrier before the
  exchange) favour ThunderKittens, so they do not explain the gap.


## TileLang v2 failure and fix

Job 47301466's first TileLang v2 check failed with `CUDA_ERROR_LAUNCH_FAILED`.
Debug job 47301592 showed that a synchronous-exchange variant of the same
kernel was correct and that the bulk variant spun in `Barrier::wait` until
TileLang's wait trapped (compute-sanitizer memcheck located every thread there;
synccheck clean). TileLang's `tma_store_cluster` already performs a remote
`mbarrier.arrive.expect_tx` per copy, while the kernel had also armed the
receiver with an arrive count of 1, so arrivals and transaction bytes never
matched. With the receiver count set to k − 1 and no receiver-side arming,
debug job 47301920 ran every case, including 16-CTA clusters, with the same
error against FP64 as the synchronous variant and memcheck clean.
