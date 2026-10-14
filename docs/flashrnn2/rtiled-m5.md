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
* TileLang v2: <!-- TL_QUAL -->

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

## ThunderKittens vs TileLang (job 47302393, name 36)

<!-- TK_VS_TL -->

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
