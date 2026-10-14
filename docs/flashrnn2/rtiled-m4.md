# R tiling: paired timing (M4: E2, E3, E4)

Job `47299690` (name 31), one ACC node `as07r3b09`, four H100 lanes, 26 min 16 s.
All 300 planned process starts passed; 60/60 comparisons are complete; the
finalizer re-hashed every member of the 2062-file raw archive
(`ARCHIVE_VERIFIED`, kept on MN5). Evidence in this repository:
`evidence/flashrnn2-rtiled-m4-20261014/job47299690/` (frozen table, summary,
v1 build records, split-0 selections, per-start JSON in
`phase2-starts-json.tar.gz`). A first attempt, job `47299215`, was cancelled by
its owner in phase 0 for two harness defects (see below); it timed nothing.

## Workload and protocol

* Workload: the existing external-baseline classifier (four layers, vocabulary
  512, FP16, C requests served in B-sized groups of T tokens; ID transfer,
  embedding, every input projection and recurrence, readout and host delivery
  are timed). Only the width D changes. (B, C, T) ∈ (1, 8, 32), (16, 32, 32),
  (32, 64, 128).
* Every FlashRNN2 arm uses the public per-call API with hidden output and the
  FP32 compute-carry contract; v1 uses its public `flashrnn()` call.
* Selection on split 0 (its own ID fixtures) among the M0-qualified
  configurations; the comparison table was frozen and hashed before any
  split-1 run. E3 candidates are additionally served through
  `ExecutionPolicy.select` with the frozen exact-shape table and identity.
* Before timing, every arm was checked against an FP64 model oracle (FP64
  embedding, projections, recurrence with the hidden operand rounded to FP16,
  readout) and against the other arm, for the logits and every layer's hidden
  history, under the original model budget (atol 0.015, rtol 0.03,
  max 0.03). Every timed block re-checks the delivered logits.
* Timing: 5 process starts × 20 paired blocks, ABBA on even blocks and BAAB on
  odd ones; the unit is one burst of all C/B groups. Statistic: geometric mean
  of baseline/candidate time, 10000-draw hierarchical bootstrap (starts, then
  blocks), seed 20261013 + comparison index, recomputed from the raw blocks by
  `tools/rtiled_m4_analyze.py`.
* Four GPU lanes ran concurrently, each pinned to the CPU cores local to its
  GPU; no compiler ran during timing (builds and selection finished behind
  barriers first). The earlier external cohort ran serially on one GPU.

## Results

### E3: R tiling vs pinned FlashRNN1 CUDA fused

| Cell | D | B / C / T | Baseline | Candidate (k) | Baseline ms | Candidate ms | Speedup [95% CI] |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| LSTM | 192 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (6) | 7.204 | 4.484 | 1.606 [1.599, 1.614] |
| LSTM | 192 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (6) | 1.855 | 1.355 | 1.373 [1.363, 1.384] |
| LSTM | 192 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps, padded K (6) | 4.766 | 3.728 | 1.281 [1.280, 1.282] |
| LSTM | 256 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (8) | 7.662 | 4.849 | 1.582 [1.578, 1.587] |
| LSTM | 256 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (8) | 1.828 | 1.376 | 1.333 [1.322, 1.344] |
| LSTM | 256 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (8) | 5.019 | 3.910 | 1.282 [1.280, 1.284] |
| LSTM | 384 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (12) | 7.629 | 4.833 | 1.578 [1.570, 1.586] |
| LSTM | 384 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (12) | 1.749 | 1.365 | 1.280 [1.269, 1.291] |
| LSTM | 384 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (12) | 6.316 | 4.072 | 1.552 [1.550, 1.554] |
| LSTM | 512 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (16) | 7.758 | 5.519 | 1.403 [1.398, 1.408] |
| LSTM | 512 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (16) | 1.787 | 1.591 | 1.120 [1.108, 1.130] |
| LSTM | 512 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (16) | 7.479 | 5.689 | 1.315 [1.314, 1.316] |
| LSTM | 768 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (24) | 7.948 | 5.895 | 1.349 [1.342, 1.358] |
| LSTM | 768 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (24) | 2.088 | 1.741 | 1.214 [1.199, 1.232] |
| LSTM | 768 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (24) | 10.864 | 6.517 | 1.666 [1.658, 1.670] |
| sLSTM | 192 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (6) | 7.621 | 4.841 | 1.571 [1.561, 1.579] |
| sLSTM | 192 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (6) | 1.954 | 1.467 | 1.321 [1.310, 1.333] |
| sLSTM | 192 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (6) | 4.957 | 3.461 | 1.434 [1.430, 1.439] |
| sLSTM | 256 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=16, 4 warps (16) | 8.098 | 5.308 | 1.529 [1.522, 1.539] |
| sLSTM | 256 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (8) | 1.937 | 1.509 | 1.291 [1.280, 1.301] |
| sLSTM | 256 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=16, 4 warps (16) | 5.015 | 3.942 | 1.270 [1.268, 1.272] |
| sLSTM | 384 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 8 warps (12) | 8.222 | 5.896 | 1.394 [1.386, 1.401] |
| sLSTM | 384 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 8 warps (12) | 1.869 | 1.633 | 1.145 [1.136, 1.154] |
| sLSTM | 384 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 8 warps (12) | 6.520 | 4.989 | 1.304 [1.303, 1.306] |
| sLSTM | 512 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (16) | 8.189 | 6.012 | 1.360 [1.350, 1.368] |
| sLSTM | 512 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (16) | 1.907 | 1.704 | 1.123 [1.114, 1.134] |
| sLSTM | 512 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (16) | 7.890 | 5.919 | 1.332 [1.328, 1.336] |
| sLSTM | 768 | 1 / 8 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (24) | 8.353 | 6.276 | 1.332 [1.322, 1.342] |
| sLSTM | 768 | 16 / 32 / 32 | v1 CUDA fused | R-tiled S=32, 4 warps (24) | 2.213 | 1.825 | 1.208 [1.190, 1.227] |
| sLSTM | 768 | 32 / 64 / 128 | v1 CUDA fused | R-tiled S=32, 4 warps (24) | 11.210 | 6.515 | 1.718 [1.713, 1.721] |

### E2: D ≤ 128 switch point (single CTA vs R tiling)

| Cell | D | B / C / T | Baseline | Candidate (k) | Baseline ms | Candidate ms | Speedup [95% CI] |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| LSTM | 96 | 1 / 8 / 32 | single CTA, 8 warps | R-tiled S=64, 8 warps (2) | 3.605 | 4.300 | 0.837 [0.831, 0.842] |
| LSTM | 96 | 1 / 8 / 32 | single CTA, 8 warps | R-tiled S=32, 4 warps (3) | 3.621 | 4.171 | 0.868 [0.864, 0.872] |
| LSTM | 96 | 16 / 32 / 32 | single CTA, 8 warps | R-tiled S=64, 8 warps (2) | 1.017 | 1.225 | 0.827 [0.820, 0.834] |
| LSTM | 96 | 16 / 32 / 32 | single CTA, 8 warps | R-tiled S=32, 4 warps (3) | 1.007 | 1.142 | 0.871 [0.858, 0.882] |
| LSTM | 96 | 32 / 64 / 128 | single CTA, 8 warps | R-tiled S=64, 8 warps (2) | 3.021 | 3.544 | 0.855 [0.852, 0.858] |
| LSTM | 96 | 32 / 64 / 128 | single CTA, 8 warps | R-tiled S=32, 4 warps (3) | 3.042 | 3.250 | 0.935 [0.934, 0.937] |
| LSTM | 128 | 1 / 8 / 32 | single CTA, 8 warps | R-tiled S=64, 8 warps (2) | 3.652 | 4.360 | 0.838 [0.833, 0.842] |
| LSTM | 128 | 1 / 8 / 32 | single CTA, 8 warps | R-tiled S=32, 4 warps (4) | 3.645 | 4.296 | 0.841 [0.833, 0.848] |
| LSTM | 128 | 16 / 32 / 32 | single CTA, 8 warps | R-tiled S=64, 8 warps (2) | 1.031 | 1.175 | 0.875 [0.869, 0.880] |
| LSTM | 128 | 16 / 32 / 32 | single CTA, 8 warps | R-tiled S=32, 4 warps (4) | 1.024 | 1.165 | 0.875 [0.872, 0.877] |
| LSTM | 128 | 32 / 64 / 128 | single CTA, 8 warps | R-tiled S=64, 8 warps (2) | 3.063 | 3.431 | 0.893 [0.891, 0.895] |
| LSTM | 128 | 32 / 64 / 128 | single CTA, 8 warps | R-tiled S=32, 4 warps (4) | 3.080 | 3.222 | 0.957 [0.955, 0.959] |
| sLSTM | 96 | 1 / 8 / 32 | single CTA, 8 warps | R-tiled S=64, 8 warps, padded K (2) | 3.980 | 4.862 | 0.817 [0.813, 0.821] |
| sLSTM | 96 | 1 / 8 / 32 | single CTA, 8 warps | R-tiled S=32, 4 warps, padded K (3) | 4.013 | 4.788 | 0.836 [0.830, 0.841] |
| sLSTM | 96 | 16 / 32 / 32 | single CTA, 4 warps | R-tiled S=64, 8 warps, padded K (2) | 1.106 | 1.305 | 0.839 [0.833, 0.845] |
| sLSTM | 96 | 16 / 32 / 32 | single CTA, 4 warps | R-tiled S=32, 4 warps, padded K (3) | 1.104 | 1.294 | 0.842 [0.831, 0.851] |
| sLSTM | 96 | 32 / 64 / 128 | single CTA, 4 warps | R-tiled S=64, 8 warps, padded K (2) | 3.148 | 3.577 | 0.880 [0.876, 0.884] |
| sLSTM | 96 | 32 / 64 / 128 | single CTA, 4 warps | R-tiled S=32, 4 warps, padded K (3) | 3.174 | 3.321 | 0.955 [0.953, 0.958] |
| sLSTM | 128 | 1 / 8 / 32 | single CTA, 4 warps | R-tiled S=64, 8 warps (2) | 3.912 | 4.844 | 0.799 [0.790, 0.806] |
| sLSTM | 128 | 1 / 8 / 32 | single CTA, 4 warps | R-tiled S=32, 4 warps (4) | 3.876 | 4.809 | 0.806 [0.800, 0.812] |
| sLSTM | 128 | 16 / 32 / 32 | single CTA, 4 warps | R-tiled S=64, 8 warps (2) | 1.116 | 1.300 | 0.857 [0.852, 0.861] |
| sLSTM | 128 | 16 / 32 / 32 | single CTA, 4 warps | R-tiled S=32, 8 warps (4) | 1.107 | 1.295 | 0.852 [0.847, 0.857] |
| sLSTM | 128 | 32 / 64 / 128 | single CTA, 4 warps | R-tiled S=64, 8 warps (2) | 3.175 | 3.418 | 0.926 [0.916, 0.936] |
| sLSTM | 128 | 32 / 64 / 128 | single CTA, 4 warps | R-tiled S=32, 4 warps (4) | 3.138 | 3.283 | 0.955 [0.946, 0.963] |

### E4: Triton R tiling vs TileLang cluster spike

| Cell | D | B / C / T | Baseline | Candidate (k) | Baseline ms | Candidate ms | Speedup [95% CI] |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| LSTM | 256 | 1 / 8 / 32 | R-tiled S=32, 4 warps | TileLang spike, S=32 (8) | 4.717 | 5.930 | 0.796 [0.791, 0.801] |
| LSTM | 256 | 16 / 32 / 32 | R-tiled S=32, 4 warps | TileLang spike, S=32 (8) | 1.309 | 1.793 | 0.732 [0.728, 0.737] |
| LSTM | 256 | 32 / 64 / 128 | R-tiled S=32, 4 warps | TileLang spike, S=32 (8) | 3.875 | 5.246 | 0.737 [0.735, 0.738] |
| sLSTM | 256 | 1 / 8 / 32 | R-tiled S=16, 4 warps | TileLang spike, S=32 (8) | 5.277 | 6.008 | 0.878 [0.872, 0.884] |
| sLSTM | 256 | 16 / 32 / 32 | R-tiled S=32, 4 warps | TileLang spike, S=32 (8) | 1.436 | 1.699 | 0.844 [0.840, 0.848] |
| sLSTM | 256 | 32 / 64 / 128 | R-tiled S=16, 4 warps | TileLang spike, S=32 (8) | 3.914 | 5.548 | 0.705 [0.704, 0.706] |

Summary:

* **E3:** FlashRNN2 R tiling is faster than pinned FlashRNN1 CUDA fused in
  all 30 comparisons; every 95% interval lies above 1. Range 1.120× (LSTM,
  D=512, B=16) to 1.718× (sLSTM, D=768, B=32); geometric mean over the 30
  comparisons 1.367×.
* **E2:** the existing single-CTA path is faster than R tiling at D=96 and
  D=128 in all 24 comparisons (0.799×–0.957×; every interval below 1). The
  D ≤ 128 default therefore stays single CTA. The earlier unpaired
  prepared-graph screen (kernels only, CUDA Graph replay) had slightly favored
  R tiling at these widths. Here both arms go through the public per-call API,
  and the R-tiled call allocates more per call (exchange buffer, two carries,
  zeroed barrier counters and error flag). That host-side cost is a likely
  contributor; this run does not isolate it.
* **E4:** the TileLang cluster spike is slower than Triton R tiling in all six
  comparisons (0.705×–0.878×). This measures the unoptimized M3 spike.

## Pinned FlashRNN1 CUDA fused at large D

All 30 (cell, D, B) builds of `0b84fc30120ecccbde97976bc5b086f3215572d2`
built (H100 sm_90 SASS through the external cohort's TaskBuilder transport,
pinned files unchanged), ran and passed a one-layer FP64 check (T=32). The
ConstrINT-selected forward tiling was recorded per build:

| Cell | D | B | build s | FP64 check (model budget used) | ConstrINT forward tiles (gate tiling count, warp tiling count hidden, warp tile batch×gate, cached hidden) |
| --- | ---: | ---: | ---: | ---: | --- |
| LSTM | 192 | 1 | 121 | 1.2% | 24, 6, 8×32, 2 |
| LSTM | 192 | 16 | 74 | 1.7% | 48, 6, 16×16, 2 |
| LSTM | 192 | 32 | 74 | 1.7% | 96, 6, 32×8, 2 |
| LSTM | 256 | 1 | 62 | 1.4% | 32, 8, 8×32, 2 |
| LSTM | 256 | 16 | 65 | 1.8% | 64, 8, 16×16, 2 |
| LSTM | 256 | 32 | 72 | 1.6% | 128, 8, 32×8, 2 |
| LSTM | 384 | 1 | 73 | 1.3% | 48, 8, 8×32, 3 |
| LSTM | 384 | 16 | 80 | 1.6% | 96, 8, 16×16, 3 |
| LSTM | 384 | 32 | 84 | 1.6% | 96, 4, 32×8, 6 |
| LSTM | 512 | 1 | 43 | 1.2% | 64, 8, 8×32, 4 |
| LSTM | 512 | 16 | 50 | 1.6% | 128, 8, 16×16, 4 |
| LSTM | 512 | 32 | 52 | 1.6% | 128, 4, 32×8, 8 |
| LSTM | 768 | 1 | 54 | 1.5% | 96, 8, 8×32, 6 |
| LSTM | 768 | 16 | 56 | 1.9% | 96, 4, 16×16, 12 |
| LSTM | 768 | 32 | 55 | 2.0% | 128, 2, 32×8, 24 |
| sLSTM | 192 | 1 | 41 | 1.1% | 24, 6, 8×32, 2 |
| sLSTM | 192 | 16 | 39 | 1.3% | 48, 6, 16×16, 2 |
| sLSTM | 192 | 32 | 39 | 1.4% | 96, 6, 32×8, 2 |
| sLSTM | 256 | 1 | 44 | 1.2% | 32, 8, 8×32, 2 |
| sLSTM | 256 | 16 | 42 | 1.3% | 64, 8, 16×16, 2 |
| sLSTM | 256 | 32 | 50 | 1.3% | 128, 8, 32×8, 2 |
| sLSTM | 384 | 1 | 49 | 1.1% | 48, 8, 8×32, 3 |
| sLSTM | 384 | 16 | 48 | 1.3% | 96, 8, 16×16, 3 |
| sLSTM | 384 | 32 | 41 | 1.5% | 96, 4, 32×8, 6 |
| sLSTM | 512 | 1 | 47 | 1.2% | 64, 8, 8×32, 4 |
| sLSTM | 512 | 16 | 44 | 1.4% | 128, 8, 16×16, 4 |
| sLSTM | 512 | 32 | 39 | 1.5% | 128, 4, 32×8, 8 |
| sLSTM | 768 | 1 | 40 | 1.1% | 96, 8, 8×32, 6 |
| sLSTM | 768 | 16 | 52 | 1.5% | 96, 4, 16×16, 12 |
| sLSTM | 768 | 32 | 47 | 1.5% | 128, 2, 32×8, 24 |

## Deviations from the plan

* D=96 in E2 uses k=2 (S=64, partial last slice) and k=3 (S=32): power-of-two
  slices cannot give k=4.
* Final states are not compared across arms (FP32 compute carry vs
  input-dtype snapshot); logits and every layer's hidden history are.
* Saved qualification tensors hold logits and the last-step hidden state per
  layer; full histories are checked on the GPU only (size).
* The four lanes timed concurrently.
* The TileLang arm is the unoptimized M3 spike.

## Failed attempt (job 47299215, name 63)

Cancelled by its owner after 2 min 47 s in phase 0, before any timing. Both
causes were harness defects, not limits of FlashRNN1: concurrent builders
raced on the shared `gpu_info` TaskBuilder module (`PyInit_gpu_info2…`
missing, artifact hash mismatch), and the external cohort's `torch.amp`
namespace adapter (the cluster's torch 2.4 pre-release lacks
`torch.amp.custom_fwd`) had not been carried over. The repaired run builds the
first configuration alone and applies the same adapter. Its finalizer
archived and verified the partial outputs
(`evidence/flashrnn2-rtiled-m4-20261014/job47299215-cancelled/`).
