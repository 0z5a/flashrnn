# FlashRNN2 with many heads in one launch (H > 1)

Status: pre-registration, written 2026-10-17 before any H > 1 timing. Single GPU (H100) only.

## 1. Question

xLSTM-style models run many independent heads per layer. FlashRNN1's `cuda_fused` processes them as one launch per
head group (`FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT`, usually one head per group), each group on its own CUDA stream
with its own cuBLAS handle, issued by a host loop. FlashRNN2 puts the head on the grid: the single-CTA kernel
(D ≤ 128) launches one CTA per (16-row batch tile, head), and the R-tiled kernels (D > 128) schedule every
(batch tile, head) group over the co-resident slots of one persistent launch. Heads exchange nothing, so the
question is only whether one launch over all heads is correct and faster than (a) a per-head loop and (b) FlashRNN1,
in particular for H ≫ 1 with D ≤ 128.

Until now only H = 1 was measured; the packaged tables hold H = 1 shapes, so every H > 1 shape runs the untuned
default configuration (D ≤ 128: single CTA, four warps; D > 128: Triton R tiling, eight warps, S = 64 up to D = 256,
else S = 32). The training path keeps the TileLang forward only for H = 1 until TileLang is qualified at H > 1.

## 2. Qualification (gates every later phase per path)

`tools/heads_qualify.py`, one process per case, the same pinned `KernelConfig` for every run of a case.

* Paths: single CTA (D 64, 128), Triton R-tiled (D 256, 768, S 32, eight warps), TileLang cluster (D 256, S 16),
  TileLang grid (D 768, S 16); LSTM and sLSTM; FP16 and BF16.
* Heads: 2, 3, 5 and 16 (non-powers of two included; on the R-tiled grid path a case with groups not a multiple of the
  co-resident slots), and 256 / 1024 on the single-CTA path (four heads checked: 0, 1, H/2, H − 1).
* Forward: for every checked head h, the H-head output sliced to h equals an H = 1 launch on head h's inputs
  bitwise (hidden output and FP32 carry); the H-head output is within the FP64-reference budget of
  `tools/bw_qualify.py` (FP16 1e-2, BF16 5e-2 relative Frobenius; max abs ≤ 5× that, scaled).
* Training (H ≤ 16): d_wx and the initial-state gradient are kernel-produced and must be bitwise equal per head;
  dR and dbias come from GEMMs after the kernel (cuBLAS may choose another algorithm for another batch count), so
  they must be within the FP64 gradient budget, and their bitwise equality is only recorded.
* A path passes when every one of its cases passes. A path that fails is excluded from the timing phases and
  reported as a negative result.

The scale-out phase A many-heads task produced no data (CUDA events read before synchronizing, and a sticky illegal
memory access reported at the first sLSTM kernel load). Before qualification, each arm of that row (default forward,
TileLang forward, training step; LSTM and sLSTM; D = 192, B = 16, H = 1 and 4) runs in its own process with
`CUDA_LAUNCH_BLOCKING=1`, and the default forward and the training step of sLSTM H = 1 also under compute-sanitizer
memcheck. This is diagnostic; it gates nothing, and its outcome is reported as found.

## 3. Timing

`tools/heads_bench.py`, configurations as the library selects them (recorded per case).

Arms: **one** (all heads, one launch), **loop** (H calls of H = 1, same configuration), **loop_graph** (the same H
calls captured into one CUDA graph, forward only: separates one launch from host launch overhead), **v1** (pinned
FlashRNN1 `cuda_fused`, `num_heads = H`, `hidden_dim = H·D`; a build or run failure is the arm's result).

* Sweep (plain medians of 5 calls after 2 warm-up calls): LSTM/sLSTM × FP16/BF16 × D ∈ {64, 128, 256, 768} ×
  H ∈ {1, 4, 16, 64, 256, 1024} × B = 16 (plus B = 64 for D ≤ 128), T = 256, forward (arms one, loop, loop_graph)
  and training step (one, loop). Cases whose estimated memory exceeds 90% of free device memory are skipped and listed.
* Paired claim shapes: D ∈ {64, 128} × H ∈ {16, 256, 1024} × LSTM/sLSTM × FP16/BF16, B = 16, T = 256 (T = 128 at
  H = 1024: sLSTM training at T = 256 exceeds the memory estimate, 66 GB at D = 128); forward
  (one vs loop_graph, one vs v1) and training step (one vs loop, one vs v1); 3 process starts × 20 ABBA/BAAB blocks,
  each block entry the mean of 3 calls; outputs of every timed call must equal the arm's first call bitwise.
* BF16 training at H = 1 against FlashRNN1 (xLSTM was trained in BF16): D ∈ {192, 384, 768} × B ∈ {16, 64} ×
  LSTM/sLSTM, one vs v1, same paired protocol.

## 4. Checks

* **H1 qualification**: every case of a path passes (section 2).
* **H2 one launch vs the captured per-head loop** (forward, claim shapes): the ratio loop_graph / one with a 95%
  hierarchical bootstrap interval (10,000 draws over starts, then blocks); the claim holds for a shape when the
  lower bound is ≥ 1.
* **H3 against FlashRNN1** (forward and training, claim shapes): ratio v1 / one, same interval and rule.
* **H4 BF16 training at H = 1 against FlashRNN1**: ratio v1 / one per shape, same rule.
* **H5 (reported, not a check)**: per-call time against H in the sweep. Expected shape: roughly flat while
  ⌈B/16⌉·H CTAs (D ≤ 128) or groups (D > 128) fit one wave of resident CTAs, then growing in proportion to the
  number of waves.

Negative results and failed cases are kept. All timings use the untuned default for H > 1; a tuned H > 1 table is a
later step.

## 5. First run (job "heads v1", 2026-10-17) and amendment

**Repro (diagnostic).** All 14 arms of the scale-out phase A many-heads row ran cleanly in their own processes with
`CUDA_LAUNCH_BLOCKING=1` (LSTM/sLSTM, H = 1 and 4, default forward, TileLang forward, training step), and both
memcheck runs were clean. The phase A failure did not reproduce; it is attributed to that harness (CUDA events
recorded on one device and synchronized on another), not to a kernel.

**H1 as pre-registered: FAIL on every path** (single CTA 18/48, R-tiled 0/32, TileLang cluster 0/12, grid 0/8). In
every case the forward was bitwise equal per head and within the FP64 budget (relative error 2.6e-5 to 2.3e-4), and
every training case was within the FP64 gradient budget. What failed was the per-head bitwise equality of d_wx (and
of d_init, dR, dbias). Cause, confirmed on the CPU with the library functions: d_wx is not a kernel-only output. The
backward recomputes the gate preactivations with a GEMM outside the kernel, which uses `addmm` for H = 1 and
`einsum` plus a separate bias add for H > 1; the two summation orders round differently, so the kernel receives
different FP32 inputs. The same holds for the dR / dbias GEMMs. The pre-registered rule was wrong about provenance;
the kernels were not tested by it. Because H1 failed, H2, H3 and the sweep did not run.

**H4 (BF16 training at H = 1 against FlashRNN1) ran as pre-registered** (3 starts × 20 blocks, all outputs stable):
v1 / FlashRNN2 per-step ratio, 95% interval.

| cell | D | B = 16 | B = 64 |
| --- | --- | --- | --- |
| LSTM | 192 | 1.229 [1.226, 1.231] | 1.938 [1.816, 1.985] |
| LSTM | 384 | 0.977 [0.768, 1.081] (not shown) | 1.724 [1.685, 1.743] |
| LSTM | 768 | 1.197 [1.196, 1.198] | 1.629 [1.511, 1.682] |
| sLSTM | 192 | 1.182 [1.104, 1.213] | 2.023 [1.982, 2.040] |
| sLSTM | 384 | 1.003 [0.746, 1.140] (not shown) | 1.887 [1.885, 1.889] |
| sLSTM | 768 | 1.188 [1.187, 1.190] | 1.666 [1.664, 1.667] |

10 of 12 shapes hold (lower bound ≥ 1). D = 384 at B = 16 has wide intervals in both cells (start-to-start spread).

**Amendment for the second run (written before it).** Training qualification has two parts:
(a) library path: every gradient within the FP64 gradient budget;
(b) kernel isolation: `heads_qualify.per_head_gemms()` makes the H-head call compute both GEMMs head by head
through the H = 1 code path (verified on the CPU: per-head preactivations, dR and dbias bitwise equal to the H = 1
calls; the library path's are not). With it, d_wx, d_init, dR and dbias must be bitwise equal per head.
Forward rules, timing protocol and checks H2–H5 are unchanged; the repro phase is dropped (done). H4 is not rerun.

## 6. Second run (job "heads v2", 2026-10-17)

**H1**: every path qualifies — single CTA 48/48, Triton R-tiled 32/32, TileLang cluster 12/12, TileLang grid 8/8 (forward per-head bitwise and FP64; training: library-path gradients within the FP64 budget and, with the per-head GEMM hook, d_wx / d_init / dR / dbias bitwise per head).

**H2** (forward, captured per-head loop / one launch): 24/24 hold. **H3** (FlashRNN1 cuda_fused / one launch): forward 24/24 hold, training 0/24. Training against the per-head loop (reported, same protocol): 24/24 faster.

| cell | dtype | D | H | H2 forward vs loop (graph) | H3 forward vs FlashRNN1 | training vs loop | H3 training vs FlashRNN1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | bflo | 64 | 16 | 12.94 [12.87, 13.01] | 1.22 [1.20, 1.23] | 8.80 [8.78, 8.81] | 0.94 [0.85, 1.12] (no) |
| lstm | bflo | 64 | 256 | 178.53 [177.49, 179.44] | 4.62 [4.61, 4.63] | 20.75 [20.73, 20.77] | 0.50 [0.50, 0.50] (no) |
| lstm | bflo | 64 | 1024 | 263.28 [262.17, 264.30] | 6.67 [6.65, 6.68] | 22.06 [22.04, 22.08] | 0.50 [0.50, 0.50] (no) |
| lstm | bflo | 128 | 16 | 14.68 [14.62, 14.73] | 1.12 [1.11, 1.13] | 8.18 [8.17, 8.19] | 0.76 [0.70, 0.89] (no) |
| lstm | bflo | 128 | 256 | 102.51 [102.29, 102.70] | 3.94 [3.93, 3.94] | 13.56 [13.55, 13.57] | 0.57 [0.57, 0.57] (no) |
| lstm | bflo | 128 | 1024 | 109.88 [109.71, 110.02] | 4.16 [4.16, 4.16] | 13.32 [13.17, 13.40] | 0.54 [0.54, 0.54] (no) |
| lstm | floa | 64 | 16 | 12.38 [12.30, 12.44] | 1.09 [1.07, 1.10] | 8.79 [8.76, 8.81] | 0.91 [0.81, 1.09] (no) |
| lstm | floa | 64 | 256 | 178.79 [177.74, 179.67] | 4.06 [4.05, 4.07] | 20.78 [20.75, 20.81] | 0.47 [0.47, 0.47] (no) |
| lstm | floa | 64 | 1024 | 257.93 [256.89, 258.93] | 5.73 [5.72, 5.75] | 21.99 [21.94, 22.07] | 0.47 [0.47, 0.47] (no) |
| lstm | floa | 128 | 16 | 14.68 [14.61, 14.73] | 1.05 [1.04, 1.06] | 7.71 [7.32, 7.85] | 0.77 [0.70, 0.91] (no) |
| lstm | floa | 128 | 256 | 106.33 [106.01, 106.59] | 3.73 [3.73, 3.74] | 12.47 [12.32, 12.54] | 0.58 [0.57, 0.58] (no) |
| lstm | floa | 128 | 1024 | 109.51 [109.30, 109.70] | 3.80 [3.80, 3.81] | 12.38 [12.36, 12.40] | 0.55 [0.55, 0.55] (no) |
| slstm | bflo | 64 | 16 | 13.87 [13.78, 13.94] | 1.20 [1.19, 1.22] | 9.05 [9.04, 9.06] | 0.93 [0.85, 1.10] (no) |
| slstm | bflo | 64 | 256 | 183.39 [182.45, 184.28] | 4.03 [4.02, 4.04] | 22.03 [22.03, 22.04] | 0.54 [0.54, 0.54] (no) |
| slstm | bflo | 64 | 1024 | 230.61 [229.88, 231.30] | 5.02 [5.01, 5.03] | 23.32 [23.26, 23.39] | 0.54 [0.54, 0.54] (no) |
| slstm | bflo | 128 | 16 | 14.44 [14.37, 14.49] | 1.26 [1.16, 1.52] | 8.25 [8.25, 8.25] | 0.81 [0.75, 0.93] (no) |
| slstm | bflo | 128 | 256 | 106.46 [106.21, 106.69] | 4.02 [4.02, 4.03] | 13.51 [13.49, 13.52] | 0.67 [0.67, 0.68] (no) |
| slstm | bflo | 128 | 1024 | 108.73 [108.49, 108.95] | 4.06 [4.05, 4.06] | 13.06 [13.05, 13.06] | 0.65 [0.65, 0.65] (no) |
| slstm | floa | 64 | 16 | 13.78 [13.69, 13.86] | 1.24 [1.23, 1.26] | 9.05 [9.03, 9.06] | 0.87 [0.79, 1.03] (no) |
| slstm | floa | 64 | 256 | 182.35 [181.35, 183.23] | 4.35 [4.29, 4.51] | 21.99 [21.97, 22.01] | 0.54 [0.54, 0.54] (no) |
| slstm | floa | 64 | 1024 | 229.15 [228.37, 229.84] | 5.34 [5.32, 5.39] | 23.21 [23.19, 23.23] | 0.54 [0.54, 0.54] (no) |
| slstm | floa | 128 | 16 | 14.43 [14.36, 14.48] | 1.17 [1.16, 1.18] | 8.13 [8.12, 8.14] | 0.77 [0.70, 0.91] (no) |
| slstm | floa | 128 | 256 | 103.23 [102.93, 103.51] | 3.89 [3.88, 3.89] | 13.26 [13.24, 13.27] | 0.64 [0.64, 0.64] (no) |
| slstm | floa | 128 | 1024 | 108.51 [108.33, 108.67] | 4.02 [4.02, 4.03] | 12.84 [12.82, 12.87] | 0.62 [0.62, 0.62] (no) |

Reading (post hoc): one launch over all heads beats the captured per-head loop by 12-263x in the forward and 8-23x in
training, and beats FlashRNN1 in the forward (1.05x at H = 16 up to 6.7x at H = 1024). The training step loses to
FlashRNN1 at H > 1 (0.47-0.94x; at H = 256/1024 about 0.47-0.67x). The H > 1 training path recomputes the
preactivations and forms dR / dbias with einsum (H = 1 uses addmm) and runs the untuned default backward; where the
time goes is not measured here and is the next profiling target.


## 7. Profiling the H > 1 training step (diagnostic batch "heads prof", 2026-10-17, report only)

Section 6 found the H > 1 training step 0.47-0.94x of FlashRNN1 and growing linearly in H from H = 16. This batch
measures where the time goes; it has no pass/fail checks. Per shape (D ∈ {64, 128}, H ∈ {16, 256, 1024}, LSTM FP16;
sLSTM BF16 at D = 128, H ∈ {256, 1024}; B = 16, T = 256, 128 at H = 1024): the training forward alone (full FP32
history), and for four backward variants (default; TF32 GEMMs; `fast` + TF32; `fast` + TF32 + preactivations fused
into the kernel) the step time and the backward's phases (preactivation recompute, kernel, dR/dbias GEMMs), next to
FlashRNN1's step; Nsight Systems kernel timelines of FlashRNN2 and FlashRNN1 at D = 64, H = 1024 and D = 128, H = 256.

### 7.1 Profiling results (job "heads prof", 2026-10-17)

Training step, LSTM FP16, B = 16 (breakdown in `heads-evidence/hprof-breakdown.json`):

| D | H | FlashRNN2 step | FlashRNN1 step | forward | preactivation | backward kernel | dR / dbias | fused recompute: step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 16 | 3.1 ms | 2.7 ms | 0.9 | 0.43 | 1.16 | 0.33 | 2.9 |
| 64 | 256 | 21.2 | 10.0 | 1.4 | 6.5 | 4.4 | 4.8 | 15.7 |
| 64 | 1024 (T 128) | 41.7 | 19.5 | 2.1 | 13.0 | 8.9 | 9.1 | 30.6 |
| 128 | 256 | 43.5 | 25.3 | 3.0 | 13.3 | 9.7 | 9.3 | 32.4 |
| 128 | 1024 (T 128) | 93.2 | 51.8 | 5.8 | 26.8 | 18.8 | 24.8 | 70.5 |

TF32 changes nothing (the H > 1 GEMMs are memory-bound). Nsight Systems at D = 64, H = 1024, per call: FlashRNN2's
recurrent kernels take 8.9 ms (backward) + 1.8 ms (forward) against FlashRNN1's 9.9 + 5.8 ms, but FlashRNN2 spends about
25 ms in PyTorch elementwise / copy kernels (the einsum operands and outputs permuted to and from [B, T, 4, H, D] in
FP32, the preactivation sum, dtype casts) against FlashRNN1's about 6 ms, plus 3.8 ms of FP32 GEMMs against 1 ms of FP16.
The cost is outside the kernels.

## 8. H > 1 training without the FP32 preactivation tensor (pre-registration, 2026-10-17, before any run of it)

Changes: (1) for H > 1 the training path defaults to the fused kernel variant (`fast` and `fuse_pre = 2`: the gate
preactivations are recomputed inside the backward kernel, qualified at H = 1 in earlier rounds) — no FP32
[B, T, 4, H, D] preactivation tensor and none of its permuted copies; (2) dR for H > 1 is one strided batched GEMM per
gate on views of dG and h (`weight_gradients(method="bmm")`, no permuted copy of dG; on the CPU equal to the einsum
path bitwise and within 1e-7 of FP64). The earlier path stays selectable (`legacy_heads`, `wgrad="einsum"`).

Runs (batch "heads v3"):
* **H1-v3**: the training part of section 2's qualification with the new defaults passed explicitly to the H-head and
  the H = 1 runs (so the per-head bitwise comparison compares the same kernel variant), on the single-CTA and Triton
  R-tiled paths at H ∈ {2, 3, 16} (training is not served by TileLang at H > 1).
* **H3-train-v3**: training step at the claim shapes of section 3 (D ∈ {64, 128} × H ∈ {16, 256, 1024} × LSTM/sLSTM ×
  FP16/BF16) against FlashRNN1, and **H6** against the previous H > 1 path (unfused, einsum dR); protocol and rule of
  section 4.

Run "heads v3" stopped at its canary after 31 s: the test hook `per_head_gemms` replaced `weight_gradients` with a
function that did not take the new `method` argument (a harness defect; nothing ran on the GPU beyond that case). The
hook is fixed and the login preflight now calls the library's `weight_gradients` through the hook on CPU tensors; the
batch is rerun unchanged otherwise ("heads v4").

### 8.1 Results (job "heads v4", 2026-10-17)

**H1-v3**: single-CTA path 24/24 (H ∈ {2, 3, 16}, D = 64 / 128, LSTM/sLSTM, FP16/BF16: FP64 budgets and per-head
bitwise); Triton R-tiled path 12/24: every D = 256 case passes, every D = 768 case fails before running with a Triton
`OutOfResources` (the fused-recompute kernel needs 320 KB at H ∈ {2, 16} and 416 KB at H = 3 of shared memory, the
limit is 227 KB). It is a configuration error, not a numerical one: the fused variant had been qualified at H = 1 only
where it fits. Fix (after the run): the H > 1 default applies up to D = 256 (`training.FUSED_HEADS_MAX_WIDTH`); above
it the earlier, qualified H > 1 default stays (the strided-bmm dR is kept; it is bitwise equal to einsum on the CPU).
The claim shapes (D = 64 / 128) run on the single-CTA path, which qualified, so the timing below stands.

**H6** (new H > 1 training path / previous one) holds 20/24, **H3-train-v3** (FlashRNN1 / FlashRNN2) holds 4/24.
Training step, B = 16, T = 256 (128 at H = 1024):

| cell | dtype | D | H | vs previous H > 1 path (H6) | vs FlashRNN1 (H3-train-v3) |
| --- | --- | --- | --- | --- | --- |
| lstm | bflo | 64 | 16 | 1.04 [0.89, 1.14] (no) | 0.98 [0.98, 0.98] (no) |
| lstm | bflo | 64 | 256 | 1.56 [1.50, 1.60] | 0.79 [0.79, 0.79] (no) |
| lstm | bflo | 64 | 1024 | 1.57 [1.54, 1.59] | 0.79 [0.79, 0.79] (no) |
| lstm | bflo | 128 | 16 | 1.17 [1.03, 1.26] | 0.88 [0.88, 0.88] (no) |
| lstm | bflo | 128 | 256 | 1.55 [1.51, 1.56] | 0.90 [0.90, 0.90] (no) |
| lstm | bflo | 128 | 1024 | 1.66 [1.64, 1.67] | 0.91 [0.91, 0.91] (no) |
| lstm | floa | 64 | 16 | 1.04 [0.89, 1.13] (no) | 0.92 [0.92, 0.92] (no) |
| lstm | floa | 64 | 256 | 1.54 [1.49, 1.59] | 0.74 [0.74, 0.74] (no) |
| lstm | floa | 64 | 1024 | 1.58 [1.54, 1.59] | 0.74 [0.74, 0.74] (no) |
| lstm | floa | 128 | 16 | 1.18 [1.04, 1.27] | 0.89 [0.89, 0.90] (no) |
| lstm | floa | 128 | 256 | 1.55 [1.51, 1.56] | 0.90 [0.90, 0.90] (no) |
| lstm | floa | 128 | 1024 | 1.67 [1.65, 1.67] | 0.92 [0.92, 0.92] (no) |
| slstm | bflo | 64 | 16 | 1.09 [0.94, 1.18] (no) | 1.00 [1.00, 1.00] |
| slstm | bflo | 64 | 256 | 1.56 [1.50, 1.59] | 0.85 [0.85, 0.85] (no) |
| slstm | bflo | 64 | 1024 | 1.58 [1.55, 1.60] | 0.86 [0.86, 0.86] (no) |
| slstm | bflo | 128 | 16 | 1.20 [1.07, 1.29] | 0.97 [0.96, 0.97] (no) |
| slstm | bflo | 128 | 256 | 1.54 [1.51, 1.55] | 1.05 [1.05, 1.05] |
| slstm | bflo | 128 | 1024 | 1.64 [1.63, 1.64] | 1.07 [1.07, 1.07] |
| slstm | floa | 64 | 16 | 1.08 [0.93, 1.18] (no) | 0.93 [0.93, 0.93] (no) |
| slstm | floa | 64 | 256 | 1.56 [1.50, 1.59] | 0.86 [0.86, 0.86] (no) |
| slstm | floa | 64 | 1024 | 1.58 [1.55, 1.59] | 0.86 [0.86, 0.86] (no) |
| slstm | floa | 128 | 16 | 1.20 [1.07, 1.30] | 0.90 [0.90, 0.90] (no) |
| slstm | floa | 128 | 256 | 1.54 [1.51, 1.56] | 0.99 [0.99, 0.99] (no) |
| slstm | floa | 128 | 1024 | 1.64 [1.63, 1.65] | 1.02 [1.01, 1.02] |

Reading (post hoc): removing the FP32 preactivation tensor and the permuted copies makes the H > 1 step 1.5-1.7x
faster at H ≥ 256 (at H = 16, where the copies were small, the intervals include 1). Against FlashRNN1 the gap shrinks
from 0.47-0.94x (section 6) to 0.74-1.07x, and sLSTM at D = 128, H ≥ 256 is now faster; LSTM at D = 64 stays the
furthest behind (0.74-0.79x). Where the remaining time goes is measured in the next diagnostic run (step breakdown and
Nsight Systems with the new default; max-width.md section 18 batch). Evidence: `heads-evidence/heads-v4-analysis.json`.

### 8.2 Where the remaining H > 1 time goes (DSMEM batch, report only)

LSTM FP16, B = 16, new default (fused recompute, strided-bmm dR), per training step:

| D | H | FlashRNN2 step | FlashRNN1 step | forward | preactivation | backward kernel | dR / dbias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 256 | 13.4 ms | 9.9 ms | 1.4 | 0.3 | 5.1 | 2.4 |
| 128 | 1024 (T 128) | 55.5 | 51.6 | 5.8 | 1.2 | 21.8 | 9.9 |

Nsight Systems at D = 64, H = 1024, per call:

| | backward recurrent kernel | forward recurrent kernel | other |
| --- | --- | --- | --- |
| FlashRNN2 | 10.3 ms | 1.8 ms | 5.4 ms of copies, 2.8 ms FP32 dR GEMMs, 1.4 ms dbias reduction |
| FlashRNN1 | 9.8 ms | 5.7 ms | 1 ms FP16 dR GEMM |

The recurrent kernels are at parity or faster. The remaining loss is the FP32 dR GEMMs on the FP32 dG and the dtype
copies around the kernels. The next change is an FP16-operand dR with FP32 accumulation (a precision change, so it
needs its own qualification) and dropping the copies; it is not done yet.

### 8.3 Low-precision dR operands for H > 1 (pre-registration, 2026-10-18, before any run of it)

**Change.** `weight_gradients(method="bmm_lp")`, opt-in as the training option `wgrad="bmm_lp"`. The dR GEMMs take
dG rounded to the input dtype and the stored h_{t-1}, which is already in the input dtype, so no FP32 copy of h is made.
Accumulation is in FP32, with PyTorch's reduced-precision split-K reductions switched off for the call. The result is
rounded to the input dtype, the dtype dR is returned in. dbias stays the FP32 sum of the FP32 dG. This is FlashRNN1's
arrangement.

CPU check against FP64 (B = 8, T = 16, D = 64): largest dR error relative to the largest |dR|:
* FP16: 4.9e-4;
* BF16: 3.0-3.5e-3;
* FP32 path: about 5e-7.

**Runs (batch "heads lp"):**
* **H1-lp**: the training qualification of section 8 (FP64 budgets, per-head bitwise through the test hook, which
  passes the method on) with `wgrad="bmm_lp"` on the single-CTA path, H ∈ {2, 3, 16}, D ∈ {64, 128}, LSTM/sLSTM,
  FP16/BF16 (24 cases); H = 1 at the same widths (8 cases).
* **H3-lp / H6-lp**, paired training step at the claim shapes of section 3 (D ∈ {64, 128} × H ∈ {16, 256, 1024} ×
  LSTM/sLSTM × FP16/BF16): against FlashRNN1 (H3-lp) and against the current default (H6-lp); protocol of section 4.
  Taken only if H1-lp passes.
* **Operator profile** (report only, `tools/heads_opprof.py`): the default and the low-precision step at D = 64,
  H = 256 and D = 128, H = 1024, LSTM FP16. It attributes the remaining copies of section 8.2 to operators.

**Prediction.** H1-lp passes. The step gains 5-15% at H ≥ 256, and H3 rows move toward 1 but most stay below it while
the copies remain.

### 8.4 Run "heads lp" (2026-10-18): stopped by a plan defect

H1-lp: 9/32. Every case is within the FP64 budgets (rel. Frobenius error about 3-4e-4 on every gradient). The H > 1
cases fail only the per-head bitwise comparison, and in every gradient, d_wx included, which dR cannot affect. The
plan passed `wgrad="bmm_lp"` alone, so the H-head calls took the H > 1 default kernel (fused recompute, section 8)
and the per-head H = 1 calls took the H = 1 default. Section 8 avoided exactly this by passing the fused options to
both. The timing was skipped because the path did not qualify. The batch is rerun with
`fast=True, fuse_pre=2, wgrad="bmm_lp"` passed explicitly ("heads lp v2"); nothing else changes.

### 8.5 dG stored in the input dtype (pre-registration, 2026-10-18; replaces the "heads lp v2" plan of 8.4, which was cancelled before it started)

**What the profile showed.** The operator profile of the first lp run (report only, D = 64, H = 256, LSTM FP16, one
step) puts the low-precision dR GEMMs at 0.75 ms against 1.41 ms in FP32. But `bmm_lp` cast dG to the input dtype a
second time (+1.17 ms), because the training path already casts the FP32 dG for the Wx gradient. Net, it was no
faster.

**Change.**
* `triton_rtiled_bw.backward(dg_dtype="input")`, opt-in and requiring `wgrad="bmm_lp"`, stores dG from the kernel in
  the input dtype.
* Every read of dG inside the kernel rounds it to the input dtype first (`_dh_from_dg`, `_verify_dg`). The recurrence
  and dG's values are therefore bitwise unchanged; only the FP32 copy no longer exists.
* The dR GEMMs use the stored dG directly.
* dbias is the FP32-accumulated sum of the input-dtype dG. That is a precision change, inside the FP64 budgets or not
  per the qualification.
* The training path's cast of dG becomes a no-op.

**Runs (batch "heads lp v3").** These are section 8.3's runs with backward options
`fast=True, fuse_pre=2, wgrad="bmm_lp", dg_dtype="input"`, passed explicitly to the H-head and the H = 1 calls:
* H1-lp: 32 cases;
* H3-lp and H6-lp: paired, 24 shapes each;
* the operator profile of the default and the new options.

Prediction: H1-lp passes. The step gains about 1.5-2.5 ms at D = 64, H = 256 and about 10% at D = 128, H = 1024, and
the H3 rows move toward 1.
