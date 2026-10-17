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
