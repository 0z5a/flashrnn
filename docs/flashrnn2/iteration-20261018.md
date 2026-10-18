# FlashRNN2: iteration of 2026-10-17/18 (summary)

This page collects what changed in the library during this iteration and the numbers behind each change. Every
number comes from a pre-registered batch; the detailed sections are cited. Everything was measured on one node of four
H100 GPUs, with paired timing (3 process starts × 20 blocks) and a 95% hierarchical bootstrap.

## Defaults that changed

| path | old default | new default | evidence |
| --- | --- | --- | --- |
| Training backward, H = 1, D = 1024 / 1280 | Triton R-tiled | 2-D split CUDA kernel | max-width §29: 1.31-2.03x |
| Training backward, H = 1, D = 1536-2048 | 1-D CUDA kernel (R in registers + shared memory) | 2-D split CUDA kernel | max-width §21, §26: 1.28-1.91x |
| Forward, 1024 ≤ D ≤ 1280 (B ≤ 32 at D = 1280) | Triton R-tiled (unmeasured) | TileLang grid kernel | max-width §31: 2.8-5.5x |
| Training, H > 1, D ≤ 256 | FP32 dG, FP32 dR GEMMs | dG stored in the input dtype, input-dtype dR GEMMs with FP32 accumulation | heads §8.6: 1.19-1.29x |
| Training backward, hidden output, Triton path | zero-filled full history gradient | h component only | heads §8.8: bitwise identical |
| Training backward, H = 1, D = 1024-2048, any B | TF32 cuBLAS preactivation GEMM + separate Wx add | fused input-dtype Triton GEMM with the Wx / bias add in its epilogue (`pre_fused`) | max-width §38, §42, §47: 1.12-1.15x at B = 64, 1.07-1.13x at B = 16; more accurate against FP64 |
| Training backward, H = 1, D = 1024-2048, any B | FP32 dG store + TF32 dR GEMM | dG stored in the input dtype by the kernel, input-dtype dR GEMM | max-width §40.1, §42, §47, §51: 1.09-1.19x |
| Training backward, H = 1, hidden output, D = 1024-2048, any B | zero-filled FP32 history gradient | kernel reads the h gradient itself (`honly`) | max-width §42, §47, §51: 1.03-1.13x, bitwise identical |
| Training backward, H > 1, D ∈ {1024, 1536, 2048}, B ≤ 64 | Triton (D ≤ 1280) / 1-D CUDA kernel (D > 1280) | 2-D split CUDA kernel | max-width §37, §42: 1.05-1.06x (D = 1024, B = 16), 1.45-1.73x elsewhere |

Combined, the three later H = 1 options make the step 1.13-1.56x faster at B = 64 and 1.07-1.14x at B = 16
(max-width §49). Measured with the defaults as of §47 / §49, the library's training step is faster than FlashRNN1
`cuda` at every measured H = 1 shape from D = 1024 to 2048: 1.11-1.61x at B = 64 and 1.64-3.07x at B = 16.
§51 later added dG_LP and `honly` at B < 64 and at D = 1280. Each of those held its own paired comparison, and they
held against `cuda` too (1.11-3.35x), but the full current default was not re-timed end to end at those shapes.
The combined defaults are qualified against FP64 at H = 1 and H = 2 (§47 D2, §49 D3, §53 D4).

With library defaults, an H-head training call is no longer bitwise equal per head to H = 1 calls wherever an H = 1-only
option applies. Both stay within the FP64 budgets (max-width §47).

## The 2-D split backward

The 1-D kernel had every CTA reduce the full K = 4D against the exchanged dG. Under Nsight Compute it ran at 59-61% of
L2 peak with one sector per request, so it was bound by the L2 request rate (max-width §21).

The 2-D kernel instead gives CTA (i, j) an R block (K block j × column block i). It reads only its dG slice with
staged 128-bit `cp.async` loads, writes an FP32 partial, and lets the column owner sum the partials in fixed order;
each step has two grid barriers. Measured effect:
* L2 sectors drop 6-7x and requests 10-16x.
* The kernel is 2.4-2.6x faster.
* The training step is at parity with FlashRNN1 `cuda` at D = 1536 / 1792 and 1.02-1.19x faster at D = 1024 / 2048
  (B = 64). At B ≤ 32 it is 1.09-2.91x faster.

It is race-verified (two broken-barrier variants each flagged 21/21) and qualified against FP64 at D = 1024-2048,
B = 16-100 (the kernel alone; for the later options at B > 64 see max-width §54).

Two tried alternatives were negative or small:
* A thread-block-cluster (DSMEM) exchange was correct but 1.3-1.45x slower (§19).
* Row ownership of the pointwise update gave +1-3% (§23).

## Multi-GPU (one node)

* *Forward, correctness.* The multi-GPU forward (direct, ring and recursive doubling schedules) qualifies on PyTorch,
  NCCL 2.32 symmetric and NVSHMEM 3.9 buffers: 510/510 calls, bitwise across P, schedules and allocations
  (scale-out §16, §22). Two defects were found and fixed on the way:
  * a missing per-GPU barrier in ring and doubling;
  * a harness that launched its reference on the wrong GPU.
* *Forward, cost model.* With a clean calibration and one system fence per publish in the direct schedule, the phase A
  α-β model predicts 34/37 configurations within ±20% and every schedule ordering (§22).
* *Forward against one GPU.* The forward is not faster than FlashRNN2's best single-GPU forward. At D = 2048 the TileLang
  kernel on one GPU is about 3x faster than four GPUs (§24).
* *Backward.* The multi-GPU backward (dG exchange; 1-D and 2-D kernels) is bitwise equal to one GPU and race-verified.
  * The multi-GPU 2-D kernel beats one GPU with the 2-D kernel only at D = 2048 with four GPUs (1.17x).
  * Elsewhere the 16-row tiles it needs make it slower (§18).

## H > 1 (many heads)

Against FlashRNN1 the H > 1 training step went from 0.47-0.94x (heads §6) to 0.94-1.49x (heads §8.6, §8.8):
* removing the FP32 preactivation tensor and its copies;
* fused recompute;
* input-dtype dG and dR operands;
* the h-only history gradient.

The batch "defaults" re-timed all 24 claim shapes with the shipped defaults: every shape now holds against
FlashRNN1, 1.05-1.48x, including LSTM at D = 64 (heads §8.10). The h-only history gradient alone gives 1.02-1.12x
there.

The 2-D backward also helps at H > 1 with large D (max-width §37):
* 1.05-1.06x over Triton at D = 1024;
* 1.45-1.49x over the 1-D kernel at D = 2048;
* 1.19-3.26x over FlashRNN1.

## Open

* Phase A check 2 still fails by its all-within rule (34/37).
* The multi-GPU forward kernel lacks the TileLang kernel's per-GPU efficiency.
* At H > 1 with B = 64, FlashRNN1 `cuda` is still faster than the 2-D path at D = 1024 and 2048 (0.78-0.98x, §42).
  The heads run one after another, because the resident R of one head fills every SM.
* Replacing grid barrier B1 by a column-group flag wait is correct but 1-9% slower (§44). The time moves into making
  the partials visible at GPU scope (§46). The two barriers still take 3-4.3 µs per step.
* The forward is now the largest single kernel at D = 2048, B = 64: 7.2 ms against 5.3 ms for the 2-D backward (§46).
  A 2-D split forward is the next candidate.
* The H = 1 options are not applied at H > 1, so H > 1 still uses the TF32 preactivation and FP32 dG.
