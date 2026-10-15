# R tiling across SMs: M0 feasibility record (H100, Triton 3.6)

Status: design decision, compile evidence and the M0 GPU qualification (run M0). Timings in this file are exploratory and unpaired; nothing here is
an E2E or speedup claim.

## Decision

The first R-tiled path (`flashrnn/flashrnn2/triton_rtiled.py`, family
`mma_rtiled`) is written in plain Triton 3.6 and uses an **in-kernel group
barrier over L2 with a cooperative launch**, not cluster + DSMEM.

* R is split along its gate/output axis into `k = ceil(D / S)` slices of a
  power-of-two width `S`. Each CTA keeps the four gate slices
  `R[g, head, s*S:(s+1)*S, :]` in shared memory for the whole time loop, plus
  its c/n/m slices in registers. Only the rounded hidden operand `h_t` crosses
  CTAs.
* `h_t` goes through a double-buffered global buffer (L2 resident). Exchange
  loads use `ld.global.cg` so stale L1 lines cannot be read.
* One barrier per step per (batch tile, head) group: `bar.sync`, then a single
  thread does `atom.global.gpu.release.add` and spins with
  `ld.global.gpu.acquire`, broadcasting through shared memory and `bar.sync`.
  The spin is bounded and sets an error flag instead of hanging.
* The persistent launch sets Triton's `launch_cooperative_grid`, so the driver
  guarantees co-residency or rejects the launch. The grid is further bounded to
  one CTA per SM; groups beyond that are processed serially per slot.
* `sync="launch"` replaces every in-kernel barrier by a kernel boundary (one
  launch per step through the FP32 compute carry). It performs identical
  arithmetic and is the bitwise oracle for the persistent path.
* For `D` that is not a power of two but is a sum of two powers of two
  (96, 192, 384, 768), the reduction is split into two exact parts (for example
  `512 + 256`) instead of padding K. Padding remains selectable (`pad_k`).

## Why not cluster + DSMEM in Triton 3.6

Read from the installed Triton 3.6.0 (`deps36`) on the cluster:

* `CUDAOptions.launch_cooperative_grid` exists and the NVIDIA launcher sets
  `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` through `cuLaunchKernelEx`.
* Clusters are only reachable through `num_ctas`: the launcher multiplies
  `gridDimX` by `num_ctas` and sets the cluster dimension, while the compiler
  distributes one program's tensors across the CTAs of the cluster (CGA
  layouts). There is no user-level distributed-shared-memory or cluster-barrier
  primitive in `triton.language` or in Gluon's Hopper module (only
  `fence_async_shared(cluster=...)`).

So per-step cluster barriers with explicit DSMEM exchange are not expressible
without inline PTX on shared addresses that Triton does not expose. This is
the reason the M3 fallback backend should be one that does expose them;
TileLang additionally exposes a cooperative `sync_grid()` for `k > 16`.

## Generated code (sm_90a, ptxas from Triton 3.6)

* Barrier PTX: `bar.sync` → `@tid0 atom.global.gpu.release.add.u32` →
  `@tid0 ld.global.gpu.acquire.b32` spin with shared-memory broadcast.
* Exchange loads: `ld.global.cg`.
* Recurrent MMA: `mma.sync` with `ldmatrix`; no `wgmma` is emitted.

## Compile-resource sweep (FP16, no launch)

`tools/rtiled_compile_sweep.py` on the cluster login node against the submitted
source (188 configurations, 0 failures, 140 within the H100 opt-in limit of
232448 shared bytes). Raw rows: `evidence/flashrnn2-rtiled/sweep-*.json`.
Shared bytes are Triton's dynamic allocation; registers and spill-store bytes
come from `ptxas -v`. The table lists the smallest feasible `k` per width with
exact (unpadded) K parts.

| Cell | D | K parts | S | k | Shared KiB | Regs / spill B, 4 warps | Regs / spill B, 8 warps |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| LSTM | 128 | 128 | 128 | 1 | 136 | 255 / 12 | 255 / 0 |
| LSTM | 192 | 128+64 | 128 | 2 | 200 | 255 / 3152 | 255 / 604 |
| LSTM | 256 | 256 | 64 | 4 | 138 | 255 / 4 | 254 / 0 |
| LSTM | 384 | 256+128 | 64 | 6 | 204 | 255 / 504 | 255 / 348 |
| LSTM | 512 | 512 | 32 | 16 | 145 | 255 / 0 | 255 / 4 |
| LSTM | 768 | 512+256 | 32 | 24 | 216 | 255 / 64 | 255 / 24 |
| sLSTM | 128 | 128 | 128 | 1 | 136 | 255 / 88 | 255 / 0 |
| sLSTM | 192 | 128+64 | 128 | 2 | 200 | 255 / 3776 | 255 / 674 |
| sLSTM | 256 | 256 | 64 | 4 | 138 | 255 / 44 | 255 / 28 |
| sLSTM | 384 | 256+128 | 64 | 6 | 204 | 255 / 624 | 255 / 384 |
| sLSTM | 512 | 512 | 32 | 16 | 145 | 255 / 244 | 255 / 8 |
| sLSTM | 768 | 512+256 | 32 | 24 | 216 | 255 / 16 | 255 / 16 |

The smallest feasible `k` matches the plan's estimate for 192 (2), 256 (4),
384 (6) and 512 (16); D=768 fits at k=24 with exact K parts. Padding K costs
shared memory: D=768, S=32 needs 289 KiB padded and does not fit. Larger
slices trade fewer CTAs for heavy spills (D=192, S=128); a smaller slice is
spill-free (D=192, S=64, k=3, 102 KiB). Which point is fastest is an E2/E3
question, not answered by compile resources.

## GPU batches

### Run M0 v1: failed at startup, no kernel executed

Module activation sets `CC=nvc`; Triton builds its launcher C modules with
`$CC`, and `nvc` rejects `-Wno-psabi`. Every canary raised before any GPU
kernel ran; the allocation lasted 62 s. The finalizer archived and verified
the negative outputs (`evidence/flashrnn2-rtiled/m0-v1/`).
Repair: the runtime exports `CC=/usr/bin/gcc CXX=/usr/bin/g++` after
activation; a login-node dry run now builds `cuda_utils` and the `_rtiled`
launcher (and reproduces the original error without the fix); a worker aborts
when its canary raises.

### Run M0: M0 qualification, all gates passed

One H100 node, four H100 lanes, 15 min 53 s. Same kernel source and
frozen 140-configuration plan as the failed job. Receipts and every stage JSON
are in `evidence/flashrnn2-rtiled/m0/`; the complete raw
archive `archives/m0.tar.gz` (182 files) stays on the cluster with an
independent re-hash check (`ARCHIVE_VERIFIED`).

| Stage | Scope | Result |
| --- | --- | --- |
| Canary (grid vs launch, D=128, k=2) | 16 width workers | 16/16 PASS |
| Resources (launch each frozen configuration) | 140 configurations, LSTM/sLSTM, D=64–768 | 140/140 launched, error flag 0, barrier counters exact |
| Protocol | every configuration × (B,T,H) ∈ {(1,33,1),(3,17,2),(33,40,2)} × FP16/BF16 × sLSTM zero/nonzero normalizer | 1260/1260 PASS |
| E1 numerics | `fixture_math` FP64 oracle, `CELL_BUDGET`, history/carry/hidden/last, (3,17,2) and (16,128,1), sLSTM zero/mixed/warm | 1120/1120 R-tiled and 48/48 single-CTA arms PASS; worst normalized error 0.0053 of budget |
| Barrier stress | widest 4-warp configuration per (cell, D), (16,256,1), (64,128,4), (128,64,8), 50 repeats | 48/48 PASS, bit-identical repeats, no timeouts |

Protocol PASS means, bitwise: persistent grid barrier = per-step kernel
boundary; serial groups per slot = one group per slot; ragged chunked
continuation through the FP32 carry = one call; hidden and last outputs = the
full-state output; and both modes leave the expected barrier counter values.

Overlap with the single-CTA path (FP32 snapshots, same inputs):

* D=64 and D=128: bitwise equal for every slice width, including k = 2, 4, 8.
* D=96: bitwise equal when K is padded to 128 (the single-CTA reduction
  shape); the exact two-part K (64+32) is not bitwise and passes under the
  budget instead.

### Exploratory timing (run M0, unpaired)

LSTM FP16, H=1, T=256, hidden output, CUDA events over 20 calls; each R-tiled
call includes its own buffer allocation and memsets, the single-CTA call does
not. Best configuration per point, microseconds per time step:

| D | B=16 | B=64 | best (S, warps, k) | single CTA B=16 / B=64 |
| ---: | ---: | ---: | --- | --- |
| 64 | 2.29 | 2.23 | 64, 8, 1 / 32, 4, 2 | 1.60 / 1.51 |
| 96 | 2.43 | 2.49 | 32, 4, 3 | 2.85 / 2.85 |
| 128 | 2.49 | 2.99 | 32, 4, 4 | 2.83 / 3.15 |
| 192 | 2.92 | 4.03 | 32, 4, 6 / 64, 8, 3 | n/a |
| 256 | 3.11 | 4.45 | 32, 4, 8 / 64, 8, 4 | n/a |
| 384 | 3.17 | 4.67 | 32, 4, 12 | n/a |
| 512 | 4.58 | 5.50 | 32, 4, 16 | n/a |
| 768 | 5.41 | 5.71 | 32, 4, 24 | n/a |

Kernel-boundary sync (one launch per step) costs about 18.5 µs per step, so
the in-kernel barrier is essential. At k=2 the barrier adds roughly
0.6–0.9 µs per step over the single CTA (D=64). At D=96 and D=128 the S=32,
k=3–4 split was faster than the single CTA in this unpaired screen; whether
the D ≤ 128 default should change is the E2 question and needs the paired
protocol. Padded K and slot-limited grids (for example D=768, S=16, B=64:
48 CTAs per group, two slots for four groups) are clearly slower.
