# R tiling across SMs: M0 feasibility record (H100, Triton 3.6)

Status: design decision and GPU-free compile evidence. GPU results are recorded
separately once the M0 batch has run; nothing here is an E2E or speedup claim.

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

Read from the installed Triton 3.6.0 (`deps36`) on MN5:

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

`tools/rtiled_compile_sweep.py` on the MN5 login node against the submitted
source (188 configurations, 0 failures, 140 within the H100 opt-in limit of
232448 shared bytes). Raw rows: `evidence/flashrnn2-rtiled-m0-20261014/sweep-*.json`.
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

## GPU batch

Submitted job `47297792` (name 58): one ACC node, four H100s used as four
independent single-GPU lanes, three hours, cluster-side start command and
finalizer through `offline_batch.py`. Frozen case plan: 140 configurations
(`control/configs.json`), stages canary → resources → protocol (grid vs
launch bitwise, serial groups, chunked continuation, hidden/last) → E1
(`fixture_math` FP64 oracle under `CELL_BUDGET`) → barrier stress →
exploratory timing. Receipts: `evidence/flashrnn2-rtiled-m0-20261014/control/`.
