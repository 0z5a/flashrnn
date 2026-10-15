# High-concurrency serving: data planes and rounds

Goal (user, 2026-10-14, after D = 768): treat placement and resource questions as
data planes (examples given: NUMA affinity, congestion, SM cost), iterate, and
stop when high-concurrency throughput improves a lot.

**Stop criterion, fixed before any round:** at C = 1024 for D = 256 and D = 768,
LSTM and sLSTM, throughput ≥ 2× max(s0, s0p) with P90 latency no worse than s0.
The other exit is that all planes have been tried.

Workload: the existing classifier (4 layers, vocab 512, FP16; `tools/rtiled_m4.Model`).
C requests of T = 128 tokens arrive together (closed batch). The two baselines
use the same public kernel path:

* **s0**: C/32 groups served one after another (the existing harness), each with
  the config the M6 + M7 tables serve for B = 32;
* **s0p**: all C requests in one call per layer, with the config the policy
  serves for B = C (`default_config`, since the tables have no such shape).

Tool: `tools/hc_bench.py`, with modes decompose, select, start, profile and forward.

## Round 0: decomposition (job 47306265, name 89)

One node, 3 min 40 s, all 8 tasks complete, archive verified (48 files).
Evidence: `evidence/flashrnn2-hc0-20261015/job47306265/`.

| C = 1024 | s0 makespan | s0p makespan | s0 P90 | recurrence share (s0p) | GPU busy (s0p) |
|---|---|---|---|---|---|
| LSTM D768 | 69.1 ms | 50.0 ms | 62.7 ms | 90% | 0.98 |
| sLSTM D768 | 68.4 ms | 50.2 ms | 62.0 ms | 90% | 0.98 |
| LSTM D256 | 44.3 ms | 6.2 ms | 40.1 ms | 75% | 0.83 |
| sLSTM D256 | 42.4 ms | 6.3 ms | 38.4 ms | 75% | 0.80 |

* Batching alone (s0 → s0p) gives 1.37× at D = 768 and 7× at D = 256. That gain
  belongs to the baseline. max(s0, s0p) = s0p at C = 1024 everywhere.
* The recurrence dominates. At D = 256 the input projection takes 15–17% and the
  GPU is idle 15–25% of the time.
* The recurrence microbench gives a nearly constant ~9–11 SM·µs per row-step at
  BM = 16 for every configuration. At D = 256, µs/step grows 1.5–1.9× from
  B = 512 to 1024 (close to saturation). At D = 768 it is linear (slot-bound).
* Repeating C = 1024 alone on one lane matched the four-lane run (e.g. 50.0 vs
  50.04 ms, 6.20 vs 6.21 ms), so concurrent lanes do not disturb these numbers.
* 16/16 strategy × case FP64-oracle checks pass. Recurrence rows are bitwise
  independent of batch composition for every configuration (26/26).

## Round 1: SM-cost plane, batch tile BM > 16

Rows per CTA per step (WGMMA N) went from 16 to 32/64/128 (commit 8b29f22).

**v1, job 47306925 (name 92): stopped at the gate.**
* The 28 BM = 32/64 configurations passed qualification (E1, protocol, stress,
  bitwise = BM 16).
* All four cluster BM = 128 configurations failed: nondeterministic, wrong
  values, NaN at larger B.
* The cause is in the generated code. At BM = 128 the WGMMA B operand hᵀ is
  stored as two swizzled 64-column atoms, so a CTA's S rows are not one
  contiguous range. The single bulk copy per peer sent the wrong bytes.
  TileLang 0.1.15's per-atom regions become per-row copies at swizzled
  addresses, so they cannot fix it.
* The cluster path is now limited to BM ≤ 64 (commit 8ebf7f7). The grid path is
  unaffected.
* The sanitizer case for that configuration also failed (bitwise sanity).
  Because the sanitizer gate then stopped the whole batch, v2 changes the rule:
  a sanitizer finding excludes only its configuration.

**v2, job 47307383 (name 95):**
* 28/28 qualified; memcheck/synccheck clean on cluster D256 S16 BM64 and grid
  D768 S16 BM64.
* Split-0 selection among s0, s0p, every BM = 16 configuration at B = C and the
  qualified BM > 16 configurations. Split-1 paired timing, 5 starts × 20 blocks.
* 100/100 starts; archive verified (540 files).
* Evidence: `evidence/flashrnn2-hc1-20261015/job47307383-v2/`.

| Case | Best (split 0) | vs s0p | vs best BM = 16 | P90 best / s0 |
|---|---|---|---|---|
| LSTM D256 C256 | TL cluster S16 BM16 | 1.382 [1.380, 1.385] | 1.000 (same config) | 2.1 ms / – |
| sLSTM D256 C256 | TL cluster S16 BM16 | 1.449 [1.429, 1.457] | 1.000 (same) | 2.1 ms / – |
| LSTM D256 C1024 | s0p default (Triton S64 w8) | 1.000 (same) | 0.999 (same) | 6.3 / 39.6 ms |
| sLSTM D256 C1024 | Triton S64 w4 | 1.011 [1.010, 1.012] | 0.999 (same) | 6.3 / 38.2 ms |
| LSTM D768 C256 | TL grid S16 BM32 | 1.175 [1.172, 1.178] | 1.011 [1.008, 1.014] | 12.8 ms / – |
| sLSTM D768 C256 | TL grid S16 BM64 | 1.311 [1.309, 1.312] | 1.148 [1.146, 1.151] | 11.4 ms / – |
| LSTM D768 C1024 | Triton S32 w4 | 1.156 [1.155, 1.156] | 1.000 (same) | 43.4 / 61.9 ms |
| sLSTM D768 C1024 | Triton S32 w4 | 1.150 [1.149, 1.151] | 1.000 (same) | 43.7 / 60.6 ms |

* **The SM-cost plane via BM does not move C = 1024.** The best configuration
  there has BM = 16 (rows marked "same" are built-in A/A controls at 1.000).
* At C = 256 the gains (1.18–1.45×) come mostly from the configuration choice at
  B = C. A BM > 16 tile adds 1.15× only for sLSTM D768 C256.
* A one-wave tile is not faster. At LSTM D256 C1024, S32/BM64 (one wave of 128
  CTAs) took 11.7 ms in screening, against 6.4–6.5 ms for the two-wave BM = 16
  configurations.
* ptxas shows why. The BM ≥ 32 variants use 211–255 registers per thread, and
  the grid BM 32/64 variants spill. Per-thread state (c, n, m, h and the gates
  fragment) grows with BM × S / 128 threads, and so does each step's serial
  epilogue work.

## Round 2: profiling, occupancy plane, launch plane (job 47308040, name 98)

One node, 4 min 55 s, `ALL_STARTS_COMPLETED` (40/40); archive verified (210 files).
Evidence: `evidence/flashrnn2-hc2-20261015/job47308040/`.

**ncu** (counters permitted; one steady-state launch, B = 1024, T = 128, LSTM):

| Config | Duration | SM % | Memory % | No eligible | Warps/scheduler | Regs | Occupancy | Long-scoreboard share of stalls |
|---|---|---|---|---|---|---|---|---|
| D256 Triton S64 w8 (default) | 1.23 ms | 15.3 | 22.9 | 83.7% | 2.0 | 175 | 12.5% | – |
| D256 TL cluster S16 BM16 | 1.28 ms | 21.4 | 42.0 | 71.8% | 3.4 | 121 | 22.7% | 51.6% |
| D256 TL cluster S32 BM64 | 2.63 ms | 6.6 | 14.9 | 86.4% | 1.0 | 211 | 6.3% | – |
| D768 Triton S32 w4 | 9.46 ms | 10.0 | 25.7 | 89.7% | 1.0 | 198 | 6.2% | 40.3% |
| D768 TL grid S16 BM64 | 13.15 ms | 4.9 | 17.9 | 93.2% | 1.0 | 255 | 6.3% | 54.6% |

Every kernel is latency-bound: neither the SM nor memory is near peak, and most
cycles have no eligible warp. The largest stall is the long scoreboard. In
every step's epilogue, each thread loads its input projections (`WX`) from
global memory, and that load sits on the step's critical path.

**Occupancy screen** (two warpgroups, 256 threads, split-0 makespans; screened,
not qualified):
* On the grid path the 256-thread twins are bitwise equal to the 128-thread
  kernels (8/8).
* Grid S16/BM64 with 256 threads gives 34.5 / 33.9 ms (LSTM / sLSTM) at
  D768 C1024, against 43.6 / 43.5 ms for the round-1 best. At C256 it gives
  8.75 / 8.70 ms against 12.9 / 11.3 ms.
* On the cluster path the 256-thread kernels were wrong: 0/12 bitwise twins and
  3 qualification failures. They were also not faster. Two warpgroups are now
  allowed only on the grid path with BM ≥ 32 (commit 64d870c).

**Launch plane** (whole 4-layer forward in a CUDA Graph vs eager, the round-1
best configuration, paired):

| Case | Speedup [95% CI] |
|---|---|
| LSTM D256 C256 / C1024 | 1.033 [1.031, 1.036] / 0.994 [0.988, 1.002] |
| sLSTM D256 C256 / C1024 | 1.044 [1.043, 1.046] / 1.004 [1.003, 1.005] |
| LSTM D768 C256 / C1024 | 1.007 [1.004, 1.009] / 1.002 [1.001, 1.003] |
| sLSTM D768 C256 / C1024 | 0.997 [0.996, 0.998] / 0.998 [0.998, 0.999] |

Graph logits are bitwise equal to eager in all 8 cases. The launch plane is
null at C = 1024; the idle time seen at D = 256 in round 0 is not launch
overhead.

## Round 3: latency plane (input-projection prefetch) and two-warpgroup grid CTAs (job 47308340, name 100)

`prefetch=True` loads the next step's projections into registers right after
each epilogue, so the load overlaps the exchange wait and the GEMM. The sums are
the same, so results must be bitwise equal to the reference kernel.

The qualifications cover the four cluster-prefetch variants at D = 256 and the
256-thread grid S16 BM32/BM64 variants (with and without prefetch) at D = 768.
The new `vs_ref` stage requires bitwise equality with the reference kernel
(BM 16, 128 threads, no prefetch). Then come split-0 selection and split-1
paired timing against s0p, the round-1 best, and s0 at C = 1024.
