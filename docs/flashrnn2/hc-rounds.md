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

Running. It contains:

* ncu on five recurrence configurations at B = 1024, or nsys timelines if ncu
  cannot read counters;
* a screen of TileLang CTAs with two warpgroups (256 threads; registers drop to
  88–192 for the cluster variants), with bitwise twin checks against
  128 threads;
* paired timing of the round-1 best configuration served through a CUDA Graph
  of the whole forward vs eager.
