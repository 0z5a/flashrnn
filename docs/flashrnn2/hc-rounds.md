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

## Round 0: decomposition (run HC0)

One node, 3 min 40 s, all 8 tasks complete, archive verified (48 files).
Evidence: `evidence/flashrnn2-rtiled/hc0/`.

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

Rows per CTA per step (WGMMA N) went from 16 to 32/64/128 (development commit 8b29f22).

**v1, run HC1 v1: stopped at the gate.**
* The 28 BM = 32/64 configurations passed qualification (E1, protocol, stress,
  bitwise = BM 16).
* All four cluster BM = 128 configurations failed: nondeterministic, wrong
  values, NaN at larger B.
* The cause is in the generated code. At BM = 128 the WGMMA B operand hᵀ is
  stored as two swizzled 64-column atoms, so a CTA's S rows are not one
  contiguous range. The single bulk copy per peer sent the wrong bytes.
  TileLang 0.1.15's per-atom regions become per-row copies at swizzled
  addresses, so they cannot fix it.
* The cluster path is now limited to BM ≤ 64 (development commit 8ebf7f7). The grid path is
  unaffected.
* The sanitizer case for that configuration also failed (bitwise sanity).
  Because the sanitizer gate then stopped the whole batch, v2 changes the rule:
  a sanitizer finding excludes only its configuration.

**v2, run HC1 v2:**
* 28/28 qualified; memcheck/synccheck clean on cluster D256 S16 BM64 and grid
  D768 S16 BM64.
* Split-0 selection among s0, s0p, every BM = 16 configuration at B = C and the
  qualified BM > 16 configurations. Split-1 paired timing, 5 starts × 20 blocks.
* 100/100 starts; archive verified (540 files).
* Evidence: `evidence/flashrnn2-rtiled/hc1-v2/`.

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

## Round 2: profiling, occupancy plane, launch plane (run HC2)

One node, 4 min 55 s, `ALL_STARTS_COMPLETED` (40/40); archive verified (210 files).
Evidence: `evidence/flashrnn2-rtiled/hc2/`.

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
  allowed only on the grid path with BM ≥ 32 (development commit 64d870c).

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

## Round 3: latency plane (input-projection prefetch) and two-warpgroup grid CTAs

`prefetch=True` loads the next step's projections into registers right after
each epilogue. The intent is to overlap that load with the exchange wait and the
GEMM; the sums are the same.

* **v1, run HC3 v1:** stopped 19 s after start by a driver bug.
  `Path / f"…" + "-pf"` binds as `(Path / f"…") + "-pf"`. No GPU work ran; the
  failure is archived.
* **v2, run HC3 v2:** one node, 20 min 27 s, 100/100 starts;
  archive verified (553 files). Evidence:
  `evidence/flashrnn2-rtiled/hc3-v2/`.

Qualification: 32/32 configurations PASS. They are the four cluster prefetch
variants at D = 256 and the 256-thread grid S16 BM32/BM64 variants at D = 768,
with and without prefetch, each for LSTM/sLSTM × FP16/BF16. The checks were E1,
protocol, stress, and `vs_ref`: bitwise equality with the reference kernel (BM 16,
128 threads, no prefetch). memcheck/synccheck were clean on cluster D256 S16
BM16 + prefetch and on grid D768 S16 BM64 256 threads + prefetch.

**Prefetch is slower everywhere** (split-0 screen):

| Case | Without | With prefetch |
|---|---|---|
| LSTM D256 C256, cluster S16 BM16 | 2.10 ms | 2.56 ms |
| sLSTM D256 C256 | 2.07 ms | 2.47 ms |
| LSTM D768 C1024, grid S16 BM64, 256 threads | 34.4 ms | 43.1 ms |
| sLSTM D768 C1024 | 34.0 ms | 42.7 ms |

ptxas on the same kernels (prefetch identified by the use of its fragments)
explains part of this:

| Kernel | Registers / spill stores, without → with prefetch | Consequence |
|---|---|---|
| cluster S16 BM16, 128 threads | 121 / 0 → 130 / 0 | allocated 128 → 136 per thread: 16,384 → 17,408 per CTA, so 4 → 3 CTAs per SM |
| cluster S16 BM32, 128 threads | 157 / 0 → 167 / 0 | 3 CTAs per SM before and after |
| grid S16 BM32, 256 threads | 255 / 12 B → 255 / 56 B | more spilling |
| grid S16 BM64, 256 threads | 255 / 0 → 255 / 0 | none (one CTA per SM either way) |

* At D = 256 the occupancy cliff from 4 to 3 CTAs per SM fits the 1.22× slowdown.
* For the grid BM64 kernel, registers and spills do not change, so the cause of
  its 1.25× slowdown is not established. One possibility, unverified, is that
  the loads issued before the grid barrier's memory fence must complete there.
* The best D = 256 kernel is capped at 4 CTAs per SM by both registers
  (≤ 128 per thread) and shared memory (53 KB). Any rewrite has to stay inside
  that budget.

Recorded as a negative result; the option stays off.

**Paired timing** (split 1, 5 starts × 20 ABBA/BAAB blocks):

| Case | Best | vs s0p | vs round-1 best | vs s0 | P90 best / s0 |
|---|---|---|---|---|---|
| LSTM D256 C256 | TL cluster S16 BM16 | 1.386 [1.382, 1.395] | 1.000 | – | 2.11 ms |
| sLSTM D256 C256 | TL cluster S16 BM16 | 1.454 [1.450, 1.457] | 1.000 | – | 2.08 ms |
| LSTM D256 C1024 | round-1 best (Triton S64 w8) | 0.999 [0.998, 1.000] | 1.003 | 7.03 | 6.20 / 39.5 ms |
| sLSTM D256 C1024 | Triton S64 w4 | 1.011 [1.010, 1.012] | 1.000 | 6.63 | 6.31 / 37.9 ms |
| LSTM D768 C256 | TL grid S16 BM64, 256 threads | 1.737 [1.731, 1.744] | 1.466 | – | 8.76 ms |
| sLSTM D768 C256 | same | 1.750 [1.744, 1.754] | 1.307 | – | 8.68 ms |
| LSTM D768 C1024 | same | **1.455** [1.454, 1.456] | 1.266 | 1.975 | 34.4 / 61.5 ms |
| sLSTM D768 C1024 | same | **1.482** [1.481, 1.483] | 1.288 | 1.970 | 34.0 / 60.6 ms |

## Where the goal stands after round 3

The stop criterion is not met, and not every plane has been tried (congestion
and NUMA have no experiment yet). At C = 1024 the best is 1.46–1.48× over s0p for
D = 768 and 1.00–1.01× for D = 256. Against s0 alone, D = 768 reaches
1.97–1.98× and D = 256 reaches 6.6–7.0×.

| Plane | Lever | Result at C = 1024 |
|---|---|---|
| SM cost | batch tile BM > 16 | null (best at BM 16) |
| Occupancy | two warpgroups per CTA (grid path) | **+1.27–1.29×** on D = 768; cluster path wrong, so disallowed |
| Launch | CUDA Graph of the whole forward | null (0.99–1.00×) |
| Latency | input-projection prefetch | negative (0.8×): D = 256 loses a CTA per SM to registers |
| Congestion | L2 / memory bandwidth | not run as an experiment; single-kernel ncu at one stream shows L2 ≤ 18% and DRAM ≤ 17% |
| NUMA | host placement, whole node (round 4) | null: local vs remote 1.0004–1.0024 |

What remains is structural. Every measured kernel is latency-bound at one CTA per
SM (6–25% occupancy) with the per-step chain on the critical path. A large gain at
D = 256 would need one of two things:

* a kernel that keeps the gate epilogue in registers (no FP32 stage round trip)
  and runs more warps per SM;
* a weight-stationary layout that streams hᵀ in K-chunks and amortizes each
  resident R slice over many more rows.

Either is multi-day kernel work.

## Round 4: structural ablation ceilings and the whole-node NUMA plane (run HC4)

One node, 8 min 57 s, `ALL_COMPLETED`; archive verified (438 files).
Evidence: `evidence/flashrnn2-rtiled/hc4/` (task logs left on the cluster).

### A. Ablation ceilings (timing only; development commit 3309c04)

`tilelang_cluster` takes a private `_ablate` set that removes one component of
the per-step chain. Ablated results are wrong by design. The switch cannot be
reached through `KernelConfig` or `ExecutionPolicy`. Without it, the generated
CUDA is identical to the qualified kernels.

Each entry is the recurrence-call speedup if that component cost nothing. The
two values are LSTM / sLSTM; the medians come from 15 interleaved repetitions.

| Removed | D256 TL cluster S16 BM16, B = 1024 (1.33 / 1.30 ms) | D768 TL grid S16 BM64, 256 threads, B = 1024 (7.19 / 7.04 ms) |
|---|---|---|
| h exchange + its synchronization | 1.15 / 1.42 | **1.61 / 1.74** |
| WGMMA | 1.32 / 1.34 | 1.17 / 1.18 |
| FP32 gate stage round trip | 1.16 / 1.15 | 1.10 / 1.12 |
| input-projection loads | 1.15 / 1.13 | 1.14 / 1.18 |
| gate nonlinearities | 1.10 / 1.05 | 1.12 / 1.12 |
| output stores | 1.04 / 1.01 | 1.04 / 1.05 |
| everything except the WGMMA | **3.96 / 3.81** | **15.98 / 15.72** |

B = 256 shows the same pattern. Reading the table:

* The time is spread over the serial per-step chain. Apart from the exchange at
  D = 768 (38–43%), no single component holds most of it.
* What separates the full kernel from its GEMM floor (4× at D = 256, 16× at
  D = 768) is the chain of exchange/synchronization, stage, loads and epilogue,
  each waiting on the previous one. Removing any one component alone yields
  1.05–1.74×.
* To reach that headroom, the components have to overlap, not each be
  optimized. Two ways to do that:
  * Run two independent batch tiles per CTA in ping-pong: one warpgroup's
    exchange wait hides behind the other's GEMM and epilogue.
  * Warp-specialize: producer warps do the exchange and loads, consumer warps
    do the GEMM and epilogue.
* At D = 768 the exchange alone is worth up to 1.6–1.7× on the recurrence
  (≈ 90% of the forward). An exchange that avoids the global round trip and the
  grid barrier would be the first target there.

### B. NUMA plane, whole node

Four serving processes, one per GPU, each serving C = 1024 per burst; a go file
starts them together after warm-up. Each GPU sits on its own NUMA node (0–3; 0/1
and 2/3 share a socket, cross-socket distance 21). Conditions:

* **local:** `numactl --physcpubind` with 4 cores of the GPU's node, plus
  `--membind` to that node;
* **remote:** the same on the farthest node, i.e. the other socket;
* **unbound:** no binding.

There were 5 rounds with the condition order rotated: 30 sets, all complete.
`/proc/<pid>/numa_maps` confirms the binding. 85% of resident pages sat on the
bound node for local, and 78–79% on the other socket for remote.

| Case (4 GPUs × C = 1024) | local | remote | unbound | local / remote | local / unbound | burst P90 |
|---|---|---|---|---|---|---|
| LSTM D256 (Triton default) | 629.7 k req/s | 627.7 k | 627.9 k | 1.0024 | 1.0022 | 6.29–6.31 ms |
| LSTM D768 (TL grid S16 BM64, 256 threads) | 117.4 k req/s | 117.4 k | 117.4 k | 1.0004 | 1.0002 | 34.47 ms |

* The NUMA plane is null for this workload: host placement changes node
  throughput by ≤ 0.25%.
* The four GPUs also do not disturb each other. Per-GPU throughput matches the
  single-GPU makespans (6.3 ms, 34.4 ms).

## Status after round 4

* The stop criterion is still not met.
* The NUMA plane is now closed (null). Congestion has only the single-stream ncu
  inference. Every remaining lever is the structural overlap above.
* The ablation gives the ceilings that work would chase: up to about 4× on the
  D = 256 recurrence and well beyond 2× at D = 768, and only if the chain
  actually overlaps.
