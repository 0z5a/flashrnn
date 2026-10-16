# FlashRNN2 scale-out: many heads, many GPUs, many nodes

Status: plan and cost model; phase A (one node) being built; nothing measured yet.

## 1. The question

One recurrence step is g_t = Wx_t + h_{t-1} R + b followed by the pointwise cell; step t needs all of h_{t-1}.
FlashRNN2 keeps each CTA's slice of R in shared memory for the whole call and exchanges only h each step. Two
things grow past one GPU:

* **H >> 1 heads.** Heads are independent; the question is only how to place them so that every SM is busy and
  nothing is exchanged.
* **One head too large for one GPU.** R has 4·D² entries. With a slice of S output columns per CTA the slice is
  4·D·S·b bytes (b = 2 for FP16): 128·D bytes at S = 16. H100 shared memory holds about 200 KB of it, so one CTA
  can keep S = 16 columns up to D ≈ 1536, S = 8 columns up to D ≈ 3072. The number of CTAs needed to keep all of
  R resident is k(D) = 4·D²·b / 200 KB: 41 at D = 1024, 164 at D = 2048, 655 at D = 4096, 2621 at D = 8192. An
  H100 has 132 SMs. Beyond D ≈ 1800 a single head needs more than one GPU to keep R in the fast tier, and beyond
  D ≈ 3600 more than one node of four.

The alternatives for a large head are:

* **A. stream.** Do not split. Keep R in HBM (or host memory, or disk) and re-read it every step.
  T_A(D) = 4·D²·b / BW + γ, bandwidth-bound and quadratic in D. BW is HBM (about 3 TB/s on H100), PCIe (about
  25 GB/s) or NVMe (below 10 GB/s).
* **B. split and stream.** Split R's columns over P GPUs, each streaming its slice from its own HBM every step, plus
  an all-gather of h. T_B = 4·D²·b / (P·BW) + X_P(m) + γ.
* **C. split and keep resident.** Split R's columns over enough GPUs (P ≥ k(D)/132) that every CTA's slice stays in
  shared memory, plus an all-gather of h. T_C = X_P(m) + γ(D/P). The exchanged message per batch tile of 16 rows is
  m = 16·D·b bytes: 32 KB at D = 1024, 128 KB at D = 4096.

X_P(m) is the all-gather cost for P participants and message m, and it depends on the topology and the schedule.
The claim to test: for D above a crossover D*(P, topology), T_C < T_A with the gap growing like D², and the
schedule chosen from the topology (powers of two → recursive doubling, coprime strides → conflict-free cyclic
rounds) attains the upper bound below, which is strictly below T_A. This is a statement inside a measured cost
model; the experiments measure its parameters and check the predicted ordering and the predicted times.

## 2. Cost model

Per directed link (i, j): latency α_ij and inverse bandwidth β_ij (seconds per byte). Three link classes on the
target machine: intra-GPU L2 exchange (α_l2, measured 1.2–1.4 µs as the barrier phase of the R-tiled kernels),
intra-node NVLink (α_nv, β_nv; the nodes are a full mesh of four GPUs, six links per pair), inter-node InfiniBand
(α_ib, β_ib; one NIC per GPU). γ(d) is the per-step compute of a CTA that owns d columns (MMA over K = D plus the
pointwise cell), measured in the kernel.

All-gather of P slices of m/P bytes each, one round = every participant sends at most one message per link and
waits for it:

| schedule | rounds | per-step cost X_P(m) | needs |
| --- | --- | --- | --- |
| direct (every peer at once) | 1 | α + β·(P−1)·m/P on the sender's egress; α + β·m/P when the P−1 links are disjoint and the crossbar is non-blocking | full mesh |
| ring | P−1 | (P−1)·(α + β·m/P) | ring |
| recursive doubling | log₂P | log₂P·α + β·m·(P−1)/P | P = 2^q, hypercube links |
| coprime-stride cyclic (stride s, gcd(s, P) = 1) | P−1 | (P−1)·(α + β·m/P), every directed link carries exactly m/P per round | any P; P prime admits every s |
| hierarchical (nodes of n, P = n·N) | 1 + 1 + 1 | 2·(α_nv + β_nv·m/P·…) + α_ib + β_ib·m/N | two link classes |

Two facts about the schedules are combinatorial and need no measurement:

* **Recursive doubling.** For P = 2^q, q rounds are necessary and sufficient when each participant receives from
  one peer per round (information doubles per round at most). Round r pairs j with j ⊕ 2^{r−1}, so every round is
  a perfect matching and no link is used twice in a round.
* **Coprime stride.** For gcd(s, P) = 1 the map j ↦ j + r·s (mod P) is a permutation for every r = 1..P−1, and
  over the P−1 rounds every ordered pair (j, j') with j' ≠ j occurs exactly once (the multiples r·s run through
  Z_P \ {0}). So the P−1 rounds form a complete, conflict-free exchange with equal load on every directed link in
  every round. For P prime every s works, which leaves the stride free to match the physical link order.

What is not combinatorial, and is measured: α, β per link class, γ, and whether the links are used in parallel
(the direct schedule's cost depends on that). On a full mesh of four with a non-blocking crossbar the model says
direct < recursive doubling < ring for P ≤ 4; the structured schedules matter where the links are not a full
mesh: across nodes, where each GPU has one NIC and the choice is between flat (P−1 remote puts per NIC) and
hierarchical (one remote put per NIC), and for P that is not a power of two.

**Upper bound and crossover.** With the measured parameters, T_C(D, P) ≤ γ(D/P) + rounds·α_max + β_max·m·(P−1)/P
+ α_l2 (the intra-GPU group barrier is still there). T_A(D) ≥ 4·D²·b / BW_hbm. The crossover D*(P) solves
T_C = T_A; above it T_A − T_C grows like D². The pre-registered checks: the measured per-step time of every
configuration is within the model's prediction ± 20% (α, β from the microbenchmarks of the same job), and the
predicted ordering of schedules holds in every paired comparison that the model separates by more than 10%.

## 3. Many heads

Heads do not exchange anything. Per step, head h on GPU j costs γ plus the intra-GPU barrier; H heads fill
132 / k(D) concurrent groups per GPU; more heads than that are serialised by the group loop. Across P GPUs the
placement rule under test is: put whole heads on GPUs while a head fits one GPU (no exchange), and split a head
across GPUs only when k(D) > 132. The measurements: one GPU, throughput versus H for D ∈ {192, 384, 768} at B = 16
and 64 (training step and forward), and four GPUs with heads partitioned (expected: linear in P, verified against
one GPU, with HBM contention as the only coupling).

## 4. Experiments

**Phase A — one node, four GPUs, one process, everything in Triton.** Peer buffers are mapped through CUDA
unified addressing; the kernels take peer buffer addresses as integers and cast them to pointers; flags use
system-scope release/acquire atomics (compile-checked on the login node). Tasks:

1. Microbenchmarks: α_nv and β_nv per GPU pair (store + flag round trip, 16 B–1 MB), HBM bandwidth, PCIe
   host-to-GPU bandwidth, α_l2 (existing). These calibrate the model for the same job.
2. Multi-GPU forward (design C) at D ∈ {1024, 2048} with P ∈ {1, 2, 3, 4}, schedules direct / ring /
   recursive doubling (P = 2, 4) / coprime stride (P = 3: s = 1, 2; P = 4: s = 3), S ∈ {8, 16} as shared memory
   allows. Gate: bitwise equal to the single-GPU R-tiled forward (the exchange carries the same rounded h).
3. Baselines: A (one GPU, R streamed from HBM) at D ∈ {1024, 2048, 4096}; A' (R in pinned host memory) at
   D = 1024; B (four GPUs, slices streamed + direct all-gather) at D = 4096.
4. Many heads (section 3).
5. In-kernel phase timing (exchange wait, compute, barrier) for every configuration; T = 64 and 256 to separate
   per-step cost from setup.

Output: the measured α, β, γ; T_C and T_A per D; the crossover; the schedule ordering against the model.

**Phase B — two nodes, eight GPUs.** Decided after phase A. The exchange across nodes needs device-initiated
communication inside the persistent kernel. Candidates, in order of evidence on this machine: NCCL 2.32.3's
device API (LSA for NVLink, GIN for the network; used by this account's earlier collective work here), NVSHMEM 3.9
(built here with IBRC and IBGDA transports, not yet run on these GPUs), and a host-driven NCCL all-gather per
step as the "trivial collective" baseline (expected 10–20 µs per step). Phase B compares flat direct, hierarchical,
recursive doubling (P = 8) and coprime-stride (P = 7) schedules at D ∈ {2048, 4096} and places the first
cross-node crossover against the stream baseline.

**What is not claimed.** The inequality is proven inside the α–β model for the measured link classes and
checked on this machine; it is not a statement about every interconnect. Phase A cannot show inter-node effects.
Disk streaming is bounded analytically (NVMe bandwidth), not run.

## 5. Status

| item | state |
| --- | --- |
| cost model, schedule facts, criteria | this document |
| peer pointer / sys-scope atomics in Triton | compile-checked on the login node |
| node topology | four GPUs, full mesh NV6, one NIC per GPU (recorded by an earlier preflight) |
| multi-GPU forward kernel, stream baselines, microbenchmarks | being built |
| phase A batch | not submitted |
| phase B | not decided |

## 6. Known model gaps (written 2026-10-16, after phase A was sealed and submitted, before any result)

These are visible in the sealed code of job "scale phase A"; check 2 is reported exactly as pre-registered, and any
corrected model that addresses them is labelled post hoc.

* **Streamed R is often L2-resident.** H100 L2 is about 50 MB. R at D = 2048 (FP16) is 4·2048²·2 ≈ 33.5 MB, and each
  GPU's slice in design B at D = 4096, P = 4 is the same size, so those rows stream from L2, not HBM. `predict()`
  prices every stream at the measured HBM copy bandwidth, so design A/B rows below about 50 MB of R per GPU are
  expected to run well under the prediction. Only P = 1 at D = 4096 (134 MB) is HBM-bound.
* **The barrier may be counted twice.** `predict()` adds α_l2 (1.3 µs) on top of the exchange term; for P > 1 the
  cross-GPU wait already is the step barrier.
* **α_l2 is not measured in this job.** `scale_micro` does not measure the intra-GPU barrier; the 1.3 µs is the value
  from the gate-batched backward profile, not from the same job as the plan says.
* **Necessity at two nodes.** Column-only slices need S ≥ 8 (smallest MMA shape), so one CTA's slice is ≥ 64·D bytes
  and fits shared memory only up to D ≈ 3200, where D/8 = 400 CTAs fit one node (528 SMs). With the phase A kernel,
  every D whose slices fit is also a one-node size; a two-node run of it is a topology test, not a necessity test.
  The two-node necessity case (D = 4096: about 1024 CTAs of 131 KB) needs a K split per CTA, a per-step partial-sum
  reduction across the K-part CTAs and a second barrier; that term goes into the phase B model.

## 7. Phase B pre-registration (written 2026-10-16 21:30 CEST, before any phase B run and before phase A results)

**What runs.** `tools/scale_exchange/exchange_bench.cu`: one MPI rank per GPU, one persistent CTA of 512 threads per
rank. Each step: spin γ ns (stands in for the recurrence compute), write the own slice of h (m/P bytes, a
pattern f(rank, step, i)), run the schedule's rounds, wait for each round's arrivals, next step. Same-node sends
run in parallel on disjoint warp groups (15 warps split evenly over the round's same-node sends); network sends
are posted one after another by one thread (GIN) or one warp per send (NVSHMEM). Arrival counters are kept per
(step parity, round). Three transports: NCCL 2.32.3 device API (LSA stores within a node, GIN put + signal across
nodes), NVSHMEM 3.9 (`putmem_signal_nbi_warp`; IBRC, plus IBGDA if the node has its prerequisites), and the
host-driven baseline (spin kernel + `ncclAllGather` per step, plain and as one CUDA graph). This is an exchange
skeleton: it measures X_P(m) of section 2 for one aggregated per-GPU message; it is not the recurrence. The real
kernel would first gather its CTAs' slices behind the intra-GPU barrier (α_l2) and then send them.

**Schedules.** The host tables in `schedule.h`, checked exhaustively on the CPU by `schedule_check` for P ≤ 64,
L ∈ {1, 2, 4, 8} (12606 cases, 0 failures): completeness, round counts, permutation rounds (cyclic, ring), exact
cover (cyclic), perfect matchings (doubling), the coprime-stride rule, and the per-round NIC/NVLink loads. On two
nodes of four (P = 8) the network carries 32 blocks per step under flat and doubling, 14 under ring, and 8 under
hier, which is the lower bound P·(N−1) (every block must reach the other node once).

**Model (per backend, parameters fitted only from micro cases of the same job).**
* Micro cases: `repeat:R` (R ∈ {1, 2, 4, 8} rounds to one peer) at P = 2 within a node and across nodes, over
  slices 16 B–1 MB; `flat` at P = 3 and P = 4 within a node (2 and 3 parallel same-node sends) over the same
  sweep; `fanout:k` (k ∈ {1..4} puts to the other node in one round) at P = 8 for slices 16 B and 8 KB.
* Fit: per_step(R, s) = o + R·(α_c + β_c·s) by least squares for c ∈ {nv, net}; β_nv(n) for n parallel
  same-node sends from the P = n + 1 within-node flat sweep (n = 1 from repeat); the per-put gap
  g = slope_k(per_step) − β_net·s, averaged over the two fanout slices.
* Prediction for a schedule (rounds k, ranks i, from the same tables the kernel runs):
  round_k = max_i max(α_nv + β_nv(n_k,i)·b_local,k,i , α_net + (r_k,i − 1)·g + β_net·B_net,k,i), where b_local is
  the largest same-node send, n the number of same-node sends, r the number of network sends and B_net their total
  bytes (terms absent when there is no such send); X = Σ_k round_k; per_step = o + γ + X.
* Hier written out (P = n·N, slice b = m/P): round 0 = α_net + (N − 2)·g + β_net·(N − 1)·b; round 1 =
  α_nv + β_nv(n − 1)·N·b. Flat on two nodes of four: one round = max(α_nv + β_nv(3)·b, α_net + 3·g + β_net·4·b).

**Checks.**
* B1 correctness (qualification passes, `--check 1`): every case has 0 wrong words at the last step, 0 per-step
  mismatches (first and last word of every block, every step) and no timeout. A backend that fails its two-rank
  cross-node smoke or B1 is excluded from B2–B5 and reported as a negative result.
* B2 model: every comparison case (γ = 0; P = 8: flat s = 1, 3, cyclic s = 1, 3, ring, doubling, hier;
  P = 7: flat s = 1, 3, cyclic s = 1, 2, 3, ring; slices 1, 4, 8, 16, 64 KB and 9376 / 18736 B for P = 7) is
  within ±20% of the prediction, and for every pair of schedules at the same (backend, P, slice) that the model
  separates by more than 10%, the predicted faster one is measured faster.
* B3 topology claim: at P = 8 on two nodes, for the claim slices (8 KB and 16 KB: m/P for D = 2048 and 4096 with a
  16-row tile in FP16), the best of hier / doubling beats the faster flat stride by a measured ratio ≥ 1.1, with
  the model predicting ≥ 1.0. At P = 7 the same test for the best coprime cyclic stride against flat.
* B4 against the trivial collective: the best device-initiated schedule beats the host-driven `ncclAllGather`
  (graph version) at the claim slices by a measured ratio ≥ 1.1.
* B5 additivity: for γ ∈ {2, 5} µs, per_step(γ) is within max(10%, 1 µs) of γ + per_step(0) (hier and flat, 8 and
  16 KB) — the condition for composing X with the measured compute.
* Composition (reported, labelled composed, not a check): T_C(D, P = 8) = γ(D/P) + α_l2 + X_8(m/P) against phase
  A's measured T_A(D). γ(D/P) at K = D = 4096 is not measured (that needs the K-split kernel of section 6); it is
  scaled from phase A's measured per-step compute and the scaling is stated with the number.

**Amendment (2026-10-16 21:50 CEST, after phase B was submitted as job "scale phase B", before any phase B
result).** The general round formula above charges same-node traffic by the largest single send; the schedule
table sends the N blocks of hier round 1 as N separate (non-contiguous) sends per peer, so the general formula and
the explicit hier row disagreed. The analysis charges same-node traffic per peer link: n = number of distinct
same-node peers in the round, b_local = bytes to the busiest same-node peer, cost α_nv + β_nv(n)·b_local (n capped
at 3, the largest measured fan-out). This makes the general formula equal to the explicit hier row
(α_nv + β_nv(n − 1)·N·b). `tools/scale_exchange/analyze_b.py` implements the fits and checks B1–B5 as written; its
port of `schedule.h` is cross-checked against the `schedule_check` loads the batch preflight writes (1148
schedules, 0 mismatches on the login build), and it was exercised on a synthetic batch generated from known
parameters (parameters recovered within the injected 3% noise; all checks evaluated).

## 8. Results (2026-10-16 22:30 CEST)

### 8.1 Pre-registered verdicts

| check | verdict |
| --- | --- |
| A check 1 qualification | FAIL: 2/40 not compared (P = 1 reference cannot exist where splitting is necessary: D = 4096, D = 2048 S = 8); bitwise vs the single-GPU R-tiled kernel false everywhere (bitwise vs P = 1 true everywhere, FP64 ≈ 5e-5); sLSTM stream at P = 1 FP64 error 0.22 (real bug); one exchange timeout on the first config (devices 0/1) |
| A check 2 model | FAIL 2/40 within ±20%, orderings 12/20 — the micro-benchmark measured pair 0–1 at 25 µs–9 ms for every size (other pairs ≈ 2.4 µs, 21 GB/s), which inflated every multi-GPU prediction to 200–800 µs; cause not yet diagnosed |
| A check 3 crossover | 0 accepted (predicted ratios invalid by the same micro fault) |
| A heads | no data: harness read CUDA events before synchronizing; an sLSTM case also hit an illegal memory access |
| B1 | FAIL for every device backend: every P = 7 launch failed at init (NCCL builds no LSA team for 4 + 3 ranks; NVSHMEM requires equal PEs per node). B2–B5 not evaluated |

### 8.2 Post hoc (not pre-registered; explains the failures, does not overturn them)

* **Phase B restricted to P = 8** (`analyze_b.py --post-hoc-p8-only`, evidence in `scale-out-evidence/`):
  qualification 120/120 cases, 0 wrong words, 0 per-step mismatches, 0 timeouts on all three device backends.
  * GIN (GDAKI, GPU-initiated, no CPU proxy): α_nv 5.2 µs, α_net 10.3 µs, per-put gap 3.8 µs, o 6.2 µs. Model 30/35
    within ±20%, orderings 75/75; the five misses are all ring (measured 0.70–0.75 of prediction — hypothesis: ring runs
    as an asynchronous pipeline, the model sums synchronous rounds). Hier beats the best flat stride by 1.18× (8 KB) and
    1.13× (16 KB), predicted 1.23× / 1.20×. Additivity 8/8. But the best device schedule (25.1 µs at 8 KB) is slower
    than the host-driven `ncclAllGather` captured in a CUDA graph (21.7 µs): B4 fails.
  * NVSHMEM IBRC: o_net 36 µs (proxy-driven), hier slower than flat. NVSHMEM IBGDA (initialised, used for device-side
    APIs): 0/35 within the model, 1.5–3.5× slower than predicted. One warp per send gives β_nv ≈ 3 GB/s, so NVSHMEM's
    hier loss is an artifact of this skeleton (6 same-node sends in hier round 1), not a topology result.
  * Coprime cyclic at P = 8, s = 3 is 2–3× slower than flat on every backend: with one NIC per GPU the per-round α
    dominates; the conflict-free property does not buy anything at these message sizes.
* **Phase A measured per-step times (LSTM; sLSTM stream excluded until its bug is fixed):** single GPU streaming R
  17.1 µs (D = 1024) and 35.2 µs (D = 2048, R = 33.5 MB is L2-resident, so this is not an HBM-bound baseline);
  R split over 2 GPUs and kept in shared memory with ring exchange 12.0 µs and 17.5 µs (S = 8) — 1.42× and 2.0×.
  Within one node at P = 4 recursive doubling (15.5 µs) beats direct (17.4) and ring (18.3).
* **Not verified:** the inequality against an HBM-bound single-GPU stream. `1:4096 stream` cannot run (stream mode
  still needs one CTA per column slice). Composed estimate only: 4·D²·b / 1.47 TB/s (measured copy bandwidth) ≈ 91 µs
  plus γ at D = 4096, against X_8 ≈ 22–25 µs plus γ(D/P), with γ at K = 4096 not measured.

### 8.3 Repairs needed before a rerun
1. A single-GPU stream kernel that loops over column groups at D = 4096 (the HBM-bound baseline the claim needs).
2. Diagnose micro pair 0–1 (slow at every size, and the first-config timeout on the same pair).
3. Fix the sLSTM stream correctness bug (FP64 error 0.22 at P = 1).
4. Heads: synchronize before reading events; find the first faulting sLSTM heads case with CUDA_LAUNCH_BLOCKING.
5. Qualification reference where P = 1 cannot exist: FP64 plus cross-P/cross-schedule bitwise agreement.
6. Prime P: not placeable unevenly on nodes of four with either library; a single-node P = 3 case shows the
   coprime property on hardware cheaply.
