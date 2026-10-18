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

## 9. Symmetric exchange buffers in the multi-GPU recurrence (pre-registration, 2026-10-17, before any run of it)

**Change.** Until now only the phase B exchange skeleton ran on symmetric memory; the real multi-GPU recurrence
(`multi_gpu.py`) allocated its exchange buffer and arrival counters with the PyTorch allocator and reached the peers
through unified addresses after enabling peer access ("uva"). `multi_gpu.recurrence(alloc="nccl_symm")` now takes both
from NCCL 2.32 symmetric windows (`symm_mem.cu` / `symm_mem.py`): one communicator per GPU of the process
(`ncclCommInitAll`), `ncclMemAlloc` (cuMem-backed) on every GPU, `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`
in one group, and every peer's address read on the device from the window's LSA mapping (`ncclGetLsaPointer`). The
kernel is unchanged; only the address tables differ. PyTorch in this environment carries NCCL 2.20.5 under the same
soname, so 2.32.3 (the copy already staged for phase B) is opened by path with `RTLD_LOCAL | RTLD_DEEPBIND` and called
through dlsym'd pointers; the login preflight checks that the opened library reports 2.32.3 in a process that has
imported PyTorch. NVSHMEM is not used here (one PE per process; this path is one process). Cross-node symmetric buffers
need a CUDA recurrence kernel with network puts (Triton cannot issue them) and are not part of this batch.

**Repairs from section 8.3 that this batch addresses, and what it leaves out.**
* Repair 2 (micro pair 0-1 slow at every size, first-configuration timeout on the same pair). Hypothesis, stated before
  the run: the first pair compiled the ping-pong kernel for every message size inside its timed window (one Triton
  specialisation per size; the later pairs reused them), and the first multi-GPU call compiled while its first GPU was
  already spinning on the exchange. The micro runs twice: unchanged (`--warmup 0`, uva only), and with one untimed
  launch per size and pair before the timed one (`--warmup 1`, uva and nccl_symm). The hypothesis is supported if pair
  0-1 is ≥ 3× slower than the median pair at 16 B without warm-up and within 1.5× of it with warm-up. Qualification and
  timing run every configuration once untimed first.
* Repair 5 (no P = 1 reference where splitting is necessary). Qualification rule for this batch: finite, FP64 within
  2e-2 relative (as before), bitwise equal to the P = 1 run where that run exists, the output digest equal across every
  P and schedule of the same (cell, D, S) (cross-P agreement, the reference where P = 1 cannot run), and the symmetric
  run bitwise equal to the uva run. The single-GPU R-tiled comparison is reported, not gating (it was false everywhere
  in phase A: a different MMA accumulation order, not a defect of the exchange).
* Repair 6 (prime P): D = 1536, S = 8 at P = 3 (direct with strides 1 and 2, ring) on one node.
* Left out: the D = 4096 single-GPU HBM-bound stream baseline (repair 1), the sLSTM stream bug (repair 3; this batch
  runs resident slices only), many heads (repair 4), cross-node runs, and the cost-model checks of phase A.

**Runs (batch "symm", one node, four H100):**
* Preflight: topology, peer-access probe (gates the batch), and, reported: NCCL versions in the process, multicast
  support per GPU (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`; the full-mesh NVLink nodes are expected to have none, so
  nothing here depends on NVLS), the symmetric session and its peer-table probe. A symmetric-session failure is a
  reported outcome: the uva rows of every later task still run.
* Micro (two runs, above): one-way latency and bandwidth per GPU pair, 16 B to 1 MiB.
* **S1 qualification**: design C (resident slices) LSTM/sLSTM, T = 48, B = 16: D = 1024, S = 16 at P ∈ {2, 4}
  (direct, ring, recursive doubling; P = 4 also stride 3); D = 2048, S = 8 at P ∈ {2, 4} (same set); D = 1536, S = 8 at
  P = 3 (direct strides 1 and 2, ring); 17 configurations × 2 cells, rule above.
* **S2 paired timing**: the same 17 configurations × 2 cells, B = 16, T = 256: per process start 20 blocks of 3 calls
  per allocation, order alternating per block, kernel time of the slowest GPU per call, three process starts; ratio
  uva / nccl_symm with the 95% hierarchical bootstrap of the earlier batches (10,000 draws, starts then blocks). Read
  per configuration: symmetric faster if the lower bound is ≥ 1.02, uva faster if the upper bound is ≤ 0.98, otherwise
  no difference resolved. Expectation stated now: no difference resolved for most configurations — both allocations
  map the same NVLink peer memory, and the exchange per step is a few KB; a difference would point to the mapping
  (cuMem granularity, TLB) or to the flag placement, and would be reported as such. Starts whose outputs were not
  bitwise stable are excluded and listed.

## 10. Results of the symmetric-buffer batch (job "symm", 2026-10-17) and the repairs that followed

Evidence: `scale-out-evidence/symm-analysis.json` (`tools/scale_symm_analyze.py`), archive verified by SHA-256.

**Preflight.** PyTorch's NCCL 2.20.5 and the opened NCCL 2.32.3 coexisted in one process; the symmetric session over
four GPUs came up and its peer-table probe passed (every rank's store through its LSA table landed in the right
buffer). No GPU reports multicast (NVLS) support.

**Micro (repair 2).** The hypothesis holds by the pre-registered numbers: without warm-up pair 0-1 took 130 µs at
16 B, 53x the median pair (2.45 µs), and moved 0.1 GB/s at 1 MiB; with one untimed launch per size it took 2.53 µs
(1.05x the median) and 23.5 GB/s. The slow pair of phase A was Triton compiling the ping-pong kernel for every message
size inside the timed window of the first pair. Symmetric buffers: 2.16-2.26 µs per pair at 16 B (uva 2.13-2.53 µs),
23.1-24.2 GB/s at 1 MiB (uva 23.0-23.9).

**S1: 0/34 by the rule.** 21 rows are clean: FP64 4-6e-5, one output digest per (cell, D, S) across every P and
schedule, and the symmetric run bitwise equal to the uva run in all 21. 5 rows timed out on rank 0 (all at D = 1024; four
of them direct). 8 rows returned corrupted outputs without a timeout (FP64 error 0.005-0.25, uva and symmetric runs
different from each other): ring or recursive doubling in every case (P = 2 doubling at D = 2048, P = 3 ring at
D = 1536, P = 4 ring/doubling and P = 2 doubling at D = 1024). Cross-P agreement fails for every (cell, D, S) because
of these rows.

**S2.** 28 of 34 rows had bitwise-stable outputs in every start: uva / symmetric 0.94-0.97 (uva faster) in 10 rows,
all at D = 1024; symmetric faster in 1 row (P = 2, D = 2048, direct, LSTM, 1.035); not resolved in 17. The 6 rows
without data are the ring/doubling configurations whose outputs changed between calls.

**Diagnosis (post hoc).** Both failures are in the phase A forward kernel, not in the allocation:
* *Ring and recursive doubling had no barrier among the CTAs of one GPU.* Their waits count only the round
  partner's CTAs. A CTA that finished a step early went on to the next step's publish and overwrote, in the double
  buffer, h that a slower CTA of the same GPU was still reading for the previous step. The direct schedule's wait
  counts every CTA on every GPU, a full barrier, so it was never affected. This also means phase A's ring/doubling
  timings omitted the intra-GPU barrier that the cost model charges (α_l2).
* *Rank-0 timeouts.* The first GPU spins on the exchange from its launch on, while the host still built the later GPUs'
  address tables (synchronous host-to-device copies) between launches. The spin limit (2^20 polls) is shorter than
  that skew can be.

**Repairs (commit after this section).** Ring and doubling now signal and wait on a per-GPU counter before overwriting
the buffer half (all CTAs of the GPU must have published the previous step); every address table is built before the
first launch; the cross-GPU spin limit is 2^26 polls. Phase A's and this batch's ring/doubling results are superseded
until they are re-measured.

## 11. NVSHMEM, the multi-GPU backward, and the repaired forward (pre-registration, 2026-10-17, before any run of it)

**Multi-GPU backward (dG exchange).** `hybrid_bw.cu` with `FRNN_MG = 1`: the D / 16 column slices are split over P GPUs;
every CTA publishes its rounded dG columns into the exchange buffer of every GPU (direct schedule), fences at system
scope and joins its grid barrier; one CTA per GPU then adds one to every GPU's arrival counter and every CTA waits for P
arrivals per step. This is a full barrier over all CTAs of all GPUs every step, so the double buffer is safe across GPUs
(the argument is in the source). The address tables come from the PyTorch allocator ("uva"), NCCL symmetric windows,
or NVSHMEM. The per-CTA arithmetic is the single-GPU kernel's, so dG and the initial-state gradient must be bitwise
equal to the single-GPU kernel with the same batch tile and register split. `hybrid_bw.backward_mg` (one process) and
`hybrid_bw.mg_rank_kernel` (one GPU's share). ptxas: no spill in any MG build (D = 2048: 168 registers at BM = 16,
255 at BM = 64).

**NVSHMEM.** One process per GPU (`tools/scale_nvshmem.py`): unique-ID bootstrap, `nvshmem_malloc` buffers and
counters, peer addresses from `nvshmem_ptr` (NVLink peers), NVSHMEM barrier before every call; the same forward and
backward kernels. NVSHMEM 3.9 is the copy staged for phase B; its host library has no CUDA runtime dependency and
PyTorch here ships none.

**Runs (batch "mg", one node, four H100):**
* **F1, repaired forward qualification**: the 17 configurations of section 9 × LSTM/sLSTM, uva and symmetric, rule of
  section 9, run three times in three processes; additionally all digests of a (cell, D, S) must agree across the
  three runs. Prediction from the diagnosis: no corrupted row and no timeout.
* **F2, uva / symmetric paired timing** on the repaired kernel, protocol and reading of section 9 (S2).
* **G1, multi-GPU backward qualification**: D ∈ {1024, 1536, 2048} × P ∈ {2, 4} (and P = 3 at D = 1536) × (B = 16,
  BM = 16; B = 64, BM = 16; B = 64, BM = 64) × LSTM/sLSTM, FP16, T = 32: dG and d_initial bitwise equal to the
  single-GPU kernel, with uva and with symmetric buffers.
* **G2, race verification of the multi-GPU backward** (rule of max-width.md section 12): P = 4 at D ∈ {1024, 2048}
  (LSTM, BM = 16, B = 40) and D = 2048 (sLSTM, BM = 64, B = 100); P = 3 at D = 1536 (sLSTM, BM = 16, B = 40); T = 48,
  21 runs per variant. The content check reads every exchanged operand against the owning GPU's stored dG. Broken
  variants: single buffer, no grid barrier, no cross-GPU wait (each keeps its last barrier).
* **G3, multi-GPU backward against one GPU**, kernel time, B = 64, T = 256, LSTM/sLSTM: P = 4 at D ∈ {1536, 2048} and
  P = 2 at D = 2048, batch tile 16 (the split leaves room for 2-4 batch tiles at once per GPU), against the single-GPU
  kernel with the library's batch tile (64), and uva against symmetric buffers; plus B = 16 at D = 2048, P = 4 (LSTM;
  one tile, where splitting cannot add concurrency). Paired, three process starts × 20 blocks × 3 calls, order rotated
  per block. Reading: one / multi-GPU ratio with the 95% hierarchical bootstrap; multi-GPU faster if the lower bound is
  ≥ 1.02, slower if the upper bound is ≤ 0.98, else not resolved. Expectation: faster at B = 64 (more tiles at once),
  slower at B = 16 (same per-step work plus a cross-GPU barrier).
* **N1, NVSHMEM correctness**: P = 4 (forward 4:1024:16 and 4:2048:8 with direct, ring, doubling; backward D ∈ {1024,
  2048} at B = 16/BM = 16 and B = 64/BM = 16, LSTM, and D = 2048 sLSTM B = 64), P = 2 (forward 2:1024:16 direct and
  ring, 2:2048:8 direct; backward D = 2048, B = 64), P = 3 (forward 3:1536:8 direct and ring; backward D = 1536, B = 64):
  outputs bitwise equal to the one-process reference (forward: the same configuration on uva buffers; backward: the
  single-GPU kernel), no wait timeout. Kernel times per call (slowest PE, 9 calls) are reported next to the reference's
  median, not paired (different processes).

## 12. Results of the "mg" batch (2026-10-18)

Evidence: `scale-out-evidence/mg-analysis.json` (`tools/scale_mg_analyze.py`), archive verified by SHA-256.

**G1, multi-GPU backward: 42/42.** dG and d_initial are bitwise equal to the single-GPU kernel with the same batch
tile and register split. This holds for D ∈ {1024, 1536, 2048}, P ∈ {2, 4} (and P = 3 at D = 1536), B = 16 / 64,
BM = 16 / 64, LSTM and sLSTM, on uva and on NCCL symmetric buffers alike.

**G2, race verification: 4/4.**
* The correct kernel counts zero violations in every run.
* No grid barrier: flagged in 17-21 of 21 runs.
* No cross-GPU wait: flagged in 21/21 runs.
* Single buffer: flagged in 21/21 runs.
* No variant changed the outputs silently.

**G3, multi-GPU against one GPU (backward kernel time, paired, 3 starts):**

| case | one GPU (1-D, BM = 64) / multi-GPU (BM = 16) | uva / symmetric |
| --- | --- | --- |
| LSTM D = 1536, P = 4, B = 64 | 2.30 [2.29, 2.31] (15.0 → 6.5 ms) | 0.98, not resolved |
| sLSTM D = 1536, P = 4, B = 64 | 2.34 [2.33, 2.35] | 0.99, not resolved |
| LSTM D = 2048, P = 4, B = 64 | 2.42 [2.41, 2.42] (17.8 → 7.4 ms) | 0.99, not resolved |
| sLSTM D = 2048, P = 4, B = 64 | 2.40 [2.40, 2.41] | 0.99, not resolved |
| LSTM D = 2048, P = 2, B = 64 | 1.27 [1.26, 1.27] | 1.00, not resolved |
| sLSTM D = 2048, P = 2, B = 64 | 1.24 [1.24, 1.25] | 1.00, not resolved |
| LSTM D = 2048, P = 4, B = 16 | 0.99 [0.94, 1.08], not resolved | 0.99, not resolved |

The prediction holds: faster at B = 64, where the split puts four batch tiles on the GPUs at once, and no gain at
B = 16. Context (max-width.md section 21, single GPU, same shapes): the 2-D split backward takes 6.0 ms at D = 1536 and
7.9 ms at D = 2048. One GPU with the 2-D kernel therefore matches four GPUs with the multi-GPU 1-D kernel. A
multi-GPU version of the 2-D kernel is the obvious combination; it is not built. Symmetric buffers make no difference.

**F1, repaired forward: 0/34; F2: 26 rows with data, 8 without.** The repair of section 10 did not fix the forward:
* Ring and doubling outputs are still wrong at P = 2 and P = 3 and at D = 1024, P = 4.
* Some wrong outputs repeat to the last digit across runs and batches. For example, 4:1024 ring LSTM has an FP64 error
  of 0.10484 in all three runs here and in section 9.
* The direct schedule at D = 1024 "timed out" on rank 0 again. Those calls returned within seconds, so the spin limit
  (2^26 polls) was never reached. The cause is something else.
* Every configuration at P = 4, D = 2048 passes, as does every P = 3 direct configuration.

Two findings (post hoc):
* *Read-back of the own block (a confirmed defect).* In round 0, ring and doubling forward the GPU's own block. Each CTA
  stores it into the local buffer, crosses a block barrier, and reads it back with L2-only loads (`.cg`). A block
  barrier orders memory only at CTA scope, which the L1 can satisfy. So the read-back can see the L2 copy before the
  store reaches it.
* *The "timeouts" without a spin-out (not explained).* They happened only in the qualification process. There, D = 1024
  is the only width where the single-GPU R-tiled reference runs just before the multi-GPU call. They never happened in
  the paired runs (T = 256, no such reference).
The multi-GPU forward is not qualified. Its timings (F2; phase A's ring/doubling) stay superseded.

**N1, NVSHMEM: 0/18, a build defect.** `nvshmemx_init_attr` is inline in the NVSHMEM headers and calls
`nvshmemi_init_thread` from `libnvshmem_device.a`, which the helper did not link. Every worker failed at `dlopen`. The
login gate compiled the helper but never loaded it. The repaired build is relocatable device code, linked against
`libnvshmem_device.a` with CUDA 13.2 (the toolkit that built that library; 12.8 rejects its ABI), and it now passes a
login-node `dlopen` check.

## 13. Diagnosis batch "mg2" (pre-registration, 2026-10-18, before any run of it)

**Kernel change.** Round 0 of ring and doubling stores the GPU's own block to the round's partner from registers. In
doubling's later rounds the own block is also sent from registers. Every block that arrived from a peer is still read
from the local buffer, behind the previous round's system-scope acquire.

**F3, forward correctness with this change** (`tools/scale_fwd_diag.py`, uva buffers, B = 16, T = 48, LSTM/sLSTM, the
17 configurations of section 9). Each configuration runs in three sequences of five calls:
* five plain multi-GPU calls in a row;
* five multi-GPU calls, each right after the single-GPU R-tiled reference (only D = 1024 can run it);
* five multi-GPU calls, each right after the P = 1 run.

The process has no NCCL communicator. Once more, at D = 1024, it has one. Rule (per configuration and cell): every
call completes without an exchange error, the FP64 error is below 2e-2, and every call of every configuration of the
same (cell, D, S) has one output digest. Readings fixed now:
* If ring/doubling pass and direct still fails only right after the single-GPU reference, the timeout is an
  interaction with that kernel, not with the exchange.
* If failures appear only with the NCCL communicator, NCCL's presence is the cause.
* If ring/doubling still fail without either, the round-0 change did not address the defect.

**N1 again** with the repaired NVSHMEM helper (the cases of section 11).

## 14. Results of the diagnosis batch "mg2" (2026-10-18) and the repairs

Evidence: `scale-out-evidence/mg2/` (diag and NVSHMEM records), archive verified by SHA-256.

**F3 separates two defects.** Pre-registered reading 1 holds for D = 1024, reading 3 for S = 8.
* *D = 1024: a harness defect, not an exchange defect.*
  * Every configuration and cell gives one output digest (FP64 5-6e-5) in all five plain calls and all five calls after
    the P = 1 run, with or without an NCCL communicator.
  * Every failure, timeouts included (direct too), happens on a call right after the single-GPU reference.
  * The cause: Triton and the CUDA runtime launch on the current device. `multi_gpu.recurrence` left the current device
    at the call's last GPU, and the reference did not set it. So the single-GPU kernel ran on another GPU, reading the
    first GPU's memory over NVLink, and was not covered by the synchronize of the first GPU. It was still running when
    the next multi-GPU call started.
  * This also explains the fast "timeouts" of sections 9 and 12, and probably phase A's "bitwise against the
    single-GPU kernel: false everywhere".
* *S = 8, P = 2 / 3, ring and doubling: a real exchange race.* Wrong outputs appear in any sequence, including plain
  calls (2:2048 ring/doubling, 3:1536 ring). After the rounds, ring and doubling did not make the CTAs of one GPU wait
  for each other. A CTA of the next step could therefore read the GPU's own block before another CTA of the same GPU
  had stored its slice. The barrier of section 10 only protected the overwrite of the previous buffer half.
* The NCCL communicator changes nothing: diag-nccl-k512 equals diag-nonccl-k512.

**N1 with the repaired helper: 15/17.** NVSHMEM (one process per GPU, `nvshmem_malloc`, `nvshmem_ptr`) gives outputs
bitwise equal to the one-process reference in:
* every direct forward;
* ring/doubling at P = 4;
* 2:1024 ring;
* the multi-GPU backward at D = 1536 / 2048 and at D = 1024, B = 16.

The two failures:
* 3:1536 ring: the race above.
* Backward at D = 1024, B = 64 at P = 4: its single-GPU reference ran into the same current-device defect (28 ms
  against 0.5 ms), so that comparison is void.

Kernel times (slowest PE, unpaired against the one-process reference, report only):
* forward 0.64-1.15 ms, against 0.71-1.32 ms on uva in one process;
* multi-GPU backward at T = 32: 0.8-1.0 ms.

**Repairs.**
* `multi_gpu.recurrence` restores the caller's current device.
* The harness launches each single-GPU reference inside its device context (`scale_bench.single_gpu`, the NVSHMEM
  reference).
* Ring and doubling signal a per-GPU counter once a CTA's slice of the own block is stored, and wait after the rounds
  until every CTA of the GPU has done so. That one barrier gives visibility of the own block for the next step, and also
  guarantees that the half overwritten next was read completely. It replaces the section 10 barrier.

## 15. Batch "mg3" (pre-registration, 2026-10-18, before any run of it)

With the section 14 repairs:
* **F4**, the section 13 rule, uva buffers, no NCCL communicator, all 17 configurations × 2 cells.
* **F5**, the section 9 qualification rule (uva and symmetric, cross-P digests, one run).
* **N2**, NVSHMEM at P = 2 / 3 / 4, the section 11 cases, rule of N1.

Prediction: F4, F5 and N2 pass in full. A failure of any ring/doubling configuration means the per-GPU barrier is
still not enough. In that case ring and doubling are withdrawn and only the direct schedule is reported.

## 16. Results of batch "mg3" (2026-10-18)

Evidence: `scale-out-evidence/mg3/`, archive verified by SHA-256. The section 15 prediction holds in full.

**F4: 510/510 calls clean.** 17 configurations × 2 cells × 15 calls in the three sequences, uva buffers, no NCCL
communicator. No exchange error, FP64 error ≤ 6e-5, and one output digest per (cell, D, S) across every P, schedule and
call. That includes ring and recursive doubling at P = 2 / 3 / 4 and calls right after the single-GPU reference.

**F5: 34/34** by the section 9 rule:
* uva and symmetric runs bitwise equal;
* one digest per (cell, D, S) across P and schedules;
* bitwise equal to the P = 1 run wherever that run exists;
* FP64 within budget.

With the reference now on the right GPU, the comparison with the single-GPU R-tiled kernel is still false at D = 1024
(14/14), so the section 14 guess that it explained phase A's "false everywhere" is withdrawn. The two kernels
accumulate the K chunks in different orders, as section 9 said.

**N2: 18/18.** NVSHMEM, one process per GPU, P = 2 / 3 / 4: every forward configuration is bitwise equal to the
one-process uva run, every multi-GPU backward bitwise equal to the single-GPU kernel.

Kernel times (report only; NVSHMEM's slowest PE against the one-process run, not paired):

| case | NVSHMEM | reference |
| --- | --- | --- |
| forward 2:1024 direct / ring | 0.67 / 0.67 ms | 0.72 / 0.74 ms (uva, one process) |
| forward 4:1024 direct / ring / doubling | 0.81 / 0.94 / 0.80 ms | 0.95 / 1.08 / 0.94 ms |
| forward 4:2048 direct / ring / doubling | 0.98 / 1.14 / 0.97 ms | 1.15 / 1.32 / 1.13 ms |
| forward 3:1536 direct / ring | 10.3 / 10.3 ms | 10.1 / 10.1 ms |
| backward D = 2048, B = 64, P = 4 (T = 32) | 0.98 ms | 3.03 ms (single GPU, 1-D, BM = 16) |
| backward D = 1024, B = 16, P = 4 | 0.45 ms | 0.35 ms (single GPU) |

The forward times are 10-15% lower under NVSHMEM than in one process. The one process launches the GPUs one after
another, while NVSHMEM's processes start after a common barrier, so part of that gap is launch order, not the
exchange. It is not a claim.

**State of the multi-GPU path after sections 9-16:**
* The forward (direct, ring, recursive doubling; P = 2 / 3 / 4) and the backward (dG exchange) qualify on uva, NCCL
  symmetric and NVSHMEM buffers.
* The backward passed race verification.
* Phase A's ring/doubling timings and its pair-0-1 micro numbers are superseded by these repairs; timing the repaired
  forward against the model (phase A check 2) has not been rerun.

## 17. Batch "mg2d": the repaired forward against the cost model, and the multi-GPU 2-D backward (pre-registration, 2026-10-18, before any run of it)

**Part 1, phase A checks 2 and 3 on the repaired forward.**
* *What runs.* The phase A timing (`scale_bench.py time`: per-step cost from the slope between T = 64 and T = 256,
  median of 9 calls; compute phase from a PROF build; prediction `predict()` unchanged from phase A) on the 17
  resident configurations of section 9 (LSTM, sLSTM).
* *Stream baselines.* The one-GPU stream baselines at D = 1024 and 2048 and the four-GPU stream configuration at
  D = 2048, S = 16; LSTM only, since repair 3 (the sLSTM stream bug) is not done.
* *Calibration.* α and β come from the micro run of the same job, with warm-up (section 10).
* *Rule.* Checks 2 and 3 are evaluated with `tools/scale_analyze.py` exactly as pre-registered for phase A.
* *Known gaps.* Section 6's model gaps still stand: α_l2 counted on top of the cross-GPU wait, α_l2 not measured, and
  streamed R that fits in L2. Section 14 added a per-GPU barrier to ring and doubling, and the model's α_l2 term stands
  for exactly that barrier. Any corrected model will be labelled post hoc.

**Part 2, the multi-GPU 2-D backward.** `hybrid_bw2d.cu` with `FRNN_MG = 1`:
* *Split.* The PC column blocks are split over P GPUs (P ∈ {2, 4} for PC = 8, every Q K block of a column block on
  its GPU). The partial sums therefore never leave a GPU, and barrier B1 stays a per-GPU grid barrier.
* *Exchange.* B2 adds the cross-GPU exchange of the multi-GPU 1-D kernel: dG is stored into every GPU's buffer, then a
  system fence, the grid barrier, one arrival per GPU and a wait for P arrivals.
* *Correctness expected.* The arithmetic is unchanged per GPU, so dG and d_initial must be bitwise equal to the
  single-GPU 2-D kernel with the same batch tile.
* *Registers.* ptxas: no spill in any multi-GPU build (D = 2048: 178 registers at BM = 16, 240 at BM = 64).
* *Runs:*
  * **H1**, bitwise against the single-GPU 2-D kernel: D ∈ {1024, 1536, 2048} × P ∈ {2, 4} × (B = 16, BM = 16;
    B = 64, BM = 16; B = 64, BM = 64) × LSTM/sLSTM, uva and symmetric buffers, T = 32.
  * **H2**, race verification (section 12 rule): P = 4 at D = 2048 (LSTM, BM = 16, B = 40), P = 2 at D = 1536 (sLSTM,
    BM = 64, B = 100), P = 4 at D = 1024 (sLSTM, BM = 16, B = 40), T = 48, 21 runs. Broken variants: no B1, no B2, and
    no cross-GPU wait.
  * **H3**, paired timing of the backward kernel at B = 64, T = 256, D ∈ {1536, 2048} × P ∈ {2, 4} × LSTM/sLSTM, BM = 16
    (P = 4 holds four batch tiles at once), and at B = 16, D = 2048, P = 4 (LSTM). Arms: one GPU with the 2-D kernel and
    its default tile, the multi-GPU 2-D kernel (uva), the multi-GPU 1-D kernel (uva), the multi-GPU 2-D kernel
    (symmetric). Reading: one GPU / multi-GPU 2-D and multi-GPU 1-D / multi-GPU 2-D with the bootstrap and the 1.02 /
    0.98 bands of section 11.
  * *Expectation.* At B = 64 the multi-GPU 2-D kernel beats one GPU with the 2-D kernel and beats the multi-GPU 1-D
    kernel. At B = 16 the gain is small or none.

## 18. Results of batch "mg2d" (2026-10-18)

Evidence: `scale-out-evidence/mg2d/`, archive verified by SHA-256.

**Part 1, phase A checks on the repaired forward.**
* **Check 2 (model): FAIL as pre-registered.** 7/37 configurations are within ±20%; 16/24 orderings hold.
* **Check 3 (crossover): accepted at D = 2048.** Resident slices on two GPUs take 19.7 µs per step against 33.7 µs
  for one GPU streaming R: 1.7x measured, ≥ 1.0 predicted.

The micro run came first in its process, with warm-up. Even so, pair 0-1 measured 36 µs at 16 B against 2.2-2.4 µs
for the other five pairs. So the warm-up of section 10 does not remove the effect every time. The first pair of a
process may also pay a one-time cost beyond compilation (a first-use peer-mapping cost is a guess, not verified).
`predict()` takes the slowest pair's α and β for every configuration, so every prediction is 2-5x too high.

*Post hoc, not a check.* With the median of the other five pairs as the link (α 2.3 µs, about 23 GB/s), 28/37
configurations fall within ±20%. The misses form three groups:
* Direct at P = 4 runs 1.36-1.45x slower than the model. Its publish phase is 8 µs against 4.5 at P = 2, so the three
  same-node stores of a CTA do not overlap as the "non-blocking crossbar" term assumes.
* D = 1536 at P = 3 is at 1.15-1.2x.
* The one-GPU stream at D = 2048 runs at 0.63x. Its R (33.5 MB) stays in L2 (section 6).

Every other configuration measures 1.1-1.2x the model. That systematic excess is the barrier/publish overhead the
model does not charge.

**Part 2, the multi-GPU 2-D backward.**
* **H1: 36/36.** dG and d_initial are bitwise equal to the single-GPU 2-D kernel with the same batch tile, on uva and
  symmetric buffers.
* **H2: 0/3 by the rule.** It fails because one broken variant was never flagged. The correct kernel counts zero
  violations and its outputs are stable in every run. No B1 and no cross-GPU wait are each flagged 21/21 and change the
  outputs, never silently. The "no B2" variant, however, was flagged in 0/21 runs and changed the outputs in 0/21. In
  the multi-GPU build that variant drops only the grid barrier and keeps the cross-GPU wait. That wait still holds
  every CTA until each GPU's CTA 0 has published, which in these runs was enough to order the exchange. The variant
  does not remove B2, so it tests nothing, and the race check of B2 in the multi-GPU build is open. The fix is to drop
  both parts of B2 in that variant.
* **H3 (backward kernel, paired, three starts):**

| case (B = 64 unless stated) | one GPU 2-D / multi-GPU 2-D | multi-GPU 1-D / multi-GPU 2-D | ms (one GPU 2-D / MG 2-D / MG 1-D) |
| --- | --- | --- | --- |
| LSTM D = 2048, P = 4 | 1.17 [1.17, 1.18] | 1.43 | 6.08 / 5.20 / 7.38 |
| sLSTM D = 2048, P = 4 | 1.10 [0.90, 1.23] | 1.23 | 6.65 / 5.45 / 7.47 |
| LSTM D = 1536, P = 4 | 0.88 [0.88, 0.89] | 1.17 | 4.92 / 5.58 / 6.48 |
| sLSTM D = 1536, P = 4 | 0.92 [0.92, 0.93] | 1.12 | 5.37 / 5.79 / 6.50 |
| LSTM D = 2048, P = 2 | 0.65 | 1.50 | 6.07 / 9.42 / 14.11 |
| sLSTM D = 2048, P = 2 | 0.69 | 1.50 | 6.67 / 9.65 / 14.44 |
| LSTM D = 1536, P = 2 | 0.51 | 1.25 | 4.90 / 9.67 / 12.09 |
| sLSTM D = 1536, P = 2 | 0.54 | 1.24 | 5.34 / 9.88 / 12.18 |
| LSTM D = 2048, P = 4, B = 16 | 0.71 | 1.56 | 2.80 / 3.93 / 6.14 |

Symmetric against uva buffers: not resolved in every row.

Against the expectation:
* The multi-GPU 2-D kernel beats the multi-GPU 1-D kernel everywhere (1.12-1.56x).
* It beats one GPU with the 2-D kernel only at D = 2048, P = 4: 1.17x for LSTM; for sLSTM the interval includes 1.
* At P = 2 and at D = 1536 it is slower. The batch tile of 16 rows that the split needs gives each CTA a 16-row GEMM
  (a quarter of the BM = 64 tile's work per R fragment read). At P = 2 the four tiles also run as two serial slots.

Splitting the backward over GPUs pays only where one GPU's grid is the limit, D = 2048 at P = 4. A better multi-GPU
layout would keep BM = 64 and split the K blocks rather than the column blocks; that needs cross-GPU partial sums. It
is not built.
