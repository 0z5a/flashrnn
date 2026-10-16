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
