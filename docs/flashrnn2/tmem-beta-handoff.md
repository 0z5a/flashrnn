# TMEM beta: cluster forward on Blackwell tensor cores (handoff)

Branch `flashrnn2-tmem-beta-20261020` of the dev repo `flashrnn2-pr-20261014`. Files:
`flashrnn/flashrnn2/hybrid_fwc_tmem.cu`, `flashrnn/flashrnn2/hybrid_fwc_tmem.py`. Status: **compiles for sm_100a
with nvcc 12.8 (188-190 registers, no spills); never run.** nvcc 12.8 does not know sm_103a (B300); CUDA 12.9 or later
is needed for `-arch=sm_103a`, or build sm_100a and rely on forward compatibility if the driver accepts it.

## What it is

The H100 cluster forward (`hybrid_fwc.cu`, max-width.md section 80: 16-CTA cluster, round(h) exchanged through
distributed shared memory, one cluster barrier per step, Wx prefetched during the barrier, bias hoisted) with the
per-step GEMM moved to `tcgen05.mma` with the accumulator in tensor memory. Target shape: B = 16, T = 256, D = 512,
LSTM FP16 and sLSTM BF16, hidden output. The comparison that matters is per step against cuDNN's recurrent kernel
(1.61 µs at D = 512 on H100; measure cuDNN on the Blackwell host too) and against `hybrid_fwc` on the same host.

## What is unverified (in order of risk)

1. **Descriptor bit layouts.** `smem_desc()` (shared-memory matrix descriptor: start >> 4 at [0,14), LBO >> 4 at
   [16,30), SBO >> 4 at [32,46), layout type 0 at [61,64)) and `inst_desc()` (kind::f16: D format at [4,6), A/B format
   at [7,10) / [10,13), transpose bits 15/16 = 0 for K-major, N >> 3 at [17,23), M >> 4 at [24,29)) follow the PTX ISA
   8.7 tables from memory. Check them against the ISA text, or against what CUTLASS 3.x / `cute::UMMA` emit for
   `SmemDescriptor` and `InstrDescriptor` (K-major, SWIZZLE_NONE, F16 inputs, F32 accumulator).
2. **The canonical no-swizzle K-major layout.** `canon(m, k) = (m / 8) * SBO + (k / 8) * LBO + (m % 8) * 16 +
   (k % 8) * 2` with LBO = 128 and SBO = D * 16. If the MMA output is wrong but finite, swap the roles of LBO and SBO
   first, then try the "leading byte offset" convention of the ISA for K-major (some versions define LBO along M/N).
3. **M = 64 with a 16-row A buffer.** The accumulator has 64 lanes; rows 16..63 read past the buffer into the following
   shared memory (R) on purpose; the products land in lanes 16..63, which are never read. If `compute-sanitizer`
   objects or the hardware faults, allocate full 64-row A buffers (64 x D x 2 bytes each, 128 KB for the pair at
   D = 512) and drop R to registers as in `hybrid_fwc.cu` — or use `.kind::f16` with cta_group::2 and M = 128 split
   over two CTAs, which is a larger redesign.
4. **Tensor-memory read-out.** Warp 0 reads lanes 0..31 with `tcgen05.ld.sync.aligned.32x32b.x32`; lane i must be row
   i (layout "32x32b": thread i of the warp gets lane i). If the pointwise results are permuted, print one step's
   `spart` against a CPU reference to find the lane mapping.
5. **Proxy fences.** Generic-proxy stores (R once, round(h) every step, by peers through DSMEM) feed async-proxy MMA
   reads; `fence.proxy.async.shared::cta` after the cluster wait is meant to order them. If results are stale by one
   step, the fence placement or scope is the suspect (the peers' stores are remote; a cluster-scope fence may be needed
   before the arrive on the writer's side).
6. **`tcgen05.commit` scope.** `mbarrier::arrive::one.shared::cluster` on a CTA-local mbarrier address; the ISA may
   require the shared::cta address form.

## Checklist

1. Build: `FRNN_TMEM_ARCH=sm_100a` (or sm_103a) `python -c "from flashrnn.flashrnn2 import hybrid_fwc_tmem as m;
   m.build(512, 16, False, False, 1, 0)"`.
2. Correctness: `tools/fw1d_bench.py` has no TMEM arm yet; add one next to the `fwc-*` arms (call
   `hybrid_fwc_tmem.recurrence(..., output_mode="hidden")`) and compare `max_abs_vs_tilelang` (expect ≤ 1e-3 for a
   different accumulation order; much more means a layout or descriptor error) for LSTM FP16 and sLSTM BF16 at
   D = 512.
3. `compute-sanitizer --tool memcheck` on one call (T = 8).
4. Timing: the bench's per-step time and phase timers (barrier / GEMM / pointwise) against `hybrid_fwc` push and pull on
   the same GPU.
5. Expectation to test: the GEMM phase of `hybrid_fwc` is 0.3-0.5 µs per step with mma.sync; tcgen05 removes the
   dependent-chain latency and the register pressure of R fragments (R lives in shared memory), so the gain, if any, is
   small at D = 512 and larger where `hybrid_fwc` ran out of registers (D = 768 at 255 registers). D = 768 does not fit
   this kernel's shared memory (R 288 KB), so a D = 768 TMEM path would need the A/B operands split differently.

Nothing in this branch changes a default or a shipped kernel.

## Update 2026-10-20 (after the first B300 run)

* **Step-0 error, decaying over steps (B300 run 1, all four variants): cause found and fixed.** The prologue zeroed both
  parity buffers with a loop that nothing ordered against the peers' round(h_0) DSMEM pushes into this CTA's buffer,
  so a fast peer's h_0 could be overwritten with zeros; step 0 then used a partial h_0 and the recurrence damped the
  error. The loop is removed (every element of both buffers is written by its owner before it is read; owners of rows
  beyond B publish zeros). `hybrid_fwc.cu` never had the loop.
* **Build form.** `build()` now emits `-gencode=arch=compute_<N>a,code=sm_<N>a` instead of `-arch=sm_<N>a`: with nvcc
  13.3 the latter also embeds plain `compute_103` PTX, which cannot hold tcgen05 (63 ptxas errors).
* The `extern "C"` entry gained trailing multi-GPU arguments (`mc_x, uc_x, mc_flag, uc_flag, rank, flag_base`); the
  Python wrapper passes nulls for single-GPU calls.

## Multi-GPU over NVSwitch multicast (FRNN_P > 1; branch `flashrnn2-tmem-mg-20261020`)

Written at the user's request ("NVSwitch 多卡 multicast，基于现在的 tmem 版"), without access to any NVSwitch machine;
nothing below has run. Files: `hybrid_fwc_tmem.cu` (`FRNN_P`), `hybrid_fwc_tmem.py` (`recurrence_mg`, `default_cluster`,
`supported(width, cluster, peers)`), `nvls_mem.py` (the multicast object), `nvls_probe.cu` + `tools/nvls_probe.py`
(stand-alone probe), `tools/tmem_mg_bench.py`.

**Design.** P GPUs of one process split the units: GPU `rank` owns [rank D / P, (rank + 1) D / P), its C CTAs NU =
D / (P C) each; every CTA still holds the whole round(h) [16 x D] as the A operand, so the per-step work is an
all-gather of P slices. Per step an owner CTA publishes its block twice: into its local peers' A buffers through DSMEM
(as before) and, element by element, into a **multicast-mapped global exchange buffer** (`nvls_mem.MulticastGroup`:
`cuMulticastCreate` / `AddDevice` / `BindMem`; a store to the multicast VA is replicated by the NVSwitch into every GPU's
copy, the sender's included). After `__syncthreads` + `__threadfence_system` one thread releases the CTA's step flag
(`st.release.sys`, also through the multicast mapping; value flag_base + step + 1, rising across calls so nothing is
reset). Before its GEMM a CTA passes the local cluster barrier, then polls the (P − 1) C remote flags of the step on
its **local** copy (`ld.acquire.sys`, 2^26-spin cap → err[1]), fences the proxy, arms an mbarrier and pulls each remote
GPU's slice from its local copy of the exchange buffer with **one `cp.async.bulk` per remote GPU**. For that copy to
be contiguous, multi-GPU builds use the K-core-major canonical layout for A (LBO_A = 256, SBO_A = 128; a slice of
D / P columns is (D / P) · 32 contiguous bytes) in both shared memory and the exchange buffer; the B operand keeps
LBO 128 / SBO 16 D. Supported: D = 512 with P = 2 (C 16) / 4 (C 8) / 8 (C 4); D = 1024 with P = 4 (C 16) / 8 (C 8) —
D = 1024 does not fit one GPU's shared memory in this kernel at all, so P ≥ 4 is what makes it reachable.

**What the probe decides first** (`tools/nvls_probe.py --devices 4,5,6,7`; no TMEM involved):
1. `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` on every device, and `cuMulticastCreate` + bind + map succeed (prints the
   granularity; expected 2 MB).
2. Correctness of the protocol the kernel uses (data stores → fence.sys → release flag through the multicast mapping;
   acquire poll on the local copy; 2000 iterations × P blocks of 8 KB): **0 mismatched words**. A nonzero count means
   multicast delivery is not ordered with the flag the way peer stores are — then the kernel needs a different signal
   (e.g. a multimem.red counter or a second fence) before it can be trusted.
3. ns per all-gather step, multicast vs unicast fan-out. Expectation: the unicast time grows with P (P − 1 remote
   store streams per GPU), the multicast time does not; the multicast absolute (one NVLink hop + switch replication,
   likely 1–2 µs) is the floor the kernel's `remote` phase inherits.

**Then the kernel** (`tools/tmem_mg_bench.py --devices 4,5,6,7 --widths 512,1024`), after the single-GPU TMEM kernel is
correct (its fix above): `mg-p2-c16` at D = 512 against `tmem-c16` and TileLang on one GPU (max_abs ≤ 1e-3 class;
anything larger is a layout error in the K-core-major A path, since the single-GPU build keeps the old layout);
then `mg-p4-c16` at D = 1024. Phase timers per rank: `barrier` (local cluster), `remote` (flag poll + slice pull),
`gemm`, `pointwise`. Expectation, stated before any run: per step ≈ max(local, remote) with `remote` 1.5–3 µs, so the
P-GPU step is slower than the one-GPU step at D = 512 (this is a reach-D experiment, not a speed-up), and at D = 1024
it is the only TMEM number there is; the comparison that matters there is against the single-GPU TileLang forward at
D = 1024 (≈ 4 µs/step on H100) and against `multi_gpu.recurrence` (uva / nccl_symm) if Triton runs on the host.

**Unverified assumptions (in order of risk):** (a) ordering of multicast data stores before the multicast flag store
under `__threadfence_system` (the probe tests exactly this); (b) `cuMemMap` of the multicast handle at one VA with
access for all P devices in one process (NCCL maps per process; one-process multi-device is the documented but less
travelled path); (c) a generic `st.global` to the multicast VA replicates (documented; NCCL uses `multimem.st`); (d)
the `fence.proxy.async` between the acquire poll and the bulk copy is sufficient for data written by another device;
(e) descriptor strides SBO_A = 128 < LBO_A = 256 are accepted by tcgen05 (any 16-byte multiple should be).
