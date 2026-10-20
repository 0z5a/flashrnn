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
