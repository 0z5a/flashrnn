# TiRex 35M forecast E2E gate

This benchmark requires a completed GPU qualification of the pinned TiRex checkpoint against the official Torch forecast. The gate checks B=1/2/4, every cell output and four states in all 12 blocks and two forecast patches, plus the 64-step quantiles and median. Its tolerance is the original reference budget, `1e-4 × (1 + |reference|)`. A missing, failed, partial, differently pinned, or wider-tolerance gate cannot start timing.

The timed operation is the public `forecast` call with 128 input points and 64 forecast points. It includes preprocessing, both recurrent passes, all 12 blocks, quantiles, and CPU forecast output. Checkpoint loading, compilation, qualification, fixture generation, and warmup are excluded. The official Torch model and the candidate remain resident together; memory is not attributed to either arm.

Choose B from 1, 2, and 4, with concurrent-series count C divisible by B. All C series arrive at once and one worker processes queued B-sized groups in order. The reported throughput is completed series/s, not tokens/s or simultaneous GPU streams. Each case uses at least 20 within-process AB/BA paired blocks. The analyzer verifies the raw checksum, complete response timestamps, arm order, numerical-gate provenance, and paired 95% bootstrap interval before writing a Markdown speed row. A failed numerical check or incomplete block cannot become a speedup.

The separate `audit_tirex_gpu_gate.py` reads the saved gate PT tensors and recomputes all 381 comparisons with NumPy float64 arithmetic. Torch only decodes the PT storage. It checks the raw checksum, pinned checkpoint/source, original tolerance, complete scope and reported pass flags. The timing runner requires a passing audit JSON tied to the exact gate and raw hashes. Preserve that JSON alongside the gate PT/JSON and actual process exits.

The first pinned GPU qualification [failed numerically](tirex-gpu-r2-results.md)
at all 381 required tensor pairs. The corrected-layout
[r3 qualification](tirex-gpu-r3-results.md) still failed 307 of 381 pairs.
[r4 qualification](tirex-gpu-r4-results.md) reduced failures to 206 of 381;
the B2 path passed exactly, but B1 and B4 full forecasts did not. The
[guarded r5 qualification](tirex-gpu-r5-results.md) passed 381 of 381 at the
original budget, with B2 accelerated and B1/B4 using the official Torch cell.
Forecast speed and pretrained-model throughput remain unmeasured. No
pretrained-model throughput claim follows from the separate synthetic RNN tables.

The qualified candidate enables FlashRNN2 only for B2 and retains the official
Torch cell at B1 and B4. Its E2E table will identify this dispatch for every
batch. B1/B4 timing will measure the
fallback overhead against Torch, not a FlashRNN2 kernel speedup.
The analyzer carries the candidate path into both its JSON summary and the
Markdown table, with B2 identified as FlashRNN2 and other batches identified
as Torch fallback.
