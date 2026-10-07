# TiRex 35M forecast E2E gate

This benchmark requires a completed GPU qualification of the pinned TiRex checkpoint against the official Torch forecast. The gate checks B=1/2/4, every cell output and four states in all 12 blocks and two forecast patches, plus the 64-step quantiles and median. Its tolerance is the original reference budget, `1e-4 × (1 + |reference|)`. A missing, failed, partial, differently pinned, or wider-tolerance gate cannot start timing.

The timed operation is the public `forecast` call with 128 input points and 64 forecast points. It includes preprocessing, both recurrent passes, all 12 blocks, quantiles, and CPU forecast output. Checkpoint loading, compilation, qualification, fixture generation, and warmup are excluded. The official Torch model and the candidate remain resident together; memory is not attributed to either arm.

Choose B from 1, 2, and 4, with concurrent-series count C divisible by B. All C series arrive at once and one worker processes queued B-sized groups in order. The reported throughput is completed series/s, not tokens/s or simultaneous GPU streams. Each case uses at least 20 within-process AB/BA paired blocks. The analyzer verifies the raw checksum, complete response timestamps, arm order, numerical-gate provenance, and paired 95% bootstrap interval before writing a Markdown speed row. A failed numerical check or incomplete block cannot become a speedup.

GPU qualification and speed results are pending. No pretrained-model throughput claim follows from the separate synthetic RNN tables.
