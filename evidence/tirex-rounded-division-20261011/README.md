# Rounded division TiRex qualification

This measures commit `12eded4` on one H100 with the pinned TiRex 35M model, B2/C32, the original Torch cell baseline, and 20 alternating AB/BA blocks in one process. The summary is reproducible from the included raw timings. The 95% bootstrap interval describes these paired blocks, not independent process restarts. Numerical qualification passes 381 model comparisons and 2048 B2/C32 diagnostic comparisons, retaining the original `1e-4` budget.

The TiRex-only rounded divisions fix BF16 hidden-state differences without changing the xLSTM path. The exact tested kernel SHA-256 is `4b25ef542b80ab0e0ac7cf17b3cde891a0efcb5932866d9fc7dfef6577bb894c`. Checkpoint loading, compilation, qualification, and warmup are excluded from timing; preprocessing, both forecast patches, all 12 blocks, quantiles, and CPU output are included. This result does not qualify other batches or general training.
