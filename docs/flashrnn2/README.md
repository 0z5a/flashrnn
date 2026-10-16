# FlashRNN2 benchmark results

The [full-model E2E matrix](full-model-e2e-speedups.md) distinguishes completed
fixed-weight synthetic controls, the qualified TiRex 35M B2/C32 pretrained
forecast result, and pretrained-checkpoint results that remain unmeasured. The [cuDNN](h100-full-matrix-results.md) and
[FlashRNN1](h100-flashrnn1-results.md) tables give every completed batch and
concurrency case with throughput and a paired 95% interval.

The [85-case timing manifest](../../evidence/flashrnn2-h100-synthetic-timing-manifest.json)
and raw paired blocks are included. The [timing analyzer](../../tools/flashrnn2/analyze_public_stacked_timing.py)
recalculates every displayed speedup and interval. Saved numerical
qualification tensors remain in a private, hash-verified archive while their
metadata is prepared for release. This local branch does not claim that the
remaining pretrained or Haste high-concurrency E2E cases pass.

The latest [TiRex rounded-division result](../../evidence/tirex-rounded-division-20261011/README.md)
passes 381 model and 2048 B2/C32 diagnostic comparisons and measures a
20.0234× paired speedup against the original Torch cell on one H100. Its
20 within-process AB/BA blocks, raw timings and numerical metadata are included.
