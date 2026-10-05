# Pure GDN-2 305M checkpoint: complete-model CPU reference

The public [Loser-GDN-2-305M checkpoint](https://huggingface.co/Jellyfish042/Loser-GDN-2-305M-20260601) at revision `2d98aff017d7a9e125a14dd02e8c4b6f631930a8` strictly loads all 267 model tensors into a CPU PyTorch adapter of the author's pinned [GatedDeltaNet-2 source](https://github.com/Jellyfish042/GatedDeltaNet-2/tree/7efb255b69411084e7e3c0e13d5083910d3258a7). The adapter covers the full 12-layer, 16-head, 64-dimensional-head language model: embeddings, both residual paths, short Q/K/V convolutions, channel-wise decay/erase/write gates, the pinned [FLA naive GDN-2 recurrence](https://github.com/fla-org/flash-linear-attention/blob/cbb0a72efb55c18ca0ef4f298298317573ad2cb3/fla/ops/gdn2/naive.py), gated RMS normalization, SwiGLU MLPs, final norm and full 32,000-token head. It uses the TinyLlama tokenizer at `ff3c701f2424c7625fdefb9dd470f45ef18b02d6` with an explicit BOS token.

The checkpoint has **304,809,920** parameter values. The model card prints **239,272,896**; this report uses the count from the actual 267 checkpoint tensors. The published checkpoint is a PyTorch training checkpoint, not a Transformers `from_pretrained` package. The author's source imports CUDA-only packages that were unavailable in the existing local environment, so this is a source-mapped CPU reference, not a run of the native GPU implementation. No environment was changed.

At FP32 and a frozen elementwise budget of `1e-4 + 1e-4 × abs(reference)`, 12 five-token prompts and four greedy choices per case passed full-vocabulary batch-versus-serial and cached-versus-full-prefix checks. Final cached states were compared both with individually decoded states and with states from a full-prefix recomputation, across all 12 recurrent matrices and 36 short-convolution histories per batch.

| Batch | Full-vocabulary steps | Token choices | Cached/serial state pairs | Cached/full state pairs | Batch/serial max abs | Cached/full logit max abs | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 12/12 | 12/12 | 144/144 | 144/144 | 0 | 0.0000238419 | Pass |
| 2 | 12/12 | 24/24 | 288/288 | 144/144 | 0.0000181198 | 0.0000247955 | Pass |
| 4 | 12/12 | 48/48 | 576/576 | 144/144 | 0.0000219345 | 0.0000247955 | Pass |

The [independent audit](evidence/gdn2-pure-reference/gdn2-pure-reference-r2.audit.json) recomputed all 36 complete-vocabulary comparisons, 84 token choices, 1,008 cached/serial state pairs and 432 cached/full state pairs from the 334,601,865-byte local binary snapshot. All failed-element counts are zero. Maximum absolute recurrent/convolution-state errors are 0.0000147820 for cached/serial and 0.0000152588 for cached/full. The snapshot SHA-256 is `3c3b953157d85c785cf72e8c2d0183c04c022a2b27f73dd6995741b757130907`. The [raw rows](evidence/gdn2-pure-reference/gdn2-pure-reference-r2.jsonl), [run metadata](evidence/gdn2-pure-reference/gdn2-pure-reference-r2.meta.json), [source and weight manifest](evidence/gdn2-pure-reference/gdn2-pure-reference-source-manifest.json), and [weight-eviction receipt](evidence/gdn2-pure-reference/gdn2-pure-weight-eviction-r2.json) preserve the exact scope and hashes. The large tensor snapshot stays in the local task workspace and is not committed.

| Required full-model GPU E2E comparison | Batch / concurrency | Baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Author/FLA fused recurrent versus FLA chunk GDN-2 | BS1/4/16/32/64 | — | — | Unmeasured |
| Native GDN-2 versus a FlashRNN2 GDN-2 path | BS1/4/16/32/64 | — | — | Path unimplemented; unmeasured |
| Other recurrent-model baselines under the same request trace | concurrency 1/8/32/64/128 | — | — | Unmeasured |
| Hopper SM90 / B200 SM100 end-to-end | prefills and decoding | — | — | Unmeasured |

This checkpoint adds pure GDN-2 complete-model CPU coverage. It does not establish native CUDA parity, BF16 equivalence, backward correctness, long-context behavior, high-concurrency throughput or any GPU speedup. The 1,219,361,090-byte weight was deleted after the completed gate, independent audit, SHA-256 and no-reader checks; pinned redownload is required for later GPU measurement.
