# Mamba2-130M high-batch complete-model CPU gate

The pinned complete `state-spaces/mamba2-130m` checkpoint is loaded through the strict Transformers mapping. Its native Torch fallback is compared with the existing B1/P128 TorchScript artifact on the [same fixed WikiText requests](evidence/mamba-highbatch/mamba-highbatch-inputs-b16-b32-b64-r1.json). Every step compares full-vocabulary logits and generated IDs; each prefill/final step also compares all-layer convolution and SSM caches at the original `1e-3` logits and `1e-5` cache absolute/relative budgets. B16 runs three independent request groups; the local B32 memory-limited gate runs the first group only.

| Batch | Request groups | Serial steps | Interleaved steps | Token choices | Runner logit/cache checks | Saved boundary tensor pairs | Max absolute error | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 3 | 96/96 | 96/96 | 3,072/3,072 | 216/216 | 6/6 | 0 | Pass |
| 32 | 1 | 32/32 | 32/32 | 2,048/2,048 | 72/72 | 6/6 | 0 | Pass |
| 64 | — | — | — | — | — | — | — | Not run |

Both runners naturally exited 0. The [B16 audit](evidence/mamba2-highbatch/mamba2-highbatch-cpu-b16-r1.audit.json) and [B32 audit](evidence/mamba2-highbatch/mamba2-highbatch-cpu-b32-g1-r1.audit.json) verify row order, every token choice, all reported checks, input and TorchScript hashes, and recompute six full-logit/convolution/SSM boundary tensor pairs per batch. The B32 audit also checks the exact gate-source hash. Their [raw rows and metadata](evidence/mamba2-highbatch/) are committed; the 1.2 GiB and 2.4 GiB SHA-256-pinned snapshots remain local. All checked tensor elements match bitwise. The B32 one-group serial/interleaved orders are identical, so its schedule parity is not a multi-group concurrency test. B32/B64 three-group and GPU runs still require a larger device.

| Required full-model GPU E2E comparison | Workload | Native accelerated tok/s | Candidate tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Mamba2 author fused backend versus Torch fallback, Script and FlashRNN candidate | B16/32/64; concurrency 16/32/64/128; P128/G32 | — | — | Unmeasured |
| Hopper SM90 / B200 SM100 paired prefill/decode | Same request and output contract | — | — | Unmeasured |

This gate uses the unaccelerated Transformers path; it does not replace the [retained cached/full-prefix failures](mamba2-qualification.md) or establish native CUDA, backward or GPU E2E performance.
