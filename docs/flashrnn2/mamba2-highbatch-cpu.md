# Mamba2-130M high-batch complete-model CPU gate

The pinned complete `state-spaces/mamba2-130m` checkpoint is loaded through the strict Transformers mapping. Its native Torch fallback is compared with the existing B1/P128 TorchScript artifact on the [same fixed WikiText requests](evidence/mamba-highbatch/mamba-highbatch-inputs-b16-b32-b64-r1.json). Every step compares full-vocabulary logits and generated IDs; each prefill/final step also compares all-layer convolution and SSM caches at the original `1e-3` logits and `1e-5` cache absolute/relative budgets. Three request groups run serially and round-robin interleaved with independent states.

| Batch | Serial steps | Interleaved steps | Token choices | Runner logit/cache checks | Saved boundary tensor pairs | Max absolute error | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 96/96 | 96/96 | 3,072/3,072 | 216/216 | 6/6 | 0 | Pass |
| 32 | — | — | — | — | — | — | Not run |
| 64 | — | — | — | — | — | — | Not run |

The B16 runner naturally exited 0 after 192/192 complete-model steps; both schedules generated identical tokens. The [independent audit](evidence/mamba2-highbatch/mamba2-highbatch-cpu-b16-r1.audit.json) verifies row order, all 3,072 token choices, 216 reported tensor checks, input and TorchScript hashes, and recomputes six full-logit/convolution/SSM boundary tensor pairs from the SHA-256-pinned 1.2 GiB local snapshot. The [raw rows and metadata](evidence/mamba2-highbatch/) are committed; the large snapshot remains local. All checked tensor elements match bitwise. The B1 artifact's cross-batch behavior beyond B4 was previously untested. Local 16 GiB memory used over 5 GiB swap during B16, so B32/B64 multi-group runs await a larger device.

| Required full-model GPU E2E comparison | Workload | Native accelerated tok/s | Candidate tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Mamba2 author fused backend versus Torch fallback, Script and FlashRNN candidate | B16/32/64; concurrency 16/32/64/128; P128/G32 | — | — | Unmeasured |
| Hopper SM90 / B200 SM100 paired prefill/decode | Same request and output contract | — | — | Unmeasured |

This gate uses the unaccelerated Transformers path; it does not replace the [retained cached/full-prefix failures](mamba2-qualification.md) or establish native CUDA, backward or GPU E2E performance.
