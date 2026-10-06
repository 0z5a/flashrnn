# Mamba-130M high-batch native CPU oracle

The pinned `state-spaces/mamba-130m-hf` checkpoint at `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` is loaded separately by Transformers `MambaForCausalLM` and wrapped with the existing `MambaExecution` prefill/decode API. This uses the same pinned `modeling_mamba.py` implementation SHA-256 `ad28b5f1a464a64d1eabf96b42394f41ef322dbe5f4276e85d4f27368b61a796` and local Torch 2.14.1 runtime as the [high-batch native/trace gate](mamba-highbatch-cpu.md), but loads checkpoint weights independently of its B2 TorchScript transport. The author fused CUDA fast path was unavailable, so this is a Torch fallback control.

All runs use the [same fixed WikiText inputs](evidence/mamba-highbatch/mamba-highbatch-inputs-b16-b32-b64-r1.json): three disjoint groups per batch, 128-token prefill, 32 fixed greedy tokens and request-owned caches. The auditor verifies input rows and hashes, the oracle's 32 complete-vocabulary logit argmax steps per group, every generated token against both serial and interleaved trace schedules, and the first group's prefill/final full logits, convolution and SSM tensors against both native and traced arms. The original tolerances remain `1e-3 + 1e-3 * abs(reference)` for logits and `1e-5 + 1e-5 * abs(reference)` for caches.

| Batch | Oracle logit/argmax steps | Oracle token choices | Trace-schedule token comparisons | Full boundary tensor pairs | Max boundary error | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 96/96 | 1,536/1,536 | 3,072/3,072 | 12/12 | 0 | Pass |
| 32 | 96/96 | 3,072/3,072 | 6,144/6,144 | 12/12 | 0 | Pass |
| 64 | 96/96 | 6,144/6,144 | 12,288/12,288 | 12/12 | 0 | Pass |
| **Total** | **288/288** | **10,752/10,752** | **21,504/21,504** | **36/36** | **0** | **Pass** |

The [metadata and audits](evidence/mamba-highbatch-oracle/) pin the original `.pt` artifact hashes; the 3.86 GiB raw logits/cache oracles remain in the local workspace. The saved boundary pairs are independently recomputed from raw tensors. Intermediate full logits are independently checked for their argmax tokens, but are not compared elementwise with the trace because the trace saved full tensors only at the two stated boundaries.

| Required full-model GPU E2E comparison | Workload | Native accelerated tok/s | Candidate tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| `mamba_ssm` fused path versus Torch fallback, Script and FlashRNN candidate | B16/32/64, concurrency 16/32/64/128, P128/G32 | — | — | Unmeasured |
| Hopper SM90 / B200 SM100 paired full-model prefill and decode | Same input and output contract | — | — | Unmeasured |

The independent checkpoint-loading path strengthens the CPU reference, but does not qualify an accelerated Mamba backend, GPU output parity, backward or high-concurrency E2E throughput.
