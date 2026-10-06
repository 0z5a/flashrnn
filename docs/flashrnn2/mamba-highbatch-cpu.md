# Mamba-130M high-batch complete-model CPU generation gate

The pinned `state-spaces/mamba-130m-hf` checkpoint at `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` supplies 129,135,360 parameters across 24 recurrent layers. Its native Transformers Torch fallback and a trace made from that same implementation run with the same fixed weights in the existing local Torch 2.14.1 CPU runtime. The trace is created separately at each declared batch shape; the existing B2 TorchScript file transports weights into the dynamic native model. Neither path is the accelerated `mamba_ssm` CUDA baseline.

The [input file](evidence/mamba-highbatch/mamba-highbatch-inputs-b16-b32-b64-r1.json) selects the first 192 WikiText validation rows that tokenize to at least 128 tokens, then forms three disjoint 64-request groups. B16 and B32 use the first 16 or 32 rows of each group. The dataset and tokenizer SHA-256 values are fixed in that file; the [input audit](evidence/mamba-highbatch/input-audit-r1.json) checks 192 unique rows, valid IDs, batch prefixes and exact agreement of the first four prompts with the prior B4 group. Each request has a 128-token prefill and 32 greedy output tokens; all three groups run serially and then round-robin interleaved with independent convolution/SSM caches.

The unchanged elementwise budgets are `1e-3 + 1e-3 * abs(reference)` for full-vocabulary logits and `1e-5 + 1e-5 * abs(reference)` for both caches. Every row compares native and traced outputs, not just generated tokens. The independent audit verifies the input hash, all saved token rows, serial/interleaved token parity, the raw snapshot hash and six full boundary tensor pairs per batch. Intermediate tensors are represented by the runner's recorded checks, not independently reconstructed from raw tensors.

| Model batch | Complete-model steps | Token choices | Runner logit/cache checks | Audited boundary tensor pairs | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 192/192 | 3,072/3,072 | 576/576 | 6/6 | Pass |
| 32 | 192/192 | 6,144/6,144 | 576/576 | 6/6 | Pass |
| 64 | 192/192 | 12,288/12,288 | 576/576 | 6/6 | Pass |
| **Total** | **576/576** | **21,504/21,504** | **1,728/1,728** | **18/18** | **Pass** |

The [run records and audits](evidence/mamba-highbatch/) retain exact source, input and snapshot hashes. Committed metadata omits duplicated prompt arrays and records the SHA-256 of each full local metadata file. Large raw tensor snapshots stay in the local workspace, not in Git. This check is a same-implementation trace and request-isolation gate; it is not an independent native-output oracle or a performance result.

| Required full-model GPU E2E comparison | Workload | Native baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Author Mamba fused kernels versus Torch fallback, Script and FlashRNN candidate | B16/32/64, concurrency 16/32/64/128, P128/G32 | — | — | Unmeasured |
| Hopper SM90 and B200 SM100 full-model prefill/decode | Same paired request trace | — | — | Unmeasured |

Native accelerated CUDA, backward and high-concurrency GPU E2E still require a granted GPU window and same-device qualification. CPU wall time from this correctness gate is not used as a speedup.
