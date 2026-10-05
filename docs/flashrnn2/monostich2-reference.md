# Monostich-2-base complete-checkpoint CPU reference

The public `kerzgrr/Monostich-2-base` checkpoint at `7779182b4dfad5939827f40cd1c819843a6825e2` strictly loads 586 tensors into 149,108,320 parameters. All 32 author TinyGDN blocks execute: 24 GDN-2 recurrence blocks and eight GQA full-attention blocks. The CPU adapter uses pinned FLA `naive_recurrent_gdn2` and matching PyTorch short convolution, q/k L2 normalization and gated RMSNorm. The author inference entry point auto-installs FLA, so the gate imports the pinned body directly without changing the environment.

At BS1/2/4, 12 shared five-token prompts and four greedy steps per case produce 36 complete-prefix batched-versus-serial full-vocabulary comparisons and 84 matching token choices. The comparison budget is `1e-4 + 1e-4 * abs(reference)`.

| Batch | Cases | Logits pass | Tokens agree | Maximum absolute logit difference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 12/12 | 12/12 | 0 |
| 2 | 3 | 12/12 | 24/24 | 0.0000104904 |
| 4 | 3 | 12/12 | 48/48 | 0.0000123978 |

The [independent audit](evidence/monostich2-reference-r1.audit.json) recomputes all saved complete logits and tokens from hashed local binary snapshots. The [pinned official GDN-2 layer body crosscheck](evidence/monostich2-layer-crosscheck-r1.json) compares the same 17 checkpoint tensors and is bitwise at BS1/T5, BS2/T5 and BS4/T17 using CPU operator substitutes. [Raw rows](evidence/monostich2-reference-r1.jsonl), [run metadata](evidence/monostich2-reference-r1.meta.json) and the [source/checkpoint manifest](evidence/monostich2-reference-source-manifest.json) preserve the scope.

| Required E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native FLA GDN-2 full-model GPU | — | — | Unmeasured |
| GPU BS1/2/4 generation | — | — | Unmeasured |
| C1/C8/C32 serving | — | — | Unmeasured |

This is full-model CPU complete-prefix correctness. Native FLA GPU, cached generation, high-concurrency E2E and backward remain unqualified. The 298 MB weight was evicted after checksum, audit and no-reader checks; its pinned SHA-256 remains in the manifest for GPU redownload.
