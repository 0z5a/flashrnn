# RWKV6 ordered chunk-prefix CPU qualification

The [checkpoint-derived gate](../../tools/flashrnn2/rwkv6_chunk_gate.py) applies the pinned RWKV6 1.6B first block's two norms and attention projections to four synthetic 134-token streams assembled from recorded prompt IDs. It compares ordered `(D,U)` chunk summaries and boundary-state injection with pinned FLA FP32 serial recurrence, including readout from the pre-update state plus current-token bonus. Resumed cases use a nonzero five-token state.

| Batch | Steps | Incoming state | FP32 output pass | FP32 state pass | Independent FP64 control |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 5 | zero | 4/4 | 4/4 | 4/4 |
| 2 | 17 | zero | 4/4 | 4/4 | 4/4 |
| 4 | 127 | zero | 2/4 | 4/4 | 4/4 |
| 4 | 128 | nonzero | 1/4 | 4/4 | 4/4 |
| 4 | 129 | nonzero | 1/4 | 4/4 | 4/4 |
| **Total** | | | **12/20** | **20/20** | **20/20** |

FP32 uses the original `1e-4 + 1e-4 * abs(reference)` budget. Its 8 failed output rows contain 110 failing elements; maximum absolute error is 0.009765625 and worst normalized error is 2.78676. The separate FP64 serial control preserves FP64 arithmetic, whereas upstream FLA's reference converts its inputs to FP32; all 20 FP64 cases pass `atol=1e-8, rtol=1e-10`, with maximum output error 1.45519e-11. This validates the affine summary algebra, while the changed FP32 operation order remains unqualified. Bonus readout changes output but not final state. The runner exits 1 for the FP32 failure.

| Required E2E speed comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native RWKV6 full-model GPU generation | — | — | Unmeasured |
| C1/C8/C32 serving | — | — | Unmeasured |

[Raw rows](evidence/rwkv6-chunk-layer0-r5.jsonl), [run metadata](evidence/rwkv6-chunk-layer0-r5.meta.json), and [row/hash audit](evidence/rwkv6-chunk-layer0-r5.audit.json) record the exact result. A separate identical-input run generated bitwise-identical raw rows. No native FLA chunk, CUDA kernel, backward or high-concurrency GPU E2E was timed; synthetic long streams are not a language-model rollout.
