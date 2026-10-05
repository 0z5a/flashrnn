# RWKV6 BlinkDL forward crosscheck

The full reverse-mapped RWKV6 1.6B checkpoint completes three B1 prompt groups
with five input tokens and four continuation tokens through BlinkDL's original
CPU forward bodies. All 12 logit comparisons and greedy token choices pass,
but none of the 72 saved final cache tensor pairs passes the existing budget.
Matching normalization epsilons resolves the attention-shift and FFN-shift
failures in this case; all 24 recurrent states still fail. Both model runs
naturally exit 1, and both independent saved-tensor audits exit 0.

| Comparison | Original passing / total | Matched-epsilon passing / total | Original max error | Matched-epsilon max error |
| --- | ---: | ---: | ---: | ---: |
| Full vocabulary logits, all groups | 12/12 | 12/12 | 0.0018234253 | 0.0001068115 |
| Greedy token choices | 12/12 | 12/12 | — | — |
| Recurrent final state, group 0 | 0/24 | 0/24 | 0.0356445313 | 0.0053710938 |
| Attention-shift final state, group 0 | 0/24 | 24/24 | 0.0003435612 | 0.0000007525 |
| FFN-shift final state, group 0 | 0/24 | 24/24 | 0.0002355576 | 0.0000038147 |

Budgets remain `1e-3 + 1e-3 * abs(reference)` for logits and
`1e-5 + 1e-5 * abs(reference)` for states. Both paths consume the same saved
FLA continuation tokens, while reporting their independent greedy choices.
This is a fixed-input comparison, not independently diverging generation.

All 582 FP32-promoted checkpoint tensors (1,599,873,024 parameters) are mapped
to the original layout without another complete weight copy. Each of the
24 layers' two low-rank mixing projections passes a bitwise roundtrip check
for the FLA r/w/k/v/g versus original w/k/v/r/g ordering. Every checkpoint key
is consumed. Embedding and vocabulary head remain untied.

The four forward method bodies come unchanged from
[BlinkDL/ChatRWKV](https://github.com/BlinkDL/ChatRWKV/blob/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/RWKV_v6_demo.py).
Construction/weight loading and TorchScript decorators are replaced; demo
initialization and its tokenizer are not executed. The reverse mapping follows
[FLA's pinned converter](https://github.com/fla-org/flash-linear-attention/blob/17dd5662554d46b6bcb1d1ff728cebb461c9aef9/utils/convert_from_rwkv6.py).
The checkpoint remains the FLA-converted Finch artifact qualified in the
[preceding reference](rwkv6-reference.md), with fixed saved token IDs. This does
not independently qualify an original BlinkDL checkpoint, tokenizer or CUDA
implementation.

BlinkDL uses LayerNorm epsilon `1e-5` and GroupNorm epsilon `6.4e-4`, whereas
the saved FLA configuration uses `1e-6` for both. The separate control changes
only those epsilons to `1e-6`; the original failed result remains intact.
The residual recurrent-state error cannot be explained by epsilon alone.

The [original audit](evidence/rwkv6-blinkdl-r1-audit.json) and
[control audit](evidence/rwkv6-blinkdl-norm-r1-audit.json) each independently
recompute all 12 saved logit pairs and 72 final cache pairs, verify token
choices and input IDs, and check source/oracle/snapshot hashes. Final state
comparisons cover group 0 only. The interrupted tool session for the control
was unavailable on resume; its controller's actual child wait, completed
metadata, saved tensors and absent PIDs confirm completion without a rerun.

| Required performance comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native GPU generation | — | — | Not measured |
| Multiple batch sizes / high concurrency E2E | — | — | Not measured |

Run either `tools/flashrnn2/rwkv6_blinkdl_gate.py` or the separate
`rwkv6_blinkdl_norm_control.py` with `--model`, `--source`, `--oracle` pointing
to the preceding reference's `.pt` snapshot, and a fresh `--output` JSONL path.
[Evidence inventory](evidence/rwkv6-blinkdl-evidence-manifest.json).
The remaining state discrepancy, long generation, native GPU baselines and
high-concurrency campaign remain unfinished; required checkpoint files are
retained.
