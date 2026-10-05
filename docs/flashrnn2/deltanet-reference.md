# Complete DeltaNet reference qualification

The full DeltaNet 1.3B checkpoint completes cached-versus-full-prefix CPU
generation at B1/B2/B4. All 84 greedy token choices agree and all 36 logit
comparisons pass, but nine recurrent-state checks and six convolution-cache
checks exceed the unchanged error budgets. The model gate therefore exits 1.

| Batch | Logit checks | Recurrent-state checks | Convolution-cache checks | Greedy tokens equal |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12/12 | 9/12 | 8/12 | 12/12 |
| 2 | 12/12 | 9/12 | 11/12 | 24/24 |
| 4 | 12/12 | 9/12 | 11/12 | 48/48 |

Each batch has three distinct prompt groups, five input tokens and four generated
tokens per request. At every step, the runner checks the full 32,000-word logit
vector, all 24 recurrent matrices and all 72 short-convolution caches. The
element criterion is `abs(actual - expected) <= atol + rtol * abs(expected)`,
with `atol=rtol=1e-3` for logits and `1e-5` for both kinds of cache. Nonfinite
values fail on either side. The reference re-evaluates the complete prefix
using the cached arm's generated tokens; token equality is checked at every step.

| Batch | Max logit error | Max recurrent error | Max convolution-cache error | Worst recurrent / convolution normalized error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.34058e-5 | 5.34058e-5 | 9.15527e-5 | 2.177735 / 1.798184 |
| 2 | 5.91278e-5 | 4.95911e-5 | 2.28882e-5 | 2.289958 / 1.022590 |
| 4 | 6.29425e-5 | 5.72205e-5 | 2.47955e-5 | 2.493133 / 1.022590 |

All recurrent failures occur at the fourth generation step, across all nine
batch/prompt cases. The earliest failing generation step is B1/case2/step1
(zero-based). There are 12 failed batch steps in the union of the two categories.
The layer and operator causing the full-model differences remain unlocalized;
token equality does not waive the state failures.

## Checkpoint and implementation

The immutable checkpoint is
`fla-hub/delta_net-1.3B-100B@b4dcbbafd4fde802717bdec3008d4aba9cb3a1f8`.
All eight selected files pass their upstream LFS SHA256 or Git blob checks.
The complete model strictly loads all 339 tensors and executes all 24 layers,
with 1,365,677,056 parameters. Original BF16 checkpoint values are exactly
promoted to FP32; embeddings and the output head remain untied.

The reference extracts hash-checked DeltaNet, block, short-convolution and naive
delta-rule bodies from FLA v0.3.0 at
`17dd5662554d46b6bcb1d1ff728cebb461c9aef9`.
It supplies Torch RMSNorm, SwiGLU, short-convolution and L2 normalization
primitives. The delta recurrence retains the full matrix state and rank-one
update. No native accelerated FLA CUDA kernel is executed or qualified here.
`FLA_CONV_BACKEND=cuda` selects the author's wrapper branch, whose helpers are
explicitly replaced by the documented Torch functions in this reference only.

Separate short-convolution controls check B1/B2, prefix lengths 1/3/5, bias
on/off, identity/SiLU and three continuation steps: all 72 FP64 comparisons pass,
maximum error 4.44089e-16 and cache values exact. Those primitive controls do
not establish bitwise FP32 full-model continuation.

## Evidence and remaining work

The [independent audit](evidence/deltanet-reference-r1-audit.json) verifies all
36 unique rows, source hashes, inputs, checkpoint dimensions and the saved
cached-logit tensor hash. It recomputes all 84 cached token argmaxes. Full-prefix
logits and cache tensors were not persisted; their comparison results remain
runner records, not independently recomputed tensor comparisons. The audit's
success confirms the retained evidence, while the model result remains failed.

| Required comparison | Baseline tok/s | Candidate tok/s | Speedup | Status |
| --- | ---: | ---: | ---: | --- |
| Full DeltaNet CUDA generation | — | — | N/A | Not run |
| Native FLA accelerated baseline | — | — | N/A | Not qualified |
| Multiple batches and high concurrency | — | — | N/A | Not run |

Reproduce with an existing runtime and the pinned assets:

```bash
FLA_CONV_BACKEND=cuda python tools/flashrnn2/gla_reference_gate.py \
  --family deltanet --model "$DELTANET" --source "$PINNED_FLA" \
  --output "$RESULTS/deltanet-reference.jsonl"
```

The complete numerical run takes its natural failure exit after recording all
cases. Weights remain needed for diagnosis and the unfinished GPU campaign.
The [inventory](evidence/deltanet-reference-evidence-manifest.json) links raw
rows, metadata, download receipts, commands/exits and primitive controls.
