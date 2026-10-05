# GLA fixed-row projection diagnosis

Fixing the Linear input row shape removes all cached-versus-full-prefix
differences in the complete GLA 1.3B CPU matrix. All 36 batch steps have bitwise
identical logits and all 24 recurrent states; all 84 greedy token choices also
match the original cached run. This intentionally changes projection arithmetic
order. The [original failed gate](gla-reference-results.md) remains retained.

| Batch | Original state comparisons | Fixed-row state comparisons | Fixed-row max logit / state error | Tokens vs original |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3/12 pass | 12/12 pass | 0 / 0 | 12/12 |
| 2 | 12/12 pass | 12/12 pass | 0 / 0 | 24/24 |
| 4 | 12/12 pass | 12/12 pass | 0 / 0 | 48/48 |

Each batch uses three prompt groups, five input tokens and four generated
tokens. The model executes every layer and all 1,365,514,240 parameters from
the hash-verified `fla-hub/gla-1.3B-100B` checkpoint at
`46b15820a4df269e99aed9d709e017677c15d24b`. Its 339 BF16 tensors are exactly
promoted to FP32. No model, norm, recurrent-state update or tolerance changes.
The control evaluates each Linear input row separately, including the SwiGLU
projection and vocabulary head, reusing the DeltaNet diagnostic function.

The original B1 state error reached 0.1015625 and 63.823467 times its element
budget. In this control it is zero across all nine cases. This demonstrates
the effect of Linear row shape on this matrix; it does not qualify native FLA
fused kernels or establish a faster execution path.

The [independent audit](evidence/gla-row-linear-reference-r1-audit.json) verifies
source hashes, the checkpoint's configuration hash, all unique case/step rows,
saved cached-logit tensor hashes and all 84 token argmaxes. Full-prefix logits
and recurrent tensors were not saved for this matrix; their comparisons are
runner evidence. The controller and child naturally exit0. An initial auditor
schema mismatch is retained separately; correcting the auditor did not rerun
or change the model experiment.

| Required comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native accelerated GLA GPU E2E | — | — | Not measured |
| Multiple batches and high concurrency | — | — | Not measured |

Reproduce on an existing runtime with the same pinned model/source assets:

```bash
python tools/flashrnn2/gla_row_linear_reference.py \
  --family gla --model "$GLA" --source "$PINNED_FLA" \
  --output "$RESULTS/gla-row-linear.jsonl"
```

The `.arithmetic.json` sidecar records the changed operation and both diagnostic
source hashes. [Evidence inventory](evidence/gla-numerics-evidence-manifest.json).
GPU generation, accelerated baselines, long sequences and training remain
unfinished, so checkpoint files are retained for those consumers.
