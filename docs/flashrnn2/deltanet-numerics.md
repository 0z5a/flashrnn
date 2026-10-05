# DeltaNet projection-shape diagnosis

The full-model cache failures in [the original reference](deltanet-reference.md)
first diverge at a Linear projection for the selected B1/case2 input. Fixing
the projection row shape brings that case within the original budgets;
changing convolution alone does not. These controls change arithmetic order
and do not qualify the original reference or native accelerated FLA.

## Localization and controls

The unchanged full 1,365,677,056-parameter model was observed at 338 module
boundaries per step for B1/case2 P5/G4. All four original error records reproduce
exactly. At step1, `model.layers.0.attn.q_proj` receives bitwise-identical
last-token inputs but produces a maximum difference of 9.15527e-5. The first
failing convolution cache is in layer2; the first failing recurrent state is
in layer3 at step3. Indices are zero-based.

The Linear control evaluates each projection with one input row, including the
SwiGLU down-projection and vocabulary head. The convolution control uses the
decode multiply/reduce path for every prefill token. Checkpoint values, prompts,
FP32 precision, recurrent update and acceptance budgets remain unchanged.

| B1/case2 P5/G4 arm | Passing steps | Worst recurrent normalized error | Worst convolution-cache normalized error | Tokens vs original |
| --- | ---: | ---: | ---: | ---: |
| Original reference | 1/4 | 2.177735 | 1.798184 | 4/4 |
| Fixed Linear row shape | 4/4 | 0.451464 | 0.211486 | 4/4 |
| Sequential convolution only | 1/4 | 0.856671 | 1.564835 | 4/4 |
| Both controls | 4/4 | 0.136501 | 0.206196 | 4/4 |

An error above1 fails. The [independent final-state audit](evidence/deltanet-arithmetic-controls-r1-audit.json)
recomputes 291 complete tensor-pair comparisons from saved logits, recurrent
matrices and convolution caches, checks finiteness and all token IDs, and
verifies the 166,096,659-byte snapshot hash. Intermediate states remain runner
checks. All diagnostic processes naturally exit0; that exit denotes collection,
not success of every controlled arm.

## Full matrix

The fixed-row control completes all nine cases, 36 batch steps and 84 token choices, with result **PASS** and natural exit0. Token differences against the original cached run: **0/84**. Only Linear row shape changes; the original convolution implementation is retained.

| Batch | Logits | Recurrent states | Convolution caches | Worst recurrent / convolution normalized error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12/12 | 12/12 | 12/12 | 0.589746 / 0.225700 |
| 2 | 12/12 | 12/12 | 12/12 | 0.589746 / 0.225700 |
| 4 | 12/12 | 12/12 | 12/12 | 0.603606 / 0.272587 |

The [full-matrix audit](evidence/deltanet-row-linear-reference-r1-audit.json) checks all raw rows, hashes, inputs and saved cached-logit argmaxes. Full-prefix logits and full cache tensors are not saved for this matrix, so those comparisons remain runner evidence. The separate factorial final-state audit above independently recomputes the complete saved tensors for one selected case.

Reproduce the fixed-row diagnostic using the same assets as the original gate:

```bash
FLA_CONV_BACKEND=cuda python tools/flashrnn2/deltanet_row_linear_reference.py \
  --family deltanet --model "$DELTANET" --source "$PINNED_FLA" \
  --output "$RESULTS/deltanet-row-linear.jsonl"
```

The output's `.arithmetic.json` sidecar explicitly records the changed Linear
order and wrapper source hash. The ordinary reference gate remains unchanged.
Layer-probe and factorial-control source snapshots in the evidence inventory
are the exact scripts executed from the task's asset root; they require its
`source/`, `models/` and `evidence/` layout.

| Required performance comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native accelerated DeltaNet CUDA E2E | — | — | Not measured |
| Multi-batch/high-concurrency service | — | — | Not measured |

The original state failures remain retained. Native FLA CUDA, long generation,
training and high-concurrency serving are still required. Fixed-row arithmetic
is a numerical control and carries no speed claim.

[Evidence inventory](evidence/deltanet-numerics-evidence-manifest.json).
