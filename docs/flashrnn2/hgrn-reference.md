# HGRN 1.3B complete-checkpoint CPU reference

The pinned 24-layer HGRN checkpoint passed cached generation against
full-prefix recomputation at BS=1/2/4 for three prompt groups, five prompt
tokens and four generated tokens. All 36 full-vocabulary logit and all-layer
recurrent-state checks pass; all 84 greedy token choices match. The model
process naturally exited 0. A separate audit recomputed 36 saved logit pairs,
84 token choices and 72 final layer-state pairs and exited 0.

| Batch | Logits pass | All-layer states pass | Matching tokens | Max logit error | Max state error |
|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 12/12 | 12/12 | 0.000041962 | 0.000213623 |
| 2 | 12/12 | 12/12 | 24/24 | 0 | 0 |
| 4 | 12/12 | 12/12 | 48/48 | 0 | 0 |

The frozen budgets are `atol=rtol=1e-3` for logits and `atol=rtol=1e-5`
for recurrent states. The largest BS=1 state difference remains within its
elementwise absolute-plus-relative budget; BS=2/4 are bitwise equal. Each step
compares every layer. The independent [audit](evidence/hgrn-reference/hgrn-reference-r1-audit.json)
checks source, checkpoint, tokenizer and snapshot hashes, row coverage, shapes
and the reported comparisons. Nine binary snapshots preserve both logit sides
for every step and both complete final caches for prompt group 0 at each batch;
their hashes are in the [evidence inventory](evidence/hgrn-reference/hgrn-reference-evidence-manifest.json).

## Checkpoint and execution scope

[`fla-hub/hgrn-1.3B-100B`](https://huggingface.co/fla-hub/hgrn-1.3B-100B/tree/1adad50103ad6b9c5f79df6b3ce6c9fa2299a572)
is pinned to revision `1adad50103ad6b9c5f79df6b3ce6c9fa2299a572`.
Eight public files were hash-verified. The 2,728,818,968-byte BF16 weight file
has SHA256 `480ed12b06d2ebfc604210c90ecb66e922d5825e7260ad49cd1061ef7d52a987`.
The metadata preflight matched all 244 tensor names and shapes. The complete
model then strictly loaded 1,364,396,032 parameters after FP32 promotion.

The reference executes pinned official FLA HGRN attention and block class
bodies plus its naive recurrent HGRN function at
[`17dd566`](https://github.com/fla-org/flash-linear-attention/commit/17dd5662554d46b6bcb1d1ff728cebb461c9aef9),
with the existing pinned Torch RMS/MLP operations. Its layer-bound calculation
is retained. A two-layer full-versus-split control passed with a nonzero
lower-bound effect before the complete-checkpoint run. The configured model
has no short convolution. The pinned `tokenizer.json` Git blob was verified
and decoded directly; the Hugging Face tokenizer wrapper was not run.

Execution used Python 3.12, Torch 2.14.1 CPU, FP32 weights and one
Torch/OMP/MKL thread. The reference does not execute native FLA CUDA or a
FlashRNN kernel. The model weight was evicted after the completed audit, hash
verification and no-reader check. Its pinned public version must be
redownloaded and reverified before GPU use.

```sh
python tools/flashrnn2/hgrn_reference_gate.py \
  --model "$HGRN_CHECKPOINT" --source "$PINNED_HGRN_FLA_SOURCES" \
  --common "$PINNED_GLA_COMMON_SOURCES" \
  --output "$RESULTS/hgrn-reference.jsonl"
```

| Required E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| Native HGRN/FLA GPU generation | — | — | Not measured |
| High-concurrency, multi-batch GPU | — | — | Not measured |

Native accelerated generation, long contexts and the requested high-concurrency
GPU E2E matrix remain unqualified.
