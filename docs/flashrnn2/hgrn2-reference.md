# HGRN2 1.3B complete-checkpoint CPU reference

The pinned 24-layer HGRN2 checkpoint ran cached generation against full-prefix
recomputation for three prompt groups at BS=1/2/4, five prompt tokens and four
generated tokens. All 36 full-vocabulary logit comparisons pass and all 84
greedy token choices match. BS=1 has seven recurrent-state failures under the
unchanged state budget; BS=2/4 are bitwise equal. The model process naturally
exited 1 for this numerical result, after writing all rows and snapshots.

| Batch | Logits pass | All-layer states pass | Matching tokens | Max logit error | Max state error |
|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 5/12 | 12/12 | 0.000041008 | 0.00048828125 |
| 2 | 12/12 | 12/12 | 24/24 | 0 | 0 |
| 4 | 12/12 | 12/12 | 48/48 | 0 | 0 |

The budgets are `atol=rtol=1e-3` for logits and `atol=rtol=1e-5` for recurrent
states; they were not relaxed. The worst BS=1 state element is 3.087 times its
allowed error. Failing BS=1 steps are (group, step) `(0,1)`, `(0,2)`, `(0,3)`,
`(1,2)`, `(2,1)`, `(2,2)`, `(2,3)`. The batch-dependent behavior does not by
itself establish a cause.

## Checkpoint and execution scope

[`fla-hub/hgrn2-1.3B-100B`](https://huggingface.co/fla-hub/hgrn2-1.3B-100B/tree/2f413dd9b63591b9b177bbf940942ea7eb70abfe)
is pinned to revision `2f413dd9b63591b9b177bbf940942ea7eb70abfe`. Eight
public files were hash-verified. The 2,728,818,968-byte BF16 weight file has
SHA256 `92ea939b56018e51e696d1cb38970ed2797fad97bfe1eacff144bdc0397cdc7c`.
All 244 tensor names and shapes match the model; all values are promoted to
FP32 and strictly loaded into 1,364,396,032 parameters. The vocabulary has
32,000 tokens, hidden width 2,048, and 24 recurrent layers. The independent
metadata-only preflight and actual full model load are separate checks.

The reference executes unchanged pinned FLA HGRN2 attention/block class bodies
at [`17dd566`](https://github.com/fla-org/flash-linear-attention/commit/17dd5662554d46b6bcb1d1ff728cebb461c9aef9),
with existing pinned Torch RMS/MLP and explicit Torch recurrent GLA primitives.
The original HGRN2 layer-bound weighting is retained. A small two-layer
full-versus-split primitive control passed before the checkpoint run, including
a nonzero lower-bound effect. This is a CPU reference, not native FLA CUDA or
an accelerated model baseline.

## Independent audit

The [audit](evidence/hgrn2-reference/hgrn2-reference-r1-audit.json) recomputes
all 36 saved logit pairs, 84 token choices and 72 complete final layer-state
pairs from both cached and full-prefix snapshots: 24 layers at group 0 for
each batch. Two of those 72 final pairs fail the frozen state budget, both at
BS=1. Other intermediate state comparisons are recorded by the executed gate
and are not claimed as independently replayed tensors. The audit checks source,
checkpoint, tokenizer and snapshot hashes, row coverage and output shapes;
it exits 0 while preserving the model's numerical failure. The nine binary
snapshots are retained in the execution workspace with their hashes in the
[evidence inventory](evidence/hgrn2-reference/hgrn2-reference-evidence-manifest.json).

```sh
python tools/flashrnn2/hgrn2_reference_gate.py \
  --model "$HGRN2_CHECKPOINT" --source "$PINNED_HGRN2_SOURCES" \
  --common "$PINNED_GLA_SOURCES" --output "$RESULTS/hgrn2-reference.jsonl"
```

The execution used Python 3.12.14, Torch 2.13.0 CPU, Transformers 4.54.1 and
one Torch/OMP/MKL thread. Shared paging and concurrent checkpoint transfer
make elapsed wall time unsuitable as a speed result.

| Required comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| CPU cached/full-prefix correctness | — | — | Not measured |
| Native FLA GPU generation | — | — | Not measured |
| High-concurrency, multi-batch GPU E2E | — | — | Not measured |

Native FLA, long-context and high-concurrency GPU performance remain to be
qualified; this result preserves the seven original BS=1 state failures.
