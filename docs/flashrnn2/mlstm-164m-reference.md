# xLSTM mLSTM 164M complete-checkpoint CPU reference

The pinned public 12-block checkpoint ran cached generation against full-prefix
recomputation at BS=1/2/4 for three prompt groups, five prompt tokens and four
generated tokens. Both runs match all 36 full-vocabulary logit comparisons and
all 84 greedy token choices. An independent audit recomputed 36 saved logit
pairs and 108 complete final layer-state pairs for each run. The original
Torch path retains five cell-state failures under its frozen `atol=rtol=1e-4`
budget. A control that fixes the row count of every Linear operation to 32
passes all logits and three state families bitwise; it keeps weights, input
tokens, recurrence and the numerical budget unchanged.

| Batch | Original logits | Original cell | Original normalizer | Original stabilizer | Fixed-row logits/states | Matching tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 10/12 | 12/12 | 12/12 | 12/12 each | 12/12 |
| 2 | 12/12 | 9/12 | 12/12 | 12/12 | 12/12 each | 24/24 |
| 4 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 each | 48/48 |

The largest original cell-state difference is `0.000869751` at BS=2;
the worst absolute-plus-relative budget ratio is `1.400211`.
The largest original logit difference is `0.000054359` at BS=1, inside the
`atol=rtol=1e-3` logit budget. Cross-run comparison finds all 84 generated
token IDs identical, with maximum original-versus-fixed cached logit difference
`0.000054359`. The fixed-row result is consistent with shape-dependent Linear
arithmetic causing the failed state checks in this CPU path; it is a changed-arithmetic
diagnostic, not a pass of the original path or evidence of native-kernel speed.

The [original audit](evidence/mlstm-164m-reference/xlstm-mlstm-164m-reference-r1-audit.json)
and [fixed-row audit](evidence/mlstm-164m-reference/xlstm-mlstm-164m-reference-r2-audit.json)
check source, checkpoint, tokenizer and snapshot hashes; exact row coverage;
all saved logits and tokens; and complete final states of the 12 active blocks.
The installed Transformers cache allocates 32 slots by default; both audits
verify that the 20 unused slots remain zero. All 18 binary snapshots stay in
the execution workspace, with sizes and hashes in the
[evidence inventory](evidence/mlstm-164m-reference/mlstm-164m-reference-evidence-manifest.json).

## Source, checkpoint and execution scope

The 164M checkpoint is a selected directory of
[`NX-AI/xlstm_scaling_laws`](https://huggingface.co/NX-AI/xlstm_scaling_laws/tree/2ce74c14add515517ae6a32d3ae80cc766f62c03)
at revision `2ce74c14add515517ae6a32d3ae80cc766f62c03`. Its
328,241,392-byte BF16 weight has SHA256
`1dc957a158235bd26f0e1554345e3f695232ea3107dd28c4f81f71d53c1f0afc`.
All 183 stored tensors load strictly after FP32 promotion into 164,110,224
parameters. The checkpoint's `q/k/v` key spelling is mapped one-to-one to
Transformers 4.54.1 `query/key/value` for 36 weight names; tensor contents and
shapes are unchanged. The built-in Torch xLSTM backend handles recurrence;
no `mlstm_kernels` package was installed or invoked.

The [official scaling-laws loader](https://github.com/NX-AI/xlstm_scaling_laws/blob/604bb3062d3a7681bb33e467108cb0a6f81267fc/xlstm_scaling_laws/checkpoint_loading.py)
specifies the [public xLSTM-7B tokenizer](https://huggingface.co/NX-AI/xLSTM-7b)
and a leading BOS token. Three tokenizer files are pinned to revision
`9dc507bd0939cf372a4a4f667335651d8e49dddb` and hash-verified locally;
only their hashes and selected IDs are in this PR. The source file hashes and
official xLSTM/scaling-law commits are in the
[source pins](evidence/mlstm-164m-reference/official-source-pins.json).
Execution used Python 3.12, Torch 2.14.1 CPU, Transformers 4.54.1,
FP32-promoted parameters and one Torch/OMP/MKL thread. The completed weight
was evicted after both audits, checksum verification and a no-reader check;
GPU use requires pinned re-download.

```sh
python tools/flashrnn2/mlstm_scaling_reference_gate.py \
  --model "$PINNED_MLSTM_164M_CHECKPOINT_AND_TOKENIZER" \
  --output "$RESULTS/mlstm-164m-original.jsonl"

python tools/flashrnn2/mlstm_scaling_reference_gate.py \
  --model "$PINNED_MLSTM_164M_CHECKPOINT_AND_TOKENIZER" \
  --fixed-linear-rows 32 \
  --output "$RESULTS/mlstm-164m-fixed-rows.jsonl"
```

| Required E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| Native `mlstm_kernels` generation | — | — | Not measured |
| High-concurrency, multi-batch GPU | — | — | Not measured |

Native CUDA, longer contexts, full-model training and high-concurrency GPU
E2E remain unqualified.
