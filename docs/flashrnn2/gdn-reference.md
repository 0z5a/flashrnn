# Gated DeltaNet 340M complete-checkpoint CPU reference

The public 24-layer checkpoint passed cached generation against full-prefix
recomputation at BS=1/2/4 for three prompt groups, five prompt tokens and four
generated tokens. All 36 full-vocabulary logit, 36 all-layer recurrent-state and
36 all-layer convolution-state checks pass; all 84 greedy token choices match.
The model controller naturally exited 0. A separate audit recomputed all saved
logits and 72 final recurrent plus 216 final convolution state pairs and exited 0.

| Batch | Logits pass | Recurrent states pass | Convolution states pass | Matching tokens | Max logit error | Max recurrent error | Max convolution error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 12/12 | 12/12 | 12/12 | 0.000026703 | 0.000006676 | 0.000049591 |
| 2 | 12/12 | 12/12 | 12/12 | 24/24 | 0.000023842 | 0.000006676 | 0.000009537 |
| 4 | 12/12 | 12/12 | 12/12 | 48/48 | 0.000028610 | 0.000005722 | 0.000010490 |

The frozen budgets are `atol=rtol=1e-3` for logits and `atol=rtol=1e-5` for
both state families. The largest absolute convolution error at BS=1 remains
within its elementwise absolute-plus-relative budget. Every step compares all
24 layers, not only the final cache. Binary snapshots preserve both logit
sides for every step and both complete final caches for prompt group 0 at each
batch size; their hashes are in the [evidence inventory](evidence/gdn-reference/gdn-reference-evidence-manifest.json).

## Checkpoint and historical source match

[`linear-moe-hub/Gated-Deltanet-340M`](https://huggingface.co/linear-moe-hub/Gated-Deltanet-340M/tree/c83bdada453cde56932f37be71338df22ca29b7d)
is pinned to revision `c83bdada453cde56932f37be71338df22ca29b7d`.
The 799,109,000-byte BF16 weight file has SHA256
`f599714aff09f07efce88afc4c327806c8d7082ccc4469ff4d1d1a2e707bf4f9`.
All eight public files were verified. The file stores 435 tensors and
399,531,296 elements; the tied embedding and LM head leave 366,763,296
effective parameters. All 435 tensor names and shapes match the pinned model,
and all tensors load strictly after FP32 promotion.

The checkpoint does not match the July 2025 FLA model class: it has an
additional per-layer `attn.D` and one fused MLP `gate_proj` in place of
separate `gate_proj` and `up_proj`. The strict r2 load failed at this boundary.
The model instead matches all 435 tensor names and shapes in official
[FLA commit `bcd9e79`](https://github.com/fla-org/flash-linear-attention/commit/bcd9e79bfea85a394a023663f164587b506e0422)
from December 2024. The CPU reference executes its unchanged attention, block,
MLP and short-convolution class bodies with explicit Torch implementations of
the fused recurrence, RMS/gating and SwiGLU operations. The historical `D`
parameter is preserved for strict loading; that pinned attention forward body
does not read it. This source match is structural evidence, not proof of the
exact training commit.

An earlier synthetic control of the newer FLA source found that a two-token
continuation bypasses its short-convolution cache step and fails split/full
equivalence. The historical source also enters its cache step only for one
token. The qualified path uses one token per decode step; its historical-source
single-layer control passes and the complete-checkpoint run confirms that path
across all 24 layers.

## Scope and reproduction

The [independent audit](evidence/gdn-reference/gdn-reference-r3-audit.json)
checks source, checkpoint, tokenizer and snapshot hashes, complete row coverage,
output shapes and the reported tolerance comparisons. The native Hugging Face
tokenizer wrapper was unavailable in the existing local runtime; the pinned
`tokenizer.json` Git blob was verified and decoded directly. Execution used
Python 3.12, Torch 2.14.1 CPU, FP32 weights and one Torch/OMP/MKL thread.
This reference does not execute native FLA CUDA, causal-conv1d CUDA, or any
FlashRNN kernel.

```sh
python tools/flashrnn2/gdn_reference_gate.py \
  --model "$GDN_CHECKPOINT" --source "$PINNED_2024_FLA_SOURCES" \
  --delta-source "$PINNED_COMMON_TORCH_SOURCES" \
  --output "$RESULTS/gdn-reference.jsonl"
```

| Required comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| CPU cached/full-prefix correctness | — | — | Not measured |
| Native FLA GPU generation | — | — | Not measured |
| High-concurrency, multi-batch GPU E2E | — | — | Not measured |

Native/accelerated Gated DeltaNet performance, long contexts and the requested
high-concurrency GPU E2E matrix remain unqualified.
The local weight was evicted after the completed audit and a no-reader check;
its pinned public version must be downloaded and reverified before GPU use.
