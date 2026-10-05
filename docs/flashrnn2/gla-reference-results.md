# Complete GLA checkpoint reference

The GLA campaign now executes the complete public 1.3B checkpoint on CPU.
Cached generation and full-prefix recomputation agree exactly at B2 and B4.
At B1, all greedy tokens and logits comparisons pass, but nine incremental
steps exceed the frozen state budget. Those failures remain recorded.

| Batch | Compared batch steps | Greedy token choices | Failed logits steps | Failed state steps | Max logits absolute error | Max state absolute error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 12, all equal | 0 | 9 | 0.0000371933 | 0.1015625 |
| 2 | 12 | 24, all equal | 0 | 0 | 0 | 0 |
| 4 | 12 | 48, all equal | 0 | 0 | 0 | 0 |

Each batch runs three fixed text groups, P5 prompts and four generated tokens.
At every step the runner compares all 32,000 logits and recurrent matrices in
all 24 layers. Full-prefix recomputation consumes the cached path's generated
tokens, keeping both inputs aligned. The model has **1,365,514,240 parameters**;
all **339 checkpoint tensors** load strictly, with untied embedding/head weights.
No layers or matrix dimensions are reduced.

The checkpoint is
[fla-hub/gla-1.3B-100B](https://huggingface.co/fla-hub/gla-1.3B-100B/tree/46b15820a4df269e99aed9d709e017677c15d24b),
revision `46b15820a4df269e99aed9d709e017677c15d24b`. Its 2,731,066,224-byte BF16
weight file has SHA256
`97c02567af31fc5ef98280a5d84e1c6b98842d8b92972940626bce8ebb3ed73d`.
Eight selected runtime/documentation files total 2,733,359,293 bytes and pass
their pinned LFS SHA256 or Git blob checks. The downloader and controller
naturally exit 0; no shared environment is modified.

The reference uses
[FLA v0.3.0 source](https://github.com/fla-org/flash-linear-attention/tree/17dd5662554d46b6bcb1d1ff728cebb461c9aef9),
revision `17dd5662554d46b6bcb1d1ff728cebb461c9aef9`. Hash-checked AST extraction
preserves the attention, block and MLP bodies, `naive_recurrent_gla`, and
`rms_norm_ref`. Module construction and cache storage are supplied locally.
Fused RMS normalization, output gating and SwiGLU use explicit Torch formulas;
every attention dispatch uses the upstream naive recurrence. This executes
the full model but **does not qualify the accelerated native FLA backend** or
claim its precision/performance behavior. Original BF16 weights are promoted
exactly to FP32, and the computation runs on the existing CPU Torch 2.13 runtime.

The predeclared probe budgets are logits `atol=rtol=1e-3` and recurrent states
`atol=rtol=1e-5`. These are screening budgets, not calibrated trained-model
accuracy guarantees. B1's worst normalized state error is 63.823467; all B2/B4
reported errors are zero. A batch-dependent arithmetic difference is consistent
with this pattern, but its full-model cause has not been isolated. No tolerance
is widened and no state failure is replaced by token agreement.

| Required GPU E2E comparison | Native FLA tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Full-model generation across batch sizes | Not measured | Not measured | N/A |
| High-concurrency serving, longer contexts and quality workloads | Not measured | Not measured | N/A |

CPU execution is correctness evidence only. Controller 37214 and child 37220
naturally exit 1 after completing all nine cases and 36 batch steps. The saved
snapshot contains full-vocabulary logits and token IDs, not the full state
matrices. Its independent audit checks the hash, shapes, finite logits, every
argmax/token pair, source identity and raw result summaries. It does not
independently recompute state errors from unavailable state snapshots.

After this run, the comparison helper was corrected to reject non-finite
reference values as well as non-finite candidates. Four invalid cases and one
finite identity case pass regression tests. The original executed gate source
is retained with its recorded hash. All observed error summaries are finite,
so this change preserves the recorded finite comparisons; the complete model
was not rerun under the revised gate.

To reproduce, obtain the files at the source revision and flat filenames in
[the source manifest](evidence/gla-source-manifest.json), plus the eight
checkpoint files in [the download manifest](evidence/gla-verified-download-manifest.json).
Then run:

```bash
python tools/flashrnn2/gla_reference_gate.py \
  --model "$MODEL/gla-1.3b" --source "$PINNED_FLA_SOURCES" \
  --output "$RESULTS/gla-reference.jsonl"
```

Evidence: [commands and actual exit](evidence/gla-reference-r1-controller.json),
[raw comparisons](evidence/gla-reference-r1.jsonl),
[metadata](evidence/gla-reference-r1.meta.json),
[independent audit](evidence/gla-reference-r1-audit.json),
[executed gate](evidence/gla-reference-gate.executed-r1.py),
[finite-value tests](evidence/gla-finite-comparison-tests-r1.json), and
[file manifest](evidence/gla-evidence-manifest.json).
The checkpoint remains needed for the unfinished GPU and concurrency campaign.
