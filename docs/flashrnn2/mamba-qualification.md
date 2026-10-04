# Full Mamba-130M qualification

The complete `state-spaces/mamba-130m-hf` checkpoint at revision
`1e76775f628fbf1350fbe4dbb3d971ba64af25a1` loads into the existing
Transformers 4.54.1 `MambaForCausalLM`, with all 24 layers and 129,135,360
parameters. Missing, unexpected, mismatched and error keys are empty.
The harness rechecks all five file hashes before loading. The immutable
manifest came from hf-mirror; independent canonical-HF metadata confirmation
remains pending.

This uses the unchanged built-in sequential CPU implementation, FP32,
PyTorch 2.13.0, and one CPU thread. No CUDA extension or environment package
was installed. It exercises tokenization, the full checkpoint, all layers,
the recurrent cache, the language-model head and greedy decoding.

| Check | B | Actual result | Maximum logit difference |
| --- | ---: | --- | ---: |
| Upstream 10-token generation and first 40 logits | 1 | PASS under upstream fixture tolerances | 0.0359802 |
| Cached vs full-prefix, four new tokens | 1 | Token parity PASS; strict logits FAIL | 0.0002537 |
| Cached vs full-prefix, four new tokens | 2 | PASS | 0.0002441 |
| Cached vs full-prefix, four new tokens | 4 | Token parity PASS; strict logits FAIL | 0.0002441 |

The first row reproduces the fixed test in
[Transformers v4.54.0](https://github.com/huggingface/transformers/blob/v4.54.0/tests/models/mamba/test_modeling_mamba.py),
including its `atol=rtol=0.001` and exact generated text. The installed runtime
is 4.54.1; the source-file hash is recorded rather than assuming identical
versions. The other rows retain the separately frozen stricter cache gate,
`atol=0.0001`, `rtol=0.00001`. B1 has one mismatched logit at decode step 1;
B4 has 47 at step 2. All greedy token choices agree. The upstream fixture
does not change or erase these failures.

The first strict run stopped at its first mismatch; the subsequent diagnostic
run evaluates all three batch sizes. A launcher mistake in `official-r1`
omitted the fixture flag and repeated the cache gate. That evidence remains
preserved with an explicit correction; `official-r2` is the actual fixture
run and naturally exits 0. The final harness's strict `r3` naturally exits 1.

| Speed comparison | Baseline | Candidate | Speed ratio |
| --- | --- | --- | --- |
| Full model, high concurrency, multiple batch sizes on CUDA | Not measured | Not measured | N/A |

These are baseline correctness results. No candidate optimization or model
speedup has been measured. The checkpoint is retained until its full GPU and
concurrency campaign finishes.

Evidence: [strict r3](evidence/mamba-model-gate-r3.jsonl),
[r3 metadata](evidence/mamba-model-gate-r3.meta.json),
[official fixture](evidence/mamba-model-gate-official-r2.meta.json), and
[official controller](evidence/mamba-model-gate-official-r2-controller.json).

```sh
python tools/flashrnn2/mamba_model_gate.py --model MODEL_DIR --output strict.jsonl
python tools/flashrnn2/mamba_model_gate.py --model MODEL_DIR --output official.jsonl --official-fixture-only
```
