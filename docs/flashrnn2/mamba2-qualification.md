# Full Mamba2-130M qualification

The official `state-spaces/mamba2-130m` checkpoint is pinned at
`3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49`. Its 258,059,314-byte weight file
passes SHA256 `f786aff903d8b6f2cef67ff2b1db06a9f44e7780340e6ea6d3dead16b52fc501`.
The immutable file manifest was obtained from hf-mirror. Canonical-HF
metadata confirmation is still pending.

`mamba2_checkpoint.py` maps `backbone.embedding.weight` to
`backbone.embeddings.weight` and uses the default architecture parameters
from pinned state-spaces Mamba2 source: expand=2, state=128, head dimension=64,
groups=1 and chunk=256. All 219 keys load strictly, with no missing/unexpected
keys or shape mismatches. The resulting Transformers Mamba2ForCausalLM
contains all 24 layers and 128,989,632 parameters. FP16/FP32 checkpoint values
are copied into an FP32 model. This is an explicit checkpoint adapter into
the existing Transformers implementation; the native mamba_ssm CUDA path
has not been qualified.

The complete CPU model runs prefill and four greedy decode steps, comparing
cached execution against a full-prefix forward at each step. It uses the
same tokenizer files as the pinned Mamba130M checkpoint; their hashes are
recorded. The gate retains `atol=0.0001`, `rtol=0.00001`.

| Batch | Greedy tokens | Strict logits gate | Maximum absolute logit difference |
| ---: | --- | --- | ---: |
| 1 | All match | FAIL, steps 1 and 3 | 0.0002679825 |
| 2 | All match | FAIL, steps 1 and 3 | 0.0001959801 |
| 4 | All match | FAIL, steps 1 and 3 | 0.0001959801 |

All three cases complete; the controller naturally exits 1. The failures
are retained. They are baseline cached/full-prefix differences, not results
from a candidate acceleration. CPU timing is not a performance comparison.

| Full-model CUDA multi-BS/high-concurrency comparison | Baseline | Candidate | Speed ratio |
| --- | --- | --- | --- |
| Mamba2-130M | Not measured | Not measured | N/A |

Evidence: [raw cases](evidence/mamba2-model-gate-r1.jsonl),
[metadata and source hashes](evidence/mamba2-model-gate-r1.meta.json),
[controller](evidence/mamba2-model-gate-r1-controller.json), and
[state-dict shape audit](evidence/mamba2-checkpoint-shapes.json).
Weights are retained until the GPU and concurrency campaign completes.

A separate [complete TorchScript export](mamba2-script-qualification.md) now
passes three independent native-versus-traced B1/P5 CPU cases, with bitwise
prefill/decode logits and caches. This does not revise the failures above;
its CUDA portability and performance remain untested.

```sh
python tools/flashrnn2/mamba_model_gate.py --family mamba2 --model MAMBA2_DIR --tokenizer MAMBA130M_DIR --output mamba2.jsonl
```
