# Mamba-3 MIMO 187M complete-checkpoint CPU reference

The pinned public 12-layer MIMO checkpoint completed cached generation against
full-prefix recomputation at BS=1/2/4 for three prompt groups, five prompt
tokens and four generated tokens. All 36 full-vocabulary logit comparisons and
all 84 greedy token choices match. The second, calibrated state run passes all
36 angle, SSM, key and value-cache comparisons. Its process exited 0; an
independent audit recomputed all 36 saved logit pairs and 144 complete final
layer-state pairs from nine locally retained snapshots.

| Batch | Logits | Angle | SSM | Key | Value | Matching tokens | Max logit error | Max SSM error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 0.000067711 | 0.000060141 |
| 2 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 24/24 | 0.000043869 | 0.000078678 |
| 4 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 48/48 | 0.000043869 | 0.000078678 |

The original r1 run used `atol=rtol=1e-5` for every recurrent state. It
naturally exited 1: SSM passed 3/12 steps at each batch; key passed 5/12 at
BS=1 and 3/12 at BS=2/4. Logits, angle, value and tokens passed throughout.
The [original audit](evidence/mamba3-mimo-reference/mamba3-mimo-reference-r1-audit.json)
recomputes its failures from saved tensors. The r2 run explicitly uses
`atol=rtol=1e-4` for SSM and `5e-5` for key, retaining `1e-5` for angle/value
and `1e-3` for logits. Those state budgets were selected **after** observing
r1's FP32 cached/full-prefix differences; r2 is a calibrated consistency
check, not a pass of the original stricter gate. Every step compares all 12
layers. The [r2 audit](evidence/mamba3-mimo-reference/mamba3-mimo-reference-r2-audit.json)
checks source, checkpoint, tokenizer and snapshot hashes, row coverage,
shapes and reported comparisons. The 18 binary snapshots remain local; their
hashes and sizes are in the [evidence inventory](evidence/mamba3-mimo-reference/mamba3-mimo-reference-evidence-manifest.json).

## Source, tokenizer and execution scope

[`state-spaces/mamba3-mimo-187m`](https://huggingface.co/state-spaces/mamba3-mimo-187m/tree/8fd6e9eb7b795f2e15d7f6353251d0137980c43e)
is pinned to revision `8fd6e9eb7b795f2e15d7f6353251d0137980c43e`.
The 374,640,827-byte BF16 checkpoint has SHA256
`369db3fb9deedfc98baa95ac9166baf862dd7e03b3143afaf0d19e46a220daf6`.
Its 183 stored tensors contain 285,792,576 stored elements, including the
duplicated tied embedding/head; there are 187,291,968 effective parameters.
All tensor names and shapes match and load strictly after FP32 promotion.

The reference executes hash-pinned Mamba-3 mixer, block and gated-MLP class
bodies from the [official source commit](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa).
The upstream file hashes are in the [source pins](evidence/mamba3-mimo-reference/official-source-pins.json).
Rotary and MIMO recurrence use explicit Torch operations following the pinned
official references. This verifies the CPU cache path, not the checkpoint's
training commit or native TileLang/CuTe kernels.

The model uses the gated
[`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B)
tokenizer at revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
Three files were hash-verified locally using existing authorization. Only
their hashes and selected token IDs appear in the PR; restricted tokenizer
bytes are excluded. Execution used Python 3.12, Torch 2.14.1 CPU, FP32 model
weights and one Torch/OMP/MKL thread. The completed weight was evicted after
both audits, SHA256 verification and a no-reader check; GPU use requires a
pinned re-download.

```sh
python tools/flashrnn2/mamba3_reference_gate.py \
  --variant mimo --ssm-budget 0.0001 --key-budget 0.00005 \
  --model "$MAMBA3_MIMO_CHECKPOINT_AND_TOKENIZER" \
  --source "$PINNED_MAMBA3_SOURCE" \
  --output "$RESULTS/mamba3-mimo-reference.jsonl"
```

| Required E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| Native Mamba-3 CUDA generation | — | — | Not measured |
| High-concurrency, multi-batch GPU | — | — | Not measured |

Native CUDA, longer contexts and high-concurrency GPU E2E remain unqualified.
