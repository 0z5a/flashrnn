# Mamba-3 SISO 187M complete-checkpoint CPU reference

The pinned 12-layer public checkpoint passed cached generation against
full-prefix recomputation at BS=1/2/4 for three prompt groups, five prompt
tokens and four generated tokens. All 36 full-vocabulary logit checks and all
36 checks for each of the angle, SSM, key and value caches pass. All 84 greedy
token choices match. The model process naturally exited 0; an independent
audit recomputed all 36 saved logit pairs and 144 complete final layer-state
pairs from nine binary snapshots.

| Batch | Logits | Angle | SSM | Key | Value | Matching tokens | Max logit error | Max SSM error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 0.000086784 | 0.000006378 |
| 2 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 24/24 | 0.000045300 | 0.000006735 |
| 4 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 48/48 | 0.000045300 | 0.000006735 |

The frozen budgets are `atol=rtol=1e-3` for logits and `atol=rtol=1e-5`
for every state family. The largest absolute key-state difference at BS=1 is
`0.000010729`, above the absolute term alone but inside the elementwise
absolute-plus-relative budget. Every step compares all 12 layers.
The [audit](evidence/mamba3-siso-reference/mamba3-siso-reference-r1-audit.json)
checks source, checkpoint, tokenizer and snapshot hashes, row coverage, shapes
and reported comparisons. The binary snapshots stay in the execution workspace;
their sizes and hashes are in the [evidence inventory](evidence/mamba3-siso-reference/mamba3-siso-reference-evidence-manifest.json).

## Source, tokenizer and execution scope

[`state-spaces/mamba3-siso-187m`](https://huggingface.co/state-spaces/mamba3-siso-187m/tree/6792c27c00f3bb41506db1066dcd1c51bb0f4b02)
is pinned to revision `6792c27c00f3bb41506db1066dcd1c51bb0f4b02`.
The 373,744,619-byte BF16 checkpoint has SHA256
`b462d55e9f8ab9746be641d4de550b51bf65e55ab662e66835588b5a458c95ec`.
Its 147 stored tensors include both tied embedding/head entries: 285,350,208
stored elements and 186,849,600 effective model parameters. All tensor names
and shapes match the model and load strictly after FP32 promotion.

The reference executes hash-pinned Mamba-3 mixer, block and gated-MLP class
bodies from the [official source commit](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa).
The exact upstream file hashes are in the [source pins](evidence/mamba3-siso-reference/official-source-pins.json);
the upstream source bytes are fetched from that commit for reproduction.
Rotary and recurrence operations use explicit Torch math following the pinned
official reference functions. The cached single-token call normalizes its input
shape before entering the official mixer step. This is structural source
matching and CPU cache-path validation, not proof of the checkpoint's exact
training commit or native kernel accuracy.

The model card specifies the gated
[`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B)
tokenizer. Its three required files were accessed with existing authorization,
pinned to revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` and
hash-verified locally. The tokenizer content is not included in this PR; only
its hashes and selected token IDs appear in the evidence. Execution used
Python 3.12, Torch 2.14.1 CPU, FP32 model weights and one Torch/OMP/MKL thread.
The weight was evicted after audit, checksum verification and a no-reader check;
GPU use requires pinned re-download and verification.

```sh
python tools/flashrnn2/mamba3_reference_gate.py \
  --model "$MAMBA3_SISO_CHECKPOINT_AND_TOKENIZER" \
  --source "$PINNED_MAMBA3_SOURCE" \
  --output "$RESULTS/mamba3-siso-reference.jsonl"
```

| Required E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| Native Mamba-3 CUDA generation | — | — | Not measured |
| High-concurrency, multi-batch GPU | — | — | Not measured |

Native Mamba-3 CUDA, longer contexts, MIMO and high-concurrency GPU E2E remain
unqualified.
