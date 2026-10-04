# Complete Mamba generation and cache isolation

The full Mamba-130M checkpoint now has qualified CPU TorchScript exports
for B1, B2 and B4 at a 128-token prompt length. All 24 layers and 129,135,360
parameters execute. The checkpoint and exporter are unchanged from the
earlier B1/T5 qualification; each new static shape has its own artifact hash.

Three independently sampled token inputs per batch passed export checks:
prefill logits, convolution cache and SSM cache, followed by one decode,
are all bitwise equal to native Transformers. These export inputs are
separate from the text prompts used for generation below.

## Full-model generation results

Native Transformers produces independent FP32 CPU oracles for three text
groups per batch. Each request consumes 128 prompt tokens and emits exactly
32 greedy tokens. Prefill emits the first token; 31 decode calls emit the
remaining tokens. EOS does not shorten this fixed-workload experiment.
Final caches therefore represent the prompt plus the first 31 generated tokens.

| Batch | Independent text groups | Schedules | Compared generation steps | Maximum absolute error | Greedy tokens | GPU E2E speedup |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 3 | Serial, interleaved | 192 | 0 | All match | Not measured |
| 2 | 3 | Serial, interleaved | 192 | 0 | All match | Not measured |
| 4 | 3 | Serial, interleaved | 192 | 0 | All match | Not measured |

Every emitted token compares the complete vocabulary logits. All layers'
convolution and SSM caches are compared after prefill and after the last
decode; intermediate cache tensors are not retained. Across the two
schedules there are 576 batch-step comparisons, representing 1,344 individual
token choices. All pass bitwise, including all saved cache comparisons.
The three generation controllers naturally exited 0.

Interleaving retains three independent request groups and advances each by
one token in round-robin order on the same execution stream. This checks
request cache isolation. It does not measure concurrent CUDA streams,
queueing latency or serving throughput.

The acceptance budgets remain logits atol/rtol 0.001 and caches 1e-5/1e-5,
fixed before these runs. The original stricter cached/full-prefix failures
remain in the native-model reports. A CPU pass does not qualify CUDA
retargeting, cross-version serialization or a different batch/prompt shape.

## Cache fault controls

A separate B4 first-decode experiment feeds the correct next token while
deliberately corrupting only the cache source. The unchanged logits budget
accepts the unmodified cache and rejects all three injected faults:

| Control | Maximum logit error | Frozen-budget result |
| --- | ---: | --- |
| Unmodified request cache | 0 | Accepted |
| Convolution cache from another text group | 74.394531 | Rejected as expected |
| SSM cache from another text group | 123.240036 | Rejected as expected |
| Stale zero cache | 116.179764 | Rejected as expected |

The control controller naturally exited 0 because all expected decisions
were observed. These are intentional faults, not accepted model outputs.

## Input and artifact provenance

Prompts come from [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`,
`wikitext-2-raw-v1/validation-00000-of-00001.parquet`.
The 657,209-byte file matches SHA256
`204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`.
The immutable manifest and download came through hf-mirror; independent
canonical API confirmation failed with a connection timeout and is not claimed.

Selection is deterministic: take the first 12 rows with at least 128 tokens,
without special tokens, then their first 128 tokens. Each group has four rows;
B1/B2 use its first one/two rows and B4 uses all four. Metadata records the
row indices, input-token hashes, output-token hashes, tokenizer hash,
checkpoint revision, native implementation hash and harness hashes.

Weights, text, token arrays and oracle tensors remain in task-owned local
storage. Only small metadata and raw comparison records are committed.
The existing runtime is Torch 2.13.0 / Transformers 4.54.1; no packages were
installed or updated. Model artifacts remain until the GPU and full
concurrency campaign completes.

## Reproduction

```sh
# DATASET is the hash-verified pinned validation parquet.
python tools/flashrnn2/mamba_generation_goldens.py --model "$MODEL" --dataset "$DATASET" --output "$MODEL"
python tools/flashrnn2/export_mamba.py --model "$MODEL" --batch 4 --prompt-length 128 --output "$CPU_ARTIFACT"
python tools/flashrnn2/mamba_generation_gate.py --model "$CPU_ARTIFACT" --goldens "$ORACLES" --device cpu --output "$RESULTS"
python tools/flashrnn2/mamba_cache_negative.py --model "$CPU_ARTIFACT" --goldens "$ORACLES" --device cpu --output "$NEGATIVE_RESULTS"
```

Repeat export and generation checks with batch 1 and 2. The original B1/T5
CUDA package and finite gate remain a separate frozen request; these longer
artifacts have not been transferred to or executed on a GPU.
