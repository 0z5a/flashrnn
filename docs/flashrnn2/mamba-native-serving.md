# Native Mamba queued-request qualification

The serving runner now accepts a complete native Torch fallback baseline for
both prefill and decode. Previously the measured alternatives shared scripted
prefill. Native/Script and native/decode-Graph comparisons now carry explicit
arm names and independently qualified native source and weight provenance.
The existing Script/Graph defaults remain available.

The complete 24-layer, 129,135,360-parameter Mamba-130M model passes CPU
qualification for both native and Script paths. Each request has a P128 prompt
and 32 greedy output tokens. The matrix executes 12 cohorts per path:
**511 requests and 16,352 tokens per path; 1,022 request executions and 32,704
tokens combined**. Every recorded token matches the saved native goldens.
The runner reports exact final convolution and SSM caches in every group.

| Model batch B | Queued/resident requests C | Requests per path | Tokens per path | Final cache max absolute error, both paths | Result, both paths |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 1 | 32 | 0 | PASS |
| 1 | 8 | 8 | 256 | 0 | PASS |
| 1 | 32 | 32 | 1024 | 0 | PASS |
| 2 | 2 | 2 | 64 | 0 | PASS |
| 2 | 16 | 16 | 512 | 0 | PASS |
| 2 | 32 | 32 | 1024 | 0 | PASS |
| 2 | 64 | 64 | 2048 | 0 | PASS |
| 2 | 128 | 128 | 4096 | 0 | PASS |
| 4 | 4 | 4 | 128 | 0 | PASS |
| 4 | 32 | 32 | 1024 | 0 | PASS |
| 4 | 64 | 64 | 2048 | 0 | PASS |
| 4 | 128 | 128 | 4096 | 0 | PASS |

All C requests arrive together. The single worker prefills B-sized groups and
then decodes them round robin, preserving separate state for every group.
Three fixed prompt groups cycle deterministically. This verifies queued
multi-request state isolation; C is not a GPU kernel batch or a count of CPU
workers. Actual model batches and peak resident request counts are audited
against B and C for every row. Tokenization and HTTP transport are outside this
runner, as are model loading and Graph setup.

The source and checkpoint requirements are described in
[the native reference](mamba-native-reference.md). Each native arm requires a
PASS generation qualification on the same device, model artifact, source hashes
and oracle artifact. Native and candidate execute their own full prefill/decode
paths. All loaded models and Graph storage stay resident across arms, so peak
memory is a process total and cannot establish per-arm capacity savings.

| Required GPU E2E comparison | Native baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Native vs Script, B1/B2/B4 and C up to 128 | Not measured | Not measured | N/A |
| Native vs decode Graph, B1/B2/B4 and C up to 128 | Not measured | Not measured | N/A |
| Strongest accelerated backends, HTTP and fresh-process serving | Not measured | Not measured | N/A |

The real CPU qualification input is rejected by the speed analyzer, producing
no speed table. CPU wall times remain in raw records but are not used for GPU
performance claims. CUDA/Graph execution, native accelerated backends and the
broader model/service campaign remain incomplete. Original numerical failures
and error budgets are unchanged.

The analyzer labels the actual pair and requires at least 20 matching AB/BA
blocks, equal unique request IDs, complete equal token lists, equal actual
model batch sequences and equal work. Synthetic contract tests cover two
valid native pairs and five rejected inputs: qualification-only data, duplicate
request IDs, truncated token lists, different tokens and missing pairs. These
seven subcases are test fixtures, not measured GPU results.

Controller 26723 and B1/B2/B4 children 26729/27546/31628 all naturally exited 0.
The audit rechecks frozen source hashes, actual row counts, model batches,
request IDs and all tokens against hash-verified saved native goldens. It checks
reported cache errors; it does not reconstruct unrecorded cache tensors. The
existing CPU Torch 2.13 environment uses one thread and offline inputs. Model
files remain needed for the GPU campaign.

Reproduce one CPU cohort with verified model-specific generation receipts:

```bash
python tools/flashrnn2/mamba_serving.py \
  --model "$MODEL/prefill-decode-b4-t128.pt" \
  --goldens "$MODEL/generation-b4-p128-g32.pt" \
  --qualification "$RESULTS/mamba-generation-b4-r1.meta.json" \
  --baseline native --candidate script \
  --native-model "$MODEL/prefill-decode-b1-t5-cuda.pt" \
  --native-source "$PINNED_SOURCE/modeling_mamba.py" \
  --native-config "$MODEL/config.json" \
  --native-qualification "$RESULTS/mamba-native-generation-b4-r1.meta.json" \
  --device cpu --verify-only --concurrency 4 32 64 128 \
  --output "$RESULTS/native-serving-b4.jsonl"
```

Evidence: [complete commands and natural exits](evidence/mamba-native-serving-r1-controller.json),
[request/token audit](evidence/mamba-native-serving-r1-audit.json),
[analyzer test receipt](evidence/native-serving-analysis-tests-r1.json),
[actual CPU speed-report rejection](evidence/native-serving-speed-rejection-r1-receipt.json),
and [raw-file manifest](evidence/mamba-native-serving-evidence-manifest.json).
Per-batch JSONL, metadata and logs are retained beside the manifest.
