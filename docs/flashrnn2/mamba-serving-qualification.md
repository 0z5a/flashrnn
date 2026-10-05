# Complete Mamba queued-request qualification

The runner measures an in-process path from queued token IDs to host-visible
generated tokens. It executes the complete pinned Mamba130M checkpoint,
all 24 layers and 129,135,360 parameters, with static batches and the same
P128/G32 WikiText cases as the [generation gate](mamba-generation-qualification.md).
It does not yet cover an HTTP server, tokenization, continuous arrivals,
variable lengths, EOS termination or a latency-constrained serving SLO.

All C requests arrive at time zero. One worker allocates request-owned state,
prefills each B-sized group, then advances the groups round-robin. The interval
includes queue wait, state allocation/reset, prompt transfer, model execution,
argmax and blocking token delivery. Model loading, graph setup, oracle checks
and JSON output are outside the interval. Each request records its identity,
generated tokens, admission time and all 32 delivery timestamps.

## Executed CPU functional checks

| B | C | Requests | Generated tokens | Exact oracle tokens | Final conv/SSM cache gate | Speed comparison |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | 1, 8, 32 | 41 | 1,312 | PASS | PASS | Not measured |
| 2 | 2, 16, 32, 64, 128 | 242 | 7,744 | PASS | PASS | Not measured |
| 4 | 4, 32, 64, 128 | 228 | 7,296 | PASS | PASS | Not measured |

All three controllers naturally exit 0: 12 cohorts, 511 requests and
16,352 generated tokens. Final caches are bitwise equal to the native oracle.
These are verification-only runs. Their elapsed times include the first
cold execution and must not support a performance claim. The analyzer rejects
these actual CPU results and produces no speed table.

The B2/B4 runs record every observed model batch: C/B prefill calls and
31*C/B decode calls, all matching the declared B. Peak live request state equals C.
Independent readback verifies unique request IDs, token counts, increasing
delivery timestamps, admission-before-first-token and all final cache checks.
The earlier B1 r1 `actual_batches` field records admission groups only;
it does not count decode calls. The current runner distinguishes these fields.

R1/r2 calculated TPOT/ITL quantiles over groups. The current implementation
weights every request, including replicated timestamps within a batch. This
can change finite-sample linear interpolation for B>1. Historical diagnostic
times remain unchanged. The exact old harnesses can be reconstructed using
`mamba-serving-current-to-r1.patch` and `mamba-serving-current-to-r2.patch`
from the evidence directory; their reconstructed hashes match execution
metadata. These patches apply to the harness in this commit.

## CUDA comparison still unexecuted

`GraphDecode` captures one static decode, resets its inputs before replay and
clones returned tensors so each group owns its state. Before any measured
cohort, the runner requires an independent successful generation qualification
for the exact model/oracle artifacts and device, then verifies graph logits,
tokens and final caches. This graph path has not executed on CUDA yet.
The separate B1/T5 [CUDA qualification](mamba-script-qualification.md) retained
two prefill SSM failures and cannot satisfy this prerequisite.

For a future qualified CUDA run, both arms use the same cases, observed
batches and token counts, with at least 20 alternating AB/BA pairs per C.
Analysis requires the exact planned concurrency and block sets. It reports
paired throughput ratios and bootstrap intervals within one process; fresh
process replication remains separate. Cohort p99 is a finite-sample quantile.
Graph buffers and private pools stay resident across both arms, so process
peak memory is not an isolated per-arm footprint or a capacity-saving result.

| Full-model CUDA path | Baseline tok/s | Candidate tok/s | Speedup | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| P128/G32 multi-BS/high-C | Not measured | Not measured | N/A | N/A |

```sh
python tools/flashrnn2/mamba_serving.py --model CPU_B4_P128_ARTIFACT \
  --goldens B4_P128_G32_ORACLES --qualification CPU_GENERATION_META \
  --device cpu --concurrency 4 32 64 128 --verify-only --output cpu.jsonl
# Only after separate successful CUDA generation and graph qualifications:
python tools/flashrnn2/mamba_serving.py --model CUDA_B4_P128_ARTIFACT \
  --goldens B4_P128_G32_ORACLES --qualification CUDA_GENERATION_META \
  --device cuda --concurrency 4 32 64 128 --blocks 20 --output paired.jsonl
python tools/flashrnn2/analyze_serving.py paired.jsonl --output comparison
```

Raw cases, metadata, natural controller exits and source reconstruction checks
are in [evidence](evidence/). Model weights and cache oracle binaries remain
outside the repository until their full campaign and consumers finish.
