# Mamba FP64 rounding diagnosis

The remote CPU and CUDA results previously failed comparison with the original
local native FP32 oracle. An additional full-model FP64 calculation finds that
all three saved remote CPU and all three saved CUDA cases fit the original
budgets against FP64; the original local FP32 oracle exceeds the same SSM budget
at six elements in case 1. The previous failed qualifications remain unchanged.

This is a mathematical diagnosis, not unmodified native execution. The runner
promotes six explicit FP32 casts in the pinned native methods to FP64: RMSNorm,
three selective-scan inputs, the residual, and output logits. It also converts
all weights and states to FP64. All 24 layers and 129,135,360 unique parameters
are present, and the embedding/output-head identity is checked. The source is
hash pinned as in [the native reference](mamba-native-reference.md).

Each of three B1/P5 inputs executes full prefill and one decode. Decode receives
the original oracle's next-token IDs to keep inputs aligned. Prefill and decode
greedy tokens agree for every saved implementation. All logits, convolution
caches and decode SSM caches fit the unchanged budgets. The only failures below
are in prefill SSM state.

| Case | Saved implementation | Max absolute prefill SSM error vs FP64 | Worst normalized error | Elements over budget |
| --- | --- | ---: | ---: | ---: |
| 0 | local_native_fp32 | 6.6786535e-05 | 0.990607 | 0 |
| 0 | remote_cpu_fp32 | 1.051975e-05 | 0.077110 | 0 |
| 0 | remote_cuda_fp32 | 1.6619079e-05 | 0.250265 | 0 |
| 1 | local_native_fp32 | 5.4883009e-05 | 2.015715 | 6 |
| 1 | remote_cpu_fp32 | 1.4389382e-05 | 0.163078 | 0 |
| 1 | remote_cuda_fp32 | 1.7119402e-05 | 0.628753 | 0 |
| 2 | local_native_fp32 | 1.7587913e-05 | 0.384828 | 0 |
| 2 | remote_cpu_fp32 | 4.0027722e-06 | 0.072407 | 0 |
| 2 | remote_cuda_fp32 | 7.0924469e-06 | 0.150067 | 0 |

Normalized error is `abs(actual - reference) / (atol + rtol * abs(reference))`;
values above 1 fail. Logits use `atol=rtol=1e-3`; both cache types use
`atol=rtol=1e-5`, exactly as before this diagnosis. The worst case-1 SSM coordinate
is `[23, 0, 1384, 15]`. FP32 snapshots are cast exactly to FP64 before comparison.

The local reference uses Torch 2.13 CPU; the remote saved CPU/CUDA results use
Torch 2.12.1. These comparisons demonstrate rounding in the original reference,
but do not isolate the effect of Torch version, host architecture or math
libraries. FP64 is an additional diagnostic reference, not proof that an
accelerated native CUDA backend is correct. Same-platform native execution and
longer GPU generation/serving qualification are still required.

The first diagnostic attempt exited 1 before constructing the model: the saved
snapshot schema contains `prefill` and `decode`, not `next_ids`. The second
attempt derives the saved prefill token from logits and exits 0. Both original
logs, commands and the first executed source are retained. An independent audit
rehashes inputs and recomputes all 54 tensor comparisons from the saved FP64 and
FP32 snapshots; every comparison matches the report. Neither run changes an
installed package, the native source file, nor another process's runtime.

| Required E2E comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Native/Script CUDA, multiple batch sizes | Not measured | Not measured | N/A |
| Accelerated backends, high-concurrency full-model serving | Not measured | Not measured | N/A |

No diagnostic CPU duration is used as performance evidence. All model and
snapshot binaries remain task-owned inputs for the unfinished GPU campaign.
The original [numerical-window results](cuda-numerical-diagnosis.md) are retained.

Reproduce with the same verified input artifacts and source pin:

```bash
python tools/flashrnn2/mamba_fp64_diagnostic.py \
  --model "$MODEL/prefill-decode-b1-t5-cuda.pt" \
  --source "$PINNED_SOURCE/modeling_mamba.py" --config "$MODEL/config.json" \
  --goldens "$MODEL/script-goldens-b1-t5.pt" \
  --remote-cpu "$SAVED/mamba-script-diagnostic-cpu-remote-r1.pt" \
  --remote-cuda "$SAVED/mamba-script-diagnostic-cuda-remote-r1.pt" \
  --output "$RESULTS/mamba-fp64.json"
```

Evidence: [raw report](evidence/mamba-fp64-math-r2.json),
[independent audit](evidence/mamba-fp64-math-r2-audit.json),
[actual commands and natural exit](evidence/mamba-fp64-math-r2-controller.json),
[first failure](evidence/mamba-fp64-math-r1.log), and
[file manifest](evidence/mamba-fp64-evidence-manifest.json).
The report records all six promoted AST sites, source hashes and snapshot hashes.
The FP64 snapshot SHA256 is
`8b3f3a0293e25541052d32fdb8e666ffd9db76de1d26e6fb5c9eec84ca243272`.
