# Native Mamba CPU/CUDA and PTX loading

The full Mamba130M native Torch fallback and traced candidate pass all three
short cases on each execution platform. CPU comparisons are bitwise equal;
CUDA comparisons satisfy the unchanged elementwise budgets. Comparisons with
the original local CPU oracle still fail on prefill SSM states. The same-device
result does not replace those failures or establish long-generation accuracy.

| Runtime | Case | Native vs traced max absolute error, all outputs | Same-device result | Native vs original oracle | Failed original-oracle elements |
| --- | ---: | ---: | --- | --- | ---: |
| Remote CPU | 0 | 0 | Pass | Pass | 0 |
| Remote CPU | 1 | 0 | Pass | Fail | 6 |
| Remote CPU | 2 | 0 | Pass | Pass | 0 |
| RTX5090 CUDA | 0 | 0 | Pass | Fail | 1 |
| RTX5090 CUDA | 1 | 0.0000762939 | Pass | Fail | 7 |
| RTX5090 CUDA | 2 | 0.0000877380 | Pass | Pass | 0 |

Every failure is in prefill SSM state; all prefill/decode greedy tokens agree.
The unchanged budgets are `atol=rtol=1e-3` for logits and `1e-5` for caches.
Each case uses B1/P5 and one cached decode, all 24 layers and 129,135,360
parameters. Both remote runs use Torch 2.12.1+cu130; the original CPU oracles
use Torch 2.13.0. This comparison does not isolate the cause of cross-platform
rounding or qualify the official fused Mamba kernels.

The native fallback executes the pinned Transformers methods described in
[the native reference](mamba-native-reference.md). The existing serialized
artifact only supplies its checkpoint weights. The local checkpoint-to-artifact
identity was verified in that earlier qualification; this remote run verifies
the resident artifact hash and does not reread a safetensors checkpoint.
Inherited artifact metadata such as `CUDA_RETARGETED_NOT_EXECUTED` describes
its preparation history, not the status of these newly executed cases.

The independent [audit](evidence/native-runtime-window-r1-audit.json) recomputes
36 persisted native-vs-original tensor comparisons, including maximum error,
failure count, worst coordinate and per-layer counts. All match the raw records.
Relative L2 is diagnostic only: local and recorded reductions differ by at most
3.54703e-10. The first audit's stricter norm-equality assertion failed and is
retained; its successor records both values without changing any elementwise
acceptance budget. Fresh traced tensors were not saved, so native-vs-traced
rows are the original runner checks, not independently recomputed tensor pairs.

## PTX diagnosis

All six PTX records extracted from the existing seven-unit native library load
and unload through `cuModuleLoadDataEx`/`cuModuleUnload` with `CUDA_SUCCESS` and
empty JIT error logs. Each PTX byte hash is independently checked against the
captured dump. The runtime uses driver API 13000 and
`libcuda.so.580.105.08`. No function lookup, explicit function loading, kernel
launch, extension forward or backward is performed in this probe.

The prior extension's first-forward `cudaErrorInvalidPtx` remains unresolved.
Module loading alone does not prove every function is executable; CUDA exposes
separate function loading and lazy-loading state APIs in its
[module management reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MODULE.html).
The two records without entry functions correspond to host-only translation
units; their absence of kernel entries is not a detected defect.

| Required E2E comparison | Native tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Mamba CUDA B1/B2/B4, P128/G32 generation | Not measured | Not measured | N/A |
| Multi-batch/high-concurrency GPU serving | Not measured | Not measured | N/A |
| Native FlashRNN CUDA recurrence | First-forward failure retained | Not measured | N/A |

All three children, the controller and SSH complete naturally with exit 0.
For Mamba this exit means same-device parity, not original-oracle acceptance.
The 39,423,327-byte result archive contains 15 files, all size/hash verified;
SHA256 `b81f8e9b76e61b3d7adf77b60162b6c305877f31b7bf1ecf2b0e6ffa5943148c`.
The original GPU1/IO locks were independently acquired/released after task PIDs
and model readers disappeared, then returned to the coordinator. No packages,
shared environment, process signals or model weights were changed.

Evidence: [exact commands/exits](evidence/native-runtime-window-r1-controller.json),
[CPU records](evidence/mamba-native-cpu-remote-r1.jsonl),
[CUDA records](evidence/mamba-native-cuda-remote-r1.jsonl),
[PTX records](evidence/native-ptx-jit-r1.json),
[return receipt](evidence/native-runtime-window-r1-handoff.json), and
[published file inventory](evidence/native-runtime-evidence-manifest.json).
Large tensor snapshots remain in the verified local archive rather than Git.
