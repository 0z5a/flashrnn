# TiRex 35M GPU numerical gate: r4 failure

The CUDA-libdevice pointwise candidate ran once on an H100 with BF16, Torch
2.4.0a0+gite3b9b71, and Triton 3.0.0. It used the pinned TiRex 35M checkpoint
and the unchanged `1e-4 × (1 + |reference|)` budget. The scientific process
naturally exited with `FAIL_NUMERICAL`; no forecast E2E timing ran.

| Scope | Tensor pairs failed / compared | Elements failed / compared | Largest absolute error | Speedup |
|---|---:|---:|---:|---:|
| Isolated cell, B1/2/4 | 0 / 15 | 0 / 186,368 | 0.00000763 | Not measured |
| 12 blocks × 2 forecast patches, B1/2/4 | 202 / 360 | 171,966 / 5,849,088 | 0.25 | Not measured |
| 64-step quantiles and median, B1/2/4 | 4 / 6 | 2,650 / 4,480 | 0.00377715 | Not measured |
| **Total** | **206 / 381** | **174,616 / 6,039,936** | **0.25** | **Not measured** |

The independent NumPy float64 replay verified all 381 pairs, the raw tensor
SHA256 `7ed3f4608a02016f0aae9ea1650f06ddcdb867984bd3ab6a81bbb7f5b9301fb1`,
and every producer pass/fail flag with zero disagreements. The
[producer report](evidence/tirex-gpu-r4-failure/tirex-gpu-gate-r4.json),
[independent audit](evidence/tirex-gpu-r4-failure/independent-numpy-audit.json),
and [compressed raw tensors](evidence/tirex-gpu-r4-failure/tirex-gpu-gate-r4.pt.gz)
preserve the result. Decompress the PT beside the JSON and run
`tools/flashrnn2/audit_tirex_gpu_gate.py` to reproduce the audit.

All 127 B2 pairs passed exactly, including every block, both forecast patches,
and final forecast outputs. All isolated-cell pairs passed; the B4 isolated
cell was exact. The full B1 forecast first differs in block 0 at step 11 and
first exceeds the budget at step 16. B4's first three forecast blocks pass
exactly; block 3 first differs at step 45 and first exceeds the budget at step
47. Later recurrent blocks amplify those differences.

In the first B1 forecast block, the candidate output equals the first row of
the B2 Torch reference exactly, while the B1 Torch reference differs from
that row at 177 elements in patch 0 and 176 in patch 1. This is evidence of
batch-sensitive numerical behavior in this workload, not yet a diagnosis of
the underlying GPU operation. A passing qualification is still required for
each reported batch configuration before any pretrained forecast speed claim.
