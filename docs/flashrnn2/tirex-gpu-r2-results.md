# TiRex 35M GPU numerical gate: r2 failure

The pinned TiRex 35M checkpoint ran once on an H100 with BF16, Torch
2.4.0a0+gite3b9b71 and Triton 3.0.0. The official Torch cell and the FlashRNN2
candidate were compared at the original `1e-4 × (1 + |reference|)` budget.
The run completed all 381 comparisons and naturally exited with a numerical
failure. There was no forecast E2E timing.

| Scope | Tensor pairs failed / compared | Elements failed / compared | Largest absolute error | Speedup |
|---|---:|---:|---:|---:|
| Isolated cell, B1/2/4 | 15 / 15 | 185,514 / 186,368 | 34.203125 | Not measured |
| 12 blocks × 2 forecast patches, B1/2/4 | 360 / 360 | 5,695,454 / 5,849,088 | 61.28125 | Not measured |
| 64-step quantiles and median, B1/2/4 | 6 / 6 | 4,480 / 4,480 | 2.171775 | Not measured |
| **Total** | **381 / 381** | **5,885,448 / 6,039,936** | **61.28125** | **Not measured** |

The independent NumPy float64 replay confirmed the complete 381-pair scope,
the raw tensor SHA256 `c4f4f7e53f0db2704e4ab707ecd5765b453994a58054c10aafbe66d8c15b92ed`,
all producer pass/fail flags, and all 381 failures. The [producer report](evidence/tirex-gpu-r2-failure/tirex-gpu-gate-r2.json),
[independent audit](evidence/tirex-gpu-r2-failure/independent-numpy-audit.json),
and [compressed raw tensors](evidence/tirex-gpu-r2-failure/tirex-gpu-gate-r2.pt.gz)
preserve the evidence. Decompress the PT beside the JSON and run
`tools/flashrnn2/audit_tirex_gpu_gate.py` to reproduce the audit.

The pinned Torch source contracts hidden state with recurrent weights stored
as `[head, input, gate × output]`. The r2 adapter passed those weights as
`[gate, head, input, output]` to a kernel expecting
`[gate, head, output, input]`. An offline contraction using all 12 checkpoint
matrices found 4,096 differing values per layer with the r2 layout and exact
agreement with the corrected layout; see the [checkpoint layout diagnostic](evidence/tirex-gpu-r2-failure/checkpoint-layout-diagnostic.json).
This identifies a real mapping error, but does not establish that the repaired
GPU sequence passes. A distinct numerical qualification is required before
any forecast speed comparison.
