# TiRex 35M GPU numerical gate: r3 failure

The recurrent weight-axis correction was qualified once on an H100 with BF16,
Torch 2.4.0a0+gite3b9b71, and Triton 3.0.0. The pinned checkpoint and official
Torch cell were compared with the FlashRNN2 candidate at the unchanged
`1e-4 × (1 + |reference|)` budget. All 381 required tensor pairs were saved.
The scientific process naturally exited with `FAIL_NUMERICAL`; no forecast E2E
timing ran.

| Scope | Tensor pairs failed / compared | Elements failed / compared | Largest absolute error | Speedup |
|---|---:|---:|---:|---:|
| Isolated cell, B1/2/4 | 5 / 15 | 1,401 / 186,368 | 0.09375 | Not measured |
| 12 blocks × 2 forecast patches, B1/2/4 | 296 / 360 | 383,302 / 5,849,088 | 0.75 | Not measured |
| 64-step quantiles and median, B1/2/4 | 6 / 6 | 3,455 / 4,480 | 0.00622657 | Not measured |
| **Total** | **307 / 381** | **388,158 / 6,039,936** | **0.75** | **Not measured** |

An independent NumPy float64 replay verified the raw tensor SHA256
`2d4027edd00c11d52aa956767d29a948f56e695910c34040a428ef0b052f37b1`,
complete comparison scope, and every producer flag, with zero disagreements.
The [producer report](evidence/tirex-gpu-r3-failure/tirex-gpu-gate-r3.json),
[independent audit](evidence/tirex-gpu-r3-failure/independent-numpy-audit.json),
and [compressed raw tensors](evidence/tirex-gpu-r3-failure/tirex-gpu-gate-r3.pt.gz)
preserve the result. Decompress the PT beside the JSON and run
`tools/flashrnn2/audit_tirex_gpu_gate.py` to reproduce the audit.

The correction improved the failed-pair count from the
[r2 result](tirex-gpu-r2-results.md), but did not pass. In the isolated cell,
B2 passed all five comparisons exactly; B4 first differed in the output at
step 13 and first exceeded the budget at step 35. In the full forecast,
the B1 first-block output first differed at step 11 and first exceeded the
budget at step 16. These observations localize the remaining discrepancy;
they do not yet establish whether the cause is recurrent projection rounding
or pointwise arithmetic. A new diagnostic and a passing GPU qualification
are required before any pretrained forecast speed claim.
