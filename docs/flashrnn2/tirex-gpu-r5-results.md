# TiRex 35M guarded GPU numerical qualification

The pinned TiRex 35M checkpoint completed one H100 BF16 qualification with
Torch 2.4.0a0+gite3b9b71 and Triton 3.0.0. The candidate dispatches B2 to
FlashRNN2 and retains the original TiRex Torch GPU cell at B1, B4, and other
batch sizes. At the unchanged `1e-4 × (1 + |reference|)` budget, all 381
required tensor pairs passed. The scientific process, wrapper, and GPU step
exited normally with code 0. The independent NumPy float64 audit replayed all
6,039,936 tensor elements with zero failures, missing pairs, unexpected pairs,
or disagreements with the producer. The largest error divided by its allowed
budget was 0.07618. Full-model forecast throughput was **not measured**.

| Batch | Isolated candidate cell | Full 12-block forecast | Accelerated full-forecast path | Largest absolute error | Failed pairs | Forecast speedup |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 5/5 pass | 122/122 pass | Original Torch fallback | 0.00000763 | 0/127 | Not measured |
| 2 | 5/5 pass | 122/122 pass | FlashRNN2 | 0 | 0/127 | Not measured |
| 4 | 5/5 pass | 122/122 pass | Original Torch fallback | 0 | 0/127 | Not measured |
| **Total** | **15/15** | **366/366** | **B2 only** | **0.00000763** | **0/381** | **Not measured** |

Each full forecast includes two 32-step patches through 12 blocks, every
cell output and four recurrent states, and the final 64-step quantiles and
median. The isolated-cell controls call the FlashRNN2 cell at every tested
batch; they do not change the B1/B4 full-model fallback contract. Only the B2
full-forecast path is evidence for FlashRNN2 acceleration.

The [producer report](evidence/tirex-gpu-r5-pass/tirex-gpu-gate-r5.json),
[independent audit](evidence/tirex-gpu-r5-pass/independent-numpy-audit.json),
and [compressed raw tensors](evidence/tirex-gpu-r5-pass/tirex-gpu-gate-r5.pt.gz)
preserve the numerical result. Decompress the PT beside the JSON and run
`tools/flashrnn2/audit_tirex_gpu_gate.py` to replay the audit. The raw PT
SHA256 is `8c1dfc999976c752453e15402376a7863f1cb3fd0ec68713aeec04ec2d20b988`;
the producer JSON SHA256 is
`72d239288c9e61d3ba3fc1fd7e9b280c14645b21cf3c0682075e7b7d42d7b991`;
the independent audit SHA256 is
`ced617a01da1fedb223eaa566910deeb50042bee274e3055c75aa280d6195174`.
The complete private transport archive contained 192 independently verified
payload files and has SHA256
`6d39048eb99e5863281a9b014ee0e05e2c4e4af406e4776815a7d8a19215a20d`.

This qualification permits a separate paired full-forecast E2E experiment.
It is not a speed measurement. B1/B4 rows in that experiment will measure
fallback dispatch overhead against Torch, not FlashRNN2 kernel speedup.
