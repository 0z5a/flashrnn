# Recurrent tiling correctness, SM120

The experiment compares an inferred single-CTA layout, explicit Gluon register operands, and output-coordinate tiling with a kernel boundary between timesteps. Each owner computes all four gates for its output coordinates. The timestep version has `persistent=false`; it is not a persistent cross-CTA implementation. No production dispatcher is changed.

Thirty executable forward cases produced 29 PASS and one retained sLSTM failure. Six additional records explicitly report unsupported configurations. All thirty executable cases completed; the process naturally exited1 because the strict gate failed. Raw [records](evidence/gpu-gate-r4.jsonl), [source/runtime metadata](evidence/gpu-gate-r4.meta.json) and [controller receipt](evidence/gpu-r4-controller.json) are retained.

| Check | Result |
| --- | --- |
| LSTM/sLSTM C0/C1, all three implementations | PASS |
| C1 repeat, two concurrent streams, unchanged inputs and changed R | PASS for both cells and all three implementations |
| sLSTM global-zero / mixed-normalizer initial state | PASS |
| Two heads, B3/T33/D48 | PASS for both cells and all three implementations |
| Two heads, B3/T33/D128, explicit register layout | PASS for both cells |
| B16/T128/D256, timestep tiling | LSTM PASS; sLSTM strict absolute gate failed |
| D256 single CTA | Unsupported by these prototypes |
| D128 inferred single CTA on SM120 | Unsupported: observed137216 shared bytes >101376 hardware limit |

| Compiled resource observation | Inferred layout | Explicit register layout |
| --- | --- | --- |
| LSTM C1 D64 shared bytes | 36864 | 8192 |
| LSTM C1 D64 registers / reported spills | 110 / 0 | 176 / 0 |
| sLSTM C1 D64 registers / reported spills | 128 / 2 | 224 / 0 |
| LSTM B3/T33/H2/D128 shared bytes | 137216; launch rejected | 16384 |
| LSTM B3/T33/H2/D128 registers / reported spills | Not executed | 255 / 0 |
| sLSTM B3/T33/H2/D128 registers / reported spills | Not executed | 255 / 30 |

Explicit layouts resolve the observed D128 shared-memory capacity failure. The sLSTM D128 spill count remains a performance concern. PTX contains `mma.sync`; this is not Hopper WGMMA or SM100 TMEM evidence. The inferred R-as-A orientation was also compiled separately and required135168 shared bytes at D128, still above this device limit.

The fixed gate uses atol0.002, rtol0.02 and absolute maximum0.003. At the failed sLSTM prototype coordinate S2/B2/T106/H0/D78, the FP32 values are2.148437261581421 and2.1484375. The latter is exactly a BF16 midpoint; outputs round to2.140625 and2.15625. [Exact-coordinate evidence](evidence/r2-rounding-location.json) confirms the rounding split. Five elements exceed the original absolute gate. The gate has not been widened from candidate results; independent original fused-CUDA per-state calibration remains required.

| Workload | Strong matched baseline | Candidate | Speedup |
| --- | --- | --- | --- |
| Recurrence public API | Pending qualification | Forward correctness evaluated above | Not measured |
| Projection + recurrence layer | torch.nn mapping qualified separately | Pending matched timings | Not measured |
| Full backward / training | Pending | Unimplemented | Not measured |
| Full pretrained model, multiple batches / concurrency | Pending | Pending | Not measured |

Current support is FP16/BF16 CUDA input, but only BF16 has been evaluated here. Gluon and inferred single-CTA wrappers accept16≤D≤128; the observed SM120 inferred D128 resource limit is recorded explicitly. Inputs may be materialized contiguous in the wrapper, which must be included in API/layer timing. FP32 snapshots are diagnostic only. No backward, long-sequence, graph replay, full model or serving claim is made.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python tools/flashrnn2/gpu_gate.py --output artifacts/gpu-gate.jsonl
```

The nonzero exit is expected while the original sLSTM absolute gate remains failed. Do not discard the failed row or reinterpret it as a complete suite pass.
