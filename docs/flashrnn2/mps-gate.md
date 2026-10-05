# Apple MPS full-state and gradient qualification

The public Torch backend completes all 38 small-tensor cases on an Apple M5
MPS device, with CPU fallback disabled. LSTM, sLSTM, GRU and Elman all pass full
history, final carry, four input gradients, split continuation, readonly-input
and repeat checks. This qualifies eager execution on this recorded build.

The portable runner now accepts `mps` and an explicit FP32 mathematical-input
configuration. Mathematical references remain FP64 CPU calculations on the
same representable FP32 inputs. The BF16-operand reference retains FP32 states
and explicit per-step casts on CPU. The public implementation is unchanged;
there is no implicit device transfer in it. Only the qualification oracle and
saved evidence use CPU tensors.

| Device / input policy | Mathematical cases | BF16-operand cases | Natural exit |
|---|---:|---:|---:|
| CPU, original FP64 mathematical inputs | 19/19 | 19/19 | 0 |
| CPU, FP32 mathematical inputs | 19/19 | 19/19 | 0 |
| Apple MPS, FP32 mathematical inputs | 19/19 | 19/19 | 0 |

| MPS policy | Maximum forward error | Maximum input-gradient error | Maximum chunk-gradient error |
|---|---:|---:|---:|
| Mathematical FP32 versus FP64 reference | 1.74233e-6 | 1.99241e-5 | 4.76837e-7 |
| FP32 state / BF16 operands versus CPU same-cast reference | 4.76837e-7 | 1.52588e-5 | 5.96046e-8 |

The FP32 configurations freeze `atol=rtol=1e-5` for forward and gradients.
Acceptance is `abs(actual-reference) <= 1e-5 + 1e-5*abs(reference)`; a maximum
absolute error over 1e-5 can pass at a larger reference magnitude. Original
FP64 gates keep `1e-12` forward and `1e-11` gradient tolerances. All three
independent audits recompute 456 tensor pairs and 176 per-state comparisons,
plus dtype, readonly and repeat checks. Original device placement remains a
runner assertion because snapshots are explicitly stored on CPU.

The matrix uses the same fixed seeds, four B/T/H/D shapes and sLSTM zero,
mixed-normalizer and saturated-gate cases as [portable qualification](portable-gate.md).
It runs on Torch 2.13.0, Python 3.12.14, macOS 26.6.1, Apple M5, with one CPU
thread and `PYTORCH_ENABLE_MPS_FALLBACK=0`. The existing environment is unchanged.
The controller waits for each CPU control and then MPS to finish naturally.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/flashrnn2/portable_gate.py \
  --device mps --config tools/flashrnn2/portable-fp32-small.json \
  --output results/mps.jsonl
```

[Evidence](../../evidence/flashrnn2/mps-gate/) contains raw rows, source hashes,
controller exits, independent audits, the executed controller/auditor and a
file manifest. Both sides of every tensor comparison remain in the execution
workspace under `evidence/portable-{cpu-r3,fp32-cpu-r1,mps-r1}.tensors/`; their
hashes are published in the rows. Binary tensors are not added to Git.

| Performance scope | Baseline | Candidate | Speedup |
|---|---|---|---|
| Eager state/gradient correctness | CPU reference | MPS | Not measured |
| Full-model/high-concurrency E2E | Not run | Not run | Not measured |

MPS FP64 input execution is explicitly rejected by this runner. This result
does not qualify MPS compile, dedicated recurrent kernels, multi-stream MPS,
CUDA, ROCm or XPU. The separate frozen CUDA packet remains queued for H20.
