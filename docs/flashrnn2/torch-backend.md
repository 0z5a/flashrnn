# Functional Torch backend

`from flashrnn import flashrnn_torch` provides full-state LSTM, sLSTM, GRU and
Elman recurrence with gradients attached to the caller's four tensors. The
implementation calls the existing differentiable reference. It adds no new
recurrence formula or independent Parameters.

Base revision: `09c0f39a359588c12bd948067f9d472191b5866f`. This implements the
CPU functional part of W0/P0/P1 in the Hopper/B200/PyTorch execution plan.
P0's clean installed CPU-wheel qualification remains pending. The existing
`flashrnn` and `FlashRNNConfig` entry points retain their defaults.

## Contract v1

Inputs are `wx [B,T,G,H,D]`, `R [G,H,D,D]`, `bias [G_bias,H,D]` and
`initial [S,B,1,H,D]`. Results are history `[S,B,T,H,D]` and carry
`[S,B,1,H,D]`. B/T/H/D must be positive; all inputs use one floating dtype
and device. Strided views are accepted. There is no implicit device transfer,
masking, padded-token handling or detach. Finite inputs are the qualified
domain; NaN/Inf behavior is not qualified.

| Cell | G / G_bias / S | State order | Time dependency |
|---|---|---|---|
| LSTM | 4 / 4 / 2 | h, c | Nonlinear; gates require previous h |
| sLSTM | 4 / 4 / 4 | h, c, n, m | Nonlinear; gates require previous h |
| GRU | 3 / 4 / 1 | h | Nonlinear; reset/update use previous h |
| Elman | 1 / 1 / 1 | h | Nonlinear tanh of the recurrent contraction |

GRU uses the existing reference gate order. sLSTM retains the global
`all(old_n == 0)` predicate across every batch/head/coordinate in a call;
the mixed-normalizer test checks continuation without localizing that rule.
These cells do not have precomputed affine transitions for a time scan.
RWKV6's input-conditioned matrix update is a separate ordered affine path;
its current-token bonus/readout and checkpointing remain a separate work
package, not an implementation in this backend.

| Policy | Inputs | Recurrent R / hidden | State, accumulation, public outputs |
|---|---|---|---|
| `mathematical` | FP16, BF16, FP32 or FP64, one dtype | Input dtype | Input dtype |
| `fp32_state_bf16_mma` | FP32 | R is cast BF16→FP32 once per call; h is cast BF16→FP32 each step | FP32 |

Ambient autocast is disabled inside this entry point. The BF16 policy uses
Torch cast derivatives, preserves the original R gradient link and does not
emulate Tensor Core reduction order. FP64 is the mathematical oracle. Public
outputs undergo no additional cast. Calling the next chunk with the returned
carry preserves the autograd graph. Full history is always returned; no
final-only memory reduction or dedicated backward kernel is claimed.

CUDA loaders and their configuration solver are imported only when selecting
the corresponding legacy CUDA backend. Triton/ninja are in the optional
`gpu` extra; `torch` and `einops` remain base requirements.

## Executed validation

The existing local Python 3.12.14 / Torch 2.13.0 environment was used with one
OMP/MKL thread, CPU tensors, CUDA build `None`, HIP build `None`. No packages
were installed or upgraded. This was an existing macOS environment, not a
fresh CPU-wheel installation. All three subprocesses naturally exited 0.

| Test command suffix | Methods passed | Coverage |
|---|---:|---|
| `test_torch_backend.py` | 8 / 8 | Four-cell public gradcheck; full/chunk gradients; mixed sLSTM state; views, aliases and readonly inputs; explicit BF16 cast policy; dtype/device validation; autocast; isolated import guards |
| `test_reference.py` | 5 / 5 | Unchanged upstream vanilla full-state/gradient comparison, zero and nonzero state, chunk continuation, initialization, layout ownership |
| `test_torch_layer.py` | 1 / 1 | Existing torch.nn LSTM/GRU/Elman full-input/parameter-gradient mapping |

Seed is 20261005. Existing FP64 forward and gradient budgets remain
`atol=rtol=1e-12` and `1e-11`, respectively. Public fast-mode gradcheck uses
`eps=1e-6, atol=1e-5, rtol=1e-4` on smooth FP64 fixtures. Chunk forward and
explicit same-cast policy comparisons are exact in the tested cases; chunk
gradients use `1e-11`. These do not change the GPU strict gate.

The fresh-process import guard rejects project CUDA loaders, Triton, ninja,
the extension builder, GPU property queries and external compilation while
calling both pure Torch APIs and backward. Explicit selection of each legacy
CUDA/Triton backend is separately checked to reach its deferred loader.
This verifies import routing, not GPU numerical compatibility.

Reproduce from the checkout with the existing environment:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests/flashrnn2 -p test_torch_backend.py -v
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests/flashrnn2 -p test_reference.py -v
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests/flashrnn2 -p test_torch_layer.py -v
```

[Controller, source hashes and logs](evidence/torch-backend-r2-manifest.json)
record the exact executed sources. The proposed CPU CI installs a CPU Torch
wheel and this package in an isolated environment, tests the installed package
outside the checkout, blocks GPU dependencies/toolkit and records wheel
origins. No successful execution of that CI is included in this report.

| Path | Functional status | Baseline time | Candidate time | Speedup |
|---|---|---:|---:|---:|
| Existing local CPU environment | 14 methods passed | — | — | Not measured |
| Fresh CPU-wheel installation | NOT_RUN; CI added | — | — | Not measured |
| H20 / SM90 | NOT_RUN | — | — | Not measured |
| H100 / B200 / SM120 dedicated paths | NOT_RUN for this change | — | — | Not measured |
| ROCm / XPU / MPS / compiled Torch | NOT_RUN | — | — | Not measured |

The new H20 node was inspected read-only; it is not H100 or B200 evidence.
Its environment and shared GPU window need qualification before execution.
No model E2E, concurrency or speed result is inferred from these unit tests.

## GPU continuation policy

W0 retains all recorded sLSTM and native CUDA failures. The unchanged strict
gate remains `atol=0.002, rtol=0.02, max_abs=0.003`; earlier failed rows remain
failed. Architecture runs must record actual GPU/backend, source and binary
hashes, all states, resources and completion/slot generation checks.

H0 starts with a single-CTA full recurrence and explicit packing on the actual
Hopper device. B0/TMEM stays NOT_RUN without SM100 hardware. Neither is
selected by this new eager entry point. Baseline/candidate performance must
include the same projection, output and autograd scope. Freeze at least 20
paired ABBA blocks across two new processes, report raw samples and paired
95% CIs, require a speed ratio above `max(1.05, 1 + 2*relative_variation)` and
CI lower bound above 1; protected latency-ratio CI upper bound must not exceed
1.03. One additional sampling round is allowed for an inconclusive result.
Full-model BS and request-concurrency campaigns remain required separately.
