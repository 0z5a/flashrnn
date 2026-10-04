# Native baseline qualification transports

The existing environment lacks Ninja and einops. These task-local runners
use the installed Torch/Triton runtimes and leave the environment unchanged.
Their source, layout and arithmetic boundaries are explicit; an adapter is
not silently reported as the stock public entry point.

| Path | Executed evidence | Remaining qualification |
| --- | --- | --- |
| GPU-info CUDA 13 build/import | Successful in the earlier build report | Device query |
| Original alternating CUDA, seven translation units | Source/config/flags prepared | Compile/import, then forward and all four gradients |
| Original Triton LSTM/sLSTM kernels | All five finite GPU cases executed: four PASS, one retained sLSTM zero-state failure | Failure localization, broader numerical calibration, stock einops-wrapper parity |
| Alternating LSTM cast reference | FP32 CPU T1/T17/T128 output and all-input gradient parity | BF16 native GPU error characterization |
| Explicit per-element sLSTM reference option | Nine CPU tests pass, including mixed-normalizer counterexample | Native GPU comparison |

`upstream_triton.py` selects the unchanged upstream kernel, helper and
one-configuration autotuner definitions after checking both complete source
hashes. Only the wrapper's padding and layout transforms use Torch instead
of einops. It exposes forward only. The completed five-case gate includes
nonzero/zero LSTM states and nonzero/zero/mixed sLSTM states, at B3/T17/H2/D64.
It retains the predeclared 0.002 absolute / 0.02 relative check plus 0.003
maximum absolute error. This is exploratory qualification, not the full
independently calibrated numerical budget.

On RTX 5090 GPU1, Torch 2.12.1+cu130 and Triton 3.7.1, controller 13085 and
child 13088 naturally exit 1 after all five cases. All public inputs remain
unchanged. The original source hashes and measured adapter/reference hashes
match the archived bundle. The processes are gone, GPU1 memory is zero and
the original GPU1 lock was independently acquired/released before handoff.

| Cell / initialization | History max error | Final max error | Result |
| --- | ---: | ---: | --- |
| LSTM / nonzero | 3.81e-6 | 3.81e-6 | PASS |
| LSTM / zero | 0.00012207 | 0 | PASS |
| sLSTM / nonzero | 0.00097656 | 0.00048828 | PASS |
| sLSTM / zero | **0.0078125** | 5.96e-8 | **CORRECTNESS_FAILED** |
| sLSTM / mixed | 0.00097656 | 0 | PASS |

The failure exceeds the original 0.003 maximum-error limit. It is retained
without changing the budget. Specific state/timestep localization and FP32
snapshot diagnostics are pending; no cause is inferred from the maximum
alone. This failed exploratory case does not qualify the whole upstream
baseline for timing or revise the earlier candidate's failure.

`qualify_baseline_build.py --target cuda` builds the original alternating
seven-file source list and flags with the already qualified distutils
transport. It uses homogeneous BF16 or FP32 dtypes. The prepared CUDA gate
executes T1/T17/T128 at B16/H1/D64, checks forward outputs and input ownership,
and records dWx/dR/db/dinitial errors for final-only and history-plus-final
losses. Gradient finiteness is a diagnostic; it is not gradient accuracy
acceptance. Neither this build nor this GPU gate has executed yet.

The cast reference matches the alternating backend's stored-state and
recurrent-product rounding boundaries. CUDA fast-math activations are not
emulated. Its FP32 comparison to the mathematical reference produced maximum
forward error 2.24e-8 and maximum gradient error 1.91e-6 across the three CPU
sequence lengths. These results do not qualify the CUDA implementation.

The compiler transport can reuse a completed artifact only after checking
its saved SHA-256. This reuse path is prepared but not exercised yet.
See [numerical contracts](numerical-contracts.md) for the differences between
alternating CUDA, fused CUDA, Triton and the candidate kernels.
