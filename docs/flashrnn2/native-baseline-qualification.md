# Native baseline qualification transports

The existing environment lacks Ninja and einops. These task-local runners
use the installed Torch/Triton runtimes and leave the environment unchanged.
Their source, layout and arithmetic boundaries are explicit; an adapter is
not silently reported as the stock public entry point.

| Path | Executed evidence | Remaining qualification |
| --- | --- | --- |
| GPU-info CUDA 13 build/import | Successful in the earlier build report | Device query |
| Original alternating CUDA, seven translation units | Source/config/flags prepared | Compile/import, then forward and all four gradients |
| Original Triton LSTM/sLSTM kernels | Hash-pinned AST and Torch layout adapter prepared | Five finite GPU cases; stock einops-wrapper parity |
| Alternating LSTM cast reference | FP32 CPU T1/T17/T128 output and all-input gradient parity | BF16 native GPU error characterization |
| Explicit per-element sLSTM reference option | Nine CPU tests pass, including mixed-normalizer counterexample | Native GPU comparison |

`upstream_triton.py` selects the unchanged upstream kernel, helper and
one-configuration autotuner definitions after checking both complete source
hashes. Only the wrapper's padding and layout transforms use Torch instead
of einops. It exposes forward only. The queued five-case gate includes
nonzero/zero LSTM states and nonzero/zero/mixed sLSTM states, at B3/T17/H2/D64.
It retains the predeclared 0.002 absolute / 0.02 relative check plus 0.003
maximum absolute error. This is exploratory qualification, not the full
independently calibrated numerical budget.

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
