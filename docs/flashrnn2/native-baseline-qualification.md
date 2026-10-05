# Native baseline qualification transports

The existing environment lacks Ninja and einops. These task-local runners
use the installed Torch/Triton runtimes and leave the environment unchanged.
Their source, layout and arithmetic boundaries are explicit; an adapter is
not silently reported as the stock public entry point.

| Path | Executed evidence | Remaining qualification |
| --- | --- | --- |
| GPU-info CUDA 13 build/import | Successful in the earlier build report | Device query |
| Original alternating CUDA, seven translation units | Homogeneous BF16 CUDA13 build/import PASS with upstream compute80 PTX flags; earlier missing-import failure retained | GPU forward and all four gradient comparisons |
| Original Triton LSTM/sLSTM kernels | All five finite GPU cases executed: four PASS, one retained sLSTM zero-state failure; exact failing element localized | FP32 internal-value diagnosis, broader numerical calibration, stock einops-wrapper parity |
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

The failure exceeds the original 0.003 maximum-error limit and remains
failed. A separately admitted one-case diagnostic on GPU0 reproduces exactly
one over-budget coordinate, zero-based `[S=2,B=2,T=5,H=0,D=59]`:

| State | Native BF16 | Reference FP32 | Reference BF16 | Error |
| --- | ---: | ---: | ---: | ---: |
| normalizer n | 1.8828125 | 1.87890625 | 1.875 | 0.0078125 |

The reference is exactly the midpoint between the two adjacent BF16 values.
The native internal FP32 value is not captured, so this does not identify
which arithmetic operation produces the difference. Setting the original
kernel's `DTYPE=float32` would also change its recurrent MMA operand cast;
that is not a snapshot-only diagnostic and has not been done.

The diagnostic retains the unchanged original kernel and threshold. Its
controller naturally exits 1, with a verified small tensor snapshot and
resource-return marker. The h/c/m maximum errors are 9.54e-7/6.10e-5/6.10e-5.
This case does not qualify the whole upstream baseline for timing or revise
the earlier candidate's failure.

`qualify_baseline_build.py --target cuda` builds the original alternating
seven-file source list and flags with the already qualified distutils
transport. It uses homogeneous BF16 or FP32 dtypes. The prepared CUDA gate
executes T1/T17/T128 at B16/H1/D64, checks forward outputs and input ownership,
and records dWx/dR/db/dinitial errors for final-only and history-plus-final
losses. Gradient finiteness is a diagnostic; it is not gradient accuracy
acceptance. The r4 invocation naturally exits 1 at the first project import
(`ModuleNotFoundError: flashrnn`), before any of the seven files compile.
R5 adds the task source root to process-local `PYTHONPATH`, keeps CUDA hidden
and uses the same frozen source archive and compiler flags. It successfully
builds and imports all seven translation units in 38.26 seconds. All seven
object files, the shared library and manifest have timestamps inside this
run's actual start/end interval, so this is a fresh build. The final library
SHA256 is `b2d55255b1a2079f91bf0ce824319d0b3ef0aab30a00060fc7fac9e1e49e05b3`.

The resolved B16/H1/D64 LSTM configuration uses BF16 for all storage/operand
types, including A and S; recurrent and forward clipping are disabled.
The process supplies `TORCH_CUDA_ARCH_LIST=12.0`, but the unchanged upstream
compiler flags explicitly request `arch=compute_80,code=compute_80`. The
environment variable alone does not prove a native SM120 cubin; binary ISA
inspection and actual RTX 5090 execution remain pending.
The source/header hashes and full flags are in the
[build manifest](evidence/baseline-build-r5-manifest.json). Its broad header
inventory also records transported AppleDouble sidecars; they are not
translation units or explicitly included headers. All seven compiled source
files and ordinary headers match the local checkout.

Both controller and SSH naturally exit 0. No packages change; neither the
compiled GPU-info query nor recurrence is called. The original IO window is returned after
off-host archive verification. See [r4 failure](evidence/baseline-build-r4.log),
[r5 result](evidence/baseline-build-r5.log),
[fresh-build audit](evidence/baseline-build-r5-freshness.json) and
[return receipt](evidence/baseline-build-r5-complete.json). The recurrence
GPU gate and gradient acceptance remain pending.

The cast reference matches the alternating backend's stored-state and
recurrent-product rounding boundaries. CUDA fast-math activations are not
emulated. Its FP32 comparison to the mathematical reference produced maximum
forward error 2.24e-8 and maximum gradient error 1.91e-6 across the three CPU
sequence lengths. These results do not qualify the CUDA implementation.

The compiler transport can reuse a completed artifact only after checking
its saved SHA-256. This reuse path is prepared but not exercised yet.
See [numerical contracts](numerical-contracts.md) for the differences between
alternating CUDA, fused CUDA, Triton and the candidate kernels.
