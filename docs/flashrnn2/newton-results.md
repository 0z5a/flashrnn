# Dense Newton exploration — CPU mathematical qualification

This independent extension preserves the original LSTM/sLSTM/GRU/Elman
cells and dense recurrent matrices. It solves a known complete input
sequence by Newton linearization followed by an inclusive affine scan.
It is not usable as a parallel solution of unknown autoregressive tokens.

For an iterate z, form J_t = df_t/dz_(t-1) and
q_t = f_t(z_(t-1)) - J_t z_(t-1). Solve
y_t = J_t y_(t-1) + q_t, including the fixed public initial state in the
first offset. Composition uses noncommuting dense matrices in time order.
The scan has ceil(log2(T)) levels; each level performs dense matrix products.
There is no cross-block GPU waiting protocol in this Torch prototype.

The method follows the established Newton/time-parallel approach described
by [DEER](https://arxiv.org/abs/2309.12252) and
[ParaRNN](https://github.com/apple-aiml-research/ml-pararnn/tree/513d75dade40843bb27ac4a521969997188b5459).
No new algorithmic novelty is claimed. Neither DEER's implementation nor
the full ParaRNN package/accelerated modes has been qualified here.

## Executed cases

Existing CPU Torch 2.13.0, FP64, seed 20261008. Eight B2/T17/H2/D4 cases
cover final-only and history-plus-final losses and every dWx/dR/db/dinitial.
Four additional B2/T64/H2/D8 forward cases pass. The iteration limit is 16,
residual limit 1e-10, forward atol/rtol 1e-8/1e-7, and gradient atol/rtol
1e-7/1e-6, all fixed before execution. Differentiation is through the executed
Newton iterations and Jacobians; the stopping decision is detached. This is
not a separately implemented implicit backward or a CUDA training result.

| Cell | Normal iterations | Max output error | Max gradient error | One-iteration negative control |
| --- | ---: | ---: | ---: | --- |
| lstm | 3 | 1.11e-16 | 5.33e-15 | residual 0.0111; CAP_EXHAUSTED |
| slstm | 4 | 5.03e-14 | 1.13e-13 | residual 0.662; CAP_EXHAUSTED |
| gru | 3 | 2.44e-15 | 5.1e-15 | residual 0.0169; CAP_EXHAUSTED |
| elman | 2 | 4.38e-11 | 1.12e-9 | residual 150; CAP_EXHAUSTED |

All four negative controls remain failed convergence cases, not accepted
approximations. The controller naturally exits 0 because the normal cases
pass and the controls explicitly report their unmet iteration budget.
Separate sLSTM zero/mixed-normalizer cases at seed 20261009 pass both losses
and all gradients, taking four/five iterations respectively. The mixed
condition uses the public vanilla global predicate across all batch/head
coordinates, preserving the mathematical reference semantics.

The unchanged dense reduction component from ParaRNN commit
`513d75dade40843bb27ac4a521969997188b5459` was also executed inside the full
Newton forward solve. All four B2/T64/H2/D8 cases pass at seed 20261010.
The adapter checks the complete source hash before selecting only
`parallel_reduce` and `_reduction_step_dense`; it does not import the stock
entry point, which requires its compiled extension. ParaRNN component
gradients and GPU execution are not tested. Its built-in fixed-iteration
Newton routine is not used; the prototype retains its own residual check.

## Cost and speed scope

For S states per coordinate, materializing one dense Jacobian requires
B * H * T * (S * D)^2 elements. Scan intermediates, autograd history,
activations and weights require additional memory. These are calculated
storage requirements, not measured peak allocations:

| FP32 planned shape | LSTM Jacobian | sLSTM Jacobian |
| --- | ---: | ---: |
| B16/T1024/H1/D256 (P2) | 16 GiB | 64 GiB |
| B16/T1024/H1/D512 (P3) | 64 GiB | 256 GiB |

Thus this direct dense construction cannot simply be scaled to all planned
shapes on a 32 GiB GPU. The explored cells remain dense; substituting a
diagonal cell would change the experiment. Structured factorization,
checkpointing/chunking, total compute, GPU performance, independent native
baselines and full-model quality remain to be explored.

| Scope | Sequential baseline | Newton candidate | Speedup |
| --- | --- | --- | --- |
| CPU mathematical qualification | Executed | Executed | Not measured |
| GPU complete recurrence | Pending | Pending | N/A |
| Complete training / model E2E | Pending | Pending | N/A |

Raw results and source hashes are in `evidence/newton-*` and
`evidence/pararnn-dense-*`. `newton-r1-source.patch` restores the exact first
measured source from the current source; the only later implementation
change is injection of the separately qualified reduction function.
Reproduce the normal/negative-control campaign with
`python tools/flashrnn2/newton_gate.py --output newton.jsonl`.
The ParaRNN component runner takes `--source` pointing to the pinned
`pararnn/parallel_reduction/parallel_reduction.py` and `--output` for JSONL.
