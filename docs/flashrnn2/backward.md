# FlashRNN2 training path: R-tiled backward

Status: implemented, oracle-checked on CPU; GPU qualification batch submitted (job pending), timing against
FlashRNN1 prepared. Results are added below as they arrive.

## Contract

`flashrnn2.training.recurrence(wx, R, bias, initial, cell, output_mode=..., snapshot_dtype=..., config=...,
backward_options=...)` is a differentiable whole logical call for LSTM and sLSTM (FP16/BF16 inputs, any D that is a
multiple of 16, H >= 1). `flashrnn2.dispatch.recurrence` routes every call that requires gradients there. The
forward runs on the backend `config` (or the packaged table) selects, in full output mode with an FP32 state
snapshot that is saved for the backward; the requested output (full / hidden / last, input dtype or FP32) and the
FP32 final carry are returned, and gradients flow from both. Gradients are returned in the dtypes of the inputs.
Chunk continuation stays inference only: a differentiated call with a continuation context raises.

## Gradient convention

FlashRNN v1's convention, kept unchanged (`grad_reference.py`):

* the sLSTM stabilizer m carries no gradient (neither m_t nor m_{t-1} inside log f; the initial m gets zero);
* the normalizer clamp n_t = max(f n_{t-1} + i, 1) passes its gradient straight through.

h_t = sigmoid(o) c_t / n_t is invariant to the internal stabilizers while the clamp is inactive, and in exact
arithmetic the clamp is inactive whenever a logical call starts from a zero normalizer or from n_0 >= 1. Then the
convention is the true gradient of any loss on h, except for the initial m. Losses on c or n directly are covered
by the convention, not by this equivalence (c and n each carry a factor exp(-m)).

CPU checks (`tools/bw_oracle_check.py`, float64, three seeds per fixture), all PASS:

| Check | LSTM | sLSTM zero start | sLSTM n0 >= 1 | sLSTM 0 < n0 < 1 |
| --- | --- | --- | --- | --- |
| oracle forward == `reference.recurrence` | bitwise | bitwise | bitwise | bitwise |
| oracle gradients vs FlashRNN v1 vanilla backward | 0.0 | 0.0 | 0.0 | 0.0 |
| `gradcheck` against finite differences | all outputs | h | h | h (reported, not gated) |

## Kernel

`triton_rtiled_bw.backward` mirrors the forward R tiling. The sequential part computes only the gate
preactivation gradients dG_t and the state gradients carried from t to t-1:

* CTA s of a (batch tile, head) group owns hidden units p in [sS, (s+1)S); it holds the column slice R[:, :, :, p]
  (every gate, every output row) and the c/n carries of its units.
* Step t = T-1 ... 0: dh_t of its units = dH_t + sum_g dG_{t+1,g} R_g[:, p], a GEMM over the full, rounded dG_{t+1}
  of every CTA of the group (exchanged through a double-buffered L2 buffer under the forward's counter barrier);
  the pointwise cell backward uses the saved FP32 states and the gate preactivations; dG_t is stored in FP32 and
  published rounded to the input dtype.
* After step 0 the same GEMM gives the gradient of the initial h.

Preactivations are recomputed in one FP32 GEMM (TF32 off) from the saved history with the hidden operand rounded as
in the forward MMA, so only the summation order differs from the forward kernel; the saved m and n are used, not
recomputed. dR and dbias are FP32 einsums/reductions over the stored dG (no atomics, deterministic). D <= 128 runs
the same kernel with one CTA per group. `sync="launch"` runs one launch per step and takes the GEMM operand from
the stored dG instead of the exchange buffer: the same values and arithmetic, the bitwise oracle of the persistent
kernel.

Login-node compile sweep (sm_90a, 8 warps): every planned (cell, dtype, D, S) fits shared memory, at most 260
bytes of spill; S = 64 at D = 768 needs 426 KB and is not offered.

## Qualification (pre-registered)

`tools/bw_qualify.py`, 168 cases (LSTM warm; sLSTM zero and warm starts; FP16/BF16; D = 64 ... 768; two slice
widths; B = 3 / T = 17 and B = 32 / T = 64), plus compute-sanitizer memcheck and synccheck:

* gradients of two losses (<dH, h history> and that plus <dF, final carry>) against the FP64 oracle on the same
  rounded inputs; per tensor relative Frobenius error <= 1e-2 (FP16) / 5e-2 (BF16), max-abs error <= 5x that
  times max|ref|, and where FlashRNN1 cuda_fused runs the same case: no more than 1.25x its error (floor: budget/10);
* persistent == launch-per-step, bitwise, every gradient; barrier counts and error flag;
* 20 complete training steps bitwise identical.

Results: pending.
