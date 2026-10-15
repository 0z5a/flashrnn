# FlashRNN2 training path: R-tiled backward

Status: implemented and qualified on H100 (168/168 cases); training step timed against FlashRNN1 (faster at B = 64, slower at B = 16).

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

Arithmetic check without a GPU (`tools/bw_interpret_check.py`, Triton interpreter, launch-per-step mode): 12 cases
(LSTM warm; sLSTM zero and warm; k = 1, 2 and 3 CTAs per group; ragged batch tile; two heads; D = 48 with two K
parts) against the FP64 oracle on the same FP16 inputs: worst relative Frobenius error 1.7e-4, all PASS.

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

Results (job 47323663, 30.9 min, `ALL_CASES_PASS`, archive verified, 555 files):

* 168/168 cases PASS every gate: gradients against the FP64 oracle, persistent == launch-per-step bitwise, 20
  repeated training steps bitwise identical; compute-sanitizer memcheck and synccheck 0 errors (8/8).
* Worst relative Frobenius error per tensor: FP16 2.1e-4 - 3.2e-4, BF16 1.7e-3 - 2.8e-3 (budgets 1e-2 / 5e-2), flat
  across D = 64 ... 768.
* Against FlashRNN1 cuda_fused on the same inputs and loss: our error is 0.20x - 0.40x of FlashRNN1's (the FP32
  state history and FP32 dG keep the rounding to the MMA operands). FlashRNN1 could not be loaded in 2 of 168
  calibration cases (two lanes building the same module at once; serialized since then with a node-wide lock).

## Training step against FlashRNN1 (job 47334053)

One call = forward with gradients + loss <dH, h history> + backward (`torch.autograd.grad`); FP16, T = 256, H = 1.
FlashRNN2: forward on a Triton backend (default_config, or for D > 128 the M0-qualified R-tiled S32/4 warps), backward
slice width and warp count selected on split 0; FlashRNN1: pinned cuda_fused forward + backward. Every candidate of
both arms was gradient-qualified against the FP64 oracle before selection (192/192 FlashRNN2 candidates, 28/28
FlashRNN1). Split 1: 5 process starts x 20 ABBA/BAAB blocks, hierarchical bootstrap. 39 min, 140/140 starts,
archive verified (643 files). A TileLang forward is not in the step: it needs the CUDA 12.8 toolchain, the pinned
FlashRNN1 extension the module's CUDA 11.8, and both arms share one process.

| cell | D | B | FlashRNN1 ms | FlashRNN2 ms | FlashRNN1 / FlashRNN2 [95% CI] | FlashRNN2 forward | backward S / warps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | 64 | 16 | 1.667 | 1.796 | 0.956 [0.936, 0.982] | default | 32 / 8 |
| lstm | 64 | 64 | 2.547 | 2.201 | 1.190 [1.165, 1.220] | default | 32 / 8 |
| lstm | 128 | 16 | 2.033 | 2.439 | 0.861 [0.841, 0.886] | default | 32 / 4 |
| lstm | 128 | 64 | 3.072 | 2.973 | 1.022 [1.010, 1.031] | default | 32 / 8 |
| lstm | 192 | 16 | 2.031 | 2.768 | 0.759 [0.740, 0.784] | R-tiled S32/4w | 16 / 4 |
| lstm | 192 | 64 | 4.468 | 3.707 | 1.180 [1.157, 1.200] | R-tiled S32/4w | 16 / 4 |
| lstm | 256 | 16 | 2.226 | 2.812 | 0.815 [0.795, 0.841] | R-tiled S32/4w | 32 / 4 |
| lstm | 256 | 64 | 4.869 | 3.827 | 1.243 [1.215, 1.266] | R-tiled S32/4w | 16 / 4 |
| lstm | 384 | 16 | 2.490 | 3.440 | 0.712 [0.703, 0.721] | R-tiled S32/4w | 32 / 4 |
| lstm | 384 | 64 | 6.666 | 5.084 | 1.307 [1.294, 1.313] | R-tiled S32/4w | 16 / 4 |
| lstm | 512 | 16 | 2.819 | 3.850 | 0.711 [0.691, 0.728] | R-tiled S32/4w | 32 / 4 |
| lstm | 512 | 64 | 7.305 | 6.177 | 1.182 [1.182, 1.183] | R-tiled S32/4w | 32 / 4 |
| lstm | 768 | 16 | 3.536 | 5.040 | 0.682 [0.663, 0.697] | R-tiled S32/4w | 32 / 4 |
| lstm | 768 | 64 | 12.450 | 9.075 | 1.371 [1.370, 1.372] | R-tiled S32/4w | 32 / 4 |
| slstm | 64 | 16 | 1.754 | 2.305 | 0.750 [0.738, 0.761] | default | 32 / 8 |
| slstm | 64 | 64 | 2.708 | 2.759 | 1.007 [0.987, 1.033] | default | 32 / 8 |
| slstm | 128 | 16 | 2.156 | 2.931 | 0.716 [0.698, 0.730] | default | 32 / 8 |
| slstm | 128 | 64 | 3.323 | 3.483 | 0.980 [0.960, 1.006] | default | 32 / 8 |
| slstm | 192 | 16 | 2.133 | 3.165 | 0.673 [0.672, 0.674] | R-tiled S32/4w | 16 / 4 |
| slstm | 192 | 64 | 4.715 | 4.122 | 1.144 [1.143, 1.145] | R-tiled S32/4w | 16 / 4 |
| slstm | 256 | 16 | 2.231 | 3.278 | 0.680 [0.678, 0.681] | R-tiled S32/4w | 32 / 4 |
| slstm | 256 | 64 | 5.487 | 4.383 | 1.253 [1.252, 1.255] | R-tiled S32/4w | 16 / 4 |
| slstm | 384 | 16 | 2.640 | 4.042 | 0.652 [0.652, 0.653] | R-tiled S32/4w | 16 / 4 |
| slstm | 384 | 64 | 6.547 | 5.687 | 1.150 [1.150, 1.151] | R-tiled S32/4w | 16 / 4 |
| slstm | 512 | 16 | 2.926 | 4.522 | 0.647 [0.646, 0.648] | R-tiled S32/4w | 32 / 4 |
| slstm | 512 | 64 | 7.514 | 6.848 | 1.097 [1.096, 1.098] | R-tiled S32/4w | 32 / 4 |
| slstm | 768 | 16 | 3.700 | 5.567 | 0.665 [0.664, 0.665] | R-tiled S32/4w | 32 / 4 |
| slstm | 768 | 64 | 12.623 | 9.639 | 1.310 [1.308, 1.310] | R-tiled S32/4w | 32 / 4 |

* B = 64: FlashRNN2 is faster for every D >= 192 (LSTM 1.18-1.37x, sLSTM 1.10-1.31x) and for LSTM D = 64/128
  (1.19x / 1.02x); sLSTM D = 64/128 is a tie (intervals cover 1).
* B = 16: FlashRNN2 is slower everywhere (LSTM 0.68-0.96x, sLSTM 0.65-0.75x). With one batch tile per call the step
  is a latency chain; FlashRNN2's step carries the FP32 history write in the forward, the separate preactivation and
  dR GEMMs and a per-step exchange of 4D values (the backward GEMM reduces over all four gates). These are the
  targets for the next round (profile at B = 16 first).

## B = 16 fix (breakdown job 47337798; confirmation job 47347074)

**Where the time went** (breakdown, timing only, LSTM/sLSTM x D 192/768 x B 16/64 x T 64/256; the T slope gives the
per-step cost): the sequential chains were not the problem -- the backward kernel costs about 8.8 us per step against
about 8.4 us for FlashRNN1's whole backward, and a TileLang forward about 3.4 us per step against FlashRNN1's 4.8. At
B = 16 the gap came from work proportional to T outside the kernels: the preactivation GEMM (0.25-0.7 ms), the dR GEMM
(0.15-0.5 ms) and about 0.7 ms of copies and allocation. The pinned FlashRNN1 extension also could not be imported in
the TileLang environment (its loader saw nvcc 12.8, changed its build key and tried to rebuild against torch's CUDA
11.8), so a TileLang forward could not be timed against it.

**Changes**
* fast backward variant (`fast=True`): the step's c/n/m are the next step's c_{t-1}/n_{t-1}/m_{t-1} (no reload), the
  next step's pointwise inputs are loaded before the barrier, history-gradient components that are zero are not read;
  bitwise identical to the qualified kernel (`FAST=0` compiles to the qualified PTX);
* single-head preactivation and dR GEMMs as plain `addmm`/`mm` straight into the [B, T, 4, 1, D] layout (no einsum
  transposes); optional TF32 for both (`tf32=True`; FP16/BF16 operands of the preactivation GEMM are exact in TF32);
* FlashRNN1 loads its module-built artifact in the TileLang environment (`FRNN_MODULE_CUDA_HOME`).

**Requalification** (64/64 PASS): fast variant bitwise equal to the base kernel, persistent == launch, 20 repeats, and
the gradient gate (FP16 worst 3.1e-4, BF16 2.7e-3); fast + TF32 gradient gate with FlashRNN1 calibration (FP16
3.2e-4, BF16 2.4e-3; at most 0.35x FlashRNN1's error). Race verification of the fast backward: correct kernels clean
in all runs (D 256/768, LSTM and sLSTM), single-buffer and skipped-barrier controls caught; the arrive-before-store
control was caught on LSTM but only in 9/21 and 1/21 sLSTM runs (no output ever changed, no silent corruption).

**Pre-registered criterion** (written into the plan before submission): B = 16 counts as fixed if, in at least one
environment, the selected training step's lower 95% bound against FlashRNN1 is >= 1.0 for every D >= 192 and both
cells; B = 64 must not regress. **Result: B = 16 is not fixed by that criterion; B = 64 does not regress.**

* B = 16, best environment (TileLang forward + fast backward + TF32): LSTM 0.875-1.058 for D >= 192 (D 256 faster,
  D 192/768 within 2-3%, D 384/512 9-13% behind); sLSTM 0.882-0.980; D = 128 is now faster in both cells (1.07x /
  1.06x). Before: LSTM 0.68-0.96, sLSTM 0.65-0.75.
* B = 64: FlashRNN2 faster everywhere, LSTM 1.24-1.91x, sLSTM 1.19-1.90x (before 0.98-1.37x).
* The remaining B = 16 gap sits at D = 384/512, where the step is still dominated by the T-proportional GEMMs and the
  saved FP32 history; fusing the preactivation recompute into the backward kernel would remove the largest part.

Defaults now: `fast=True`, `tf32=True`, eight warps, backward slice 32 (D <= 128), 16 (D <= 512), 32 (D = 768). The
timing selected four warps in most cases; four-warp configurations are gradient-qualified at the timed shapes but
were not in the 168-case qualification, so they are not the default.

| env | cell | D | B | before (job 47334053) | after: FlashRNN1 / FlashRNN2 [95% CI] | FlashRNN2 ms | forward | backward S / warps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| module | lstm | 64 | 16 | 0.956 | 1.091 [1.069, 1.120] | 1.620 | Triton default | 32 / 8 |
| module | lstm | 128 | 16 | 0.861 | 0.978 [0.958, 1.003] | 2.173 | Triton default | 32 / 8 |
| module | lstm | 192 | 16 | 0.759 | 0.821 [0.801, 0.846] | 2.536 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 256 | 16 | 0.815 | 0.913 [0.892, 0.940] | 2.498 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 384 | 16 | 0.712 | 0.844 [0.825, 0.869] | 2.997 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 512 | 16 | 0.711 | 0.801 [0.786, 0.815] | 3.366 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 768 | 16 | 0.682 | 0.822 [0.803, 0.838] | 4.225 | Triton R-tiled S32/4w | 16 / 4 |
| module | slstm | 64 | 16 | 0.750 | 0.939 [0.919, 0.964] | 2.010 | Triton default | 64 / 8 |
| module | slstm | 128 | 16 | 0.716 | 0.905 [0.886, 0.930] | 2.513 | Triton default | 32 / 8 |
| module | slstm | 192 | 16 | 0.673 | 0.812 [0.811, 0.813] | 2.669 | Triton R-tiled S32/4w | 32 / 8 |
| module | slstm | 256 | 16 | 0.680 | 0.808 [0.807, 0.810] | 2.809 | Triton R-tiled S32/4w | 16 / 4 |
| module | slstm | 384 | 16 | 0.652 | 0.779 [0.778, 0.780] | 3.445 | Triton R-tiled S32/4w | 16 / 4 |
| module | slstm | 512 | 16 | 0.647 | 0.783 [0.782, 0.784] | 3.758 | Triton R-tiled S32/4w | 32 / 4 |
| module | slstm | 768 | 16 | 0.665 | 0.834 [0.834, 0.835] | 4.537 | Triton R-tiled S32/4w | 32 / 4 |
| module | lstm | 64 | 64 | 1.190 | 1.385 [1.357, 1.420] | 1.895 | Triton default | 32 / 8 |
| module | lstm | 128 | 64 | 1.022 | 1.238 [1.215, 1.268] | 2.537 | Triton default | 32 / 8 |
| module | lstm | 192 | 64 | 1.180 | 1.370 [1.333, 1.400] | 3.156 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 256 | 64 | 1.243 | 1.540 [1.510, 1.579] | 3.222 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 384 | 64 | 1.307 | 1.639 [1.637, 1.640] | 4.058 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 512 | 64 | 1.182 | 1.526 [1.525, 1.527] | 4.734 | Triton R-tiled S32/4w | 16 / 4 |
| module | lstm | 768 | 64 | 1.371 | 1.912 [1.910, 1.913] | 6.429 | Triton R-tiled S32/4w | 32 / 4 |
| module | slstm | 64 | 64 | 1.007 | 1.275 [1.249, 1.308] | 2.206 | Triton default | 32 / 8 |
| module | slstm | 128 | 64 | 0.980 | 1.185 [1.161, 1.214] | 2.891 | Triton default | 32 / 8 |
| module | slstm | 192 | 64 | 1.144 | 1.423 [1.422, 1.424] | 3.321 | Triton R-tiled S32/4w | 32 / 8 |
| module | slstm | 256 | 64 | 1.253 | 1.556 [1.554, 1.558] | 3.529 | Triton R-tiled S32/4w | 16 / 4 |
| module | slstm | 384 | 64 | 1.150 | 1.488 [1.487, 1.490] | 4.458 | Triton R-tiled S32/4w | 32 / 4 |
| module | slstm | 512 | 64 | 1.097 | 1.501 [1.498, 1.503] | 5.078 | Triton R-tiled S32/4w | 32 / 4 |
| module | slstm | 768 | 64 | 1.310 | 1.897 [1.895, 1.898] | 6.740 | Triton R-tiled S32/4w | 32 / 4 |
| TileLang | lstm | 128 | 16 | 0.861 | 1.072 [1.071, 1.074] | 1.930 | TileLang cluster S32 | 32 / 8 |
| TileLang | lstm | 192 | 16 | 0.759 | 0.981 [0.980, 0.982] | 2.094 | TileLang cluster S16 | 16 / 4 |
| TileLang | lstm | 256 | 16 | 0.815 | 1.058 [1.055, 1.062] | 2.094 | TileLang cluster S16 | 16 / 4 |
| TileLang | lstm | 384 | 16 | 0.712 | 0.900 [0.899, 0.901] | 2.833 | TileLang cluster S32 | 16 / 4 |
| TileLang | lstm | 512 | 16 | 0.711 | 0.875 [0.874, 0.876] | 3.149 | TileLang cluster S32 | 16 / 4 |
| TileLang | lstm | 768 | 16 | 0.682 | 0.968 [0.967, 0.969] | 3.760 | TileLang grid S16 | 32 / 4 |
| TileLang | slstm | 128 | 16 | 0.716 | 1.056 [1.055, 1.058] | 2.015 | TileLang cluster S32 | 32 / 4 |
| TileLang | slstm | 192 | 16 | 0.673 | 0.969 [0.967, 0.970] | 2.186 | TileLang cluster S16 | 32 / 8 |
| TileLang | slstm | 256 | 16 | 0.680 | 0.980 [0.979, 0.981] | 2.249 | TileLang cluster S16 | 32 / 4 |
| TileLang | slstm | 384 | 16 | 0.652 | 0.882 [0.881, 0.883] | 2.933 | TileLang cluster S32 | 16 / 4 |
| TileLang | slstm | 512 | 16 | 0.647 | 0.892 [0.891, 0.892] | 3.231 | TileLang cluster S32 | 32 / 4 |
| TileLang | slstm | 768 | 16 | 0.665 | 0.975 [0.974, 0.976] | 3.806 | TileLang grid S16 | 32 / 4 |
