# Precision modes: accuracy and speed (advisor item 5)

This page is compiled from existing evidence. No new runs were made for it.

## What FlashRNN2 computes in which precision

| quantity | storage / operand | arithmetic |
| --- | --- | --- |
| inputs Wx, R, bias | input dtype (FP16 or BF16) | — |
| recurrent operand h_{t-1} | rounded to the input dtype (MMA operand) | FP32 accumulation |
| cell states c, n, m, forward history | FP32 | FP32 |
| exchanged dG (backward) | rounded to the input dtype (MMA operand) | FP32 accumulation |
| dG returned for Wx | input dtype | — |
| dR | FP32 GEMM, or input-dtype operands with FP32 accumulation (H > 1 default, heads §8.6) | returned in the input dtype |
| dbias, d_initial | FP32 sums | returned in the input dtype |

FP32 inputs (an "FP32/TF32 mode") are not supported by the kernels; every kernel takes FP16 or BF16 operands. In the
places where TF32 was tried (the preactivation GEMM at large D, section 25 of max-width.md), the operands are already
input-dtype values, so TF32 is exact there and changes only the summation order. There it made no measurable
difference.

## Accuracy against FP64

The largest relative Frobenius error of any training gradient (dWx, dR, dbias, d_initial h / c), per qualification
case, against an FP64 reference of the same rounded inputs (B = 16-100, T = 24). These are the 2-D backward
qualifications of 2026-10-18 at H = 1 and the H > 1 qualifications of heads §8.6:

| dtype | heads | D | cases | worst rel. Frobenius error |
| --- | --- | --- | --- | --- |
| FP16 | 1 | 64-2048 | 62 | 3.7e-4 to 3.8e-4 |
| FP16 | > 1 | 64, 128 | 12 | 3.8e-4 to 3.9e-4 |
| BF16 | 1 | 64-2048 | 62 | 2.8e-3 to 3.1e-3 |
| BF16 | > 1 | 64, 128 | 12 | 3.0e-3 to 3.1e-3 |

The error does not grow with D or H. It sits at roughly 0.75 of the unit roundoff of the output dtype: FP16 2^-11 ≈
4.9e-4, BF16 2^-8 ≈ 3.9e-3. So the returned gradients are limited by their own storage precision, and the FP32
accumulation inside the kernels contributes far less.

## Speed

On these kernels FP16 and BF16 run at the same speed. Training step medians, B = 64, T = 256 (batch "tune", 2-D
backward):

| cell | D | FP16 | BF16 |
| --- | --- | --- | --- |
| LSTM | 1024 | 8.95 ms | 8.92 ms |
| LSTM | 1536 | 13.61 | 13.60 |
| LSTM | 2048 | 21.50 | 21.55 |
| sLSTM | 1024 | 9.20 | 9.18 |
| sLSTM | 1536 | 14.18 | 14.20 |
| sLSTM | 2048 | 22.33 | 22.32 |

The choice between FP16 and BF16 is therefore one of range against precision (BF16: FP32's exponent range, about 8x
the gradient error), not one of speed.
