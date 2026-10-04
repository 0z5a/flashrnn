# Resolved numerical contracts

Source inspection uses upstream `0b84fc30120ecccbde97976bc5b086f3215572d2`.
The CUDA 13 GPU-info patch changes device metadata queries only.

| Path | State between steps | Pointwise arithmetic | sLSTM zero-normalizer condition | Gate clamp |
| --- | --- | --- | --- | --- |
| Public vanilla / FP64 reference | Input dtype | Input dtype | All elements are zero | No |
| Reference used by r2/r4 | FP32; recurrent hidden explicitly rounded to BF16 | FP32 | All elements are zero | No |
| New step / inferred persistent / Gluon persistent | FP32; BF16 recurrent operand | FP32 | Global condition on the first step | No |
| Original `triton_fused` | FP32 local states after the first update; recurrent hidden rounded to `R.dtype` | FP32 | Per element | No |
| Original `cuda_fused`, default BF16 config | BF16 (`dtype_s=dtype_w`) | BF16 (`dtype_a=dtype_b`) | Per element | Input/forget gates capped at one |
| Original `cuda_fused`, explicit `dtype_a=dtype_s=float32` | FP32 | FP32 | Per element | Input/forget gates capped at one |

The last row is a configuration to qualify, not a tested baseline. Its public
state inputs/outputs are FP32. Any BF16 conversion needed for a common API must
be explicit and included in API/layer timing. Setting `dtype_s` alone does not
make pointwise arithmetic FP32. The native default remains a separately
reported baseline. Compiler fast-math and sum ordering remain additional
differences even when the configured dtypes agree.

`FlashRNNConfig.__post_init__` resolves `dtype_b` from `dtype`, `dtype_a` from
`dtype_b`, `dtype_w` from `dtype`, and `dtype_s` from `dtype_w`. In
`fused/flashrnn_fused_forward.cu`, recurrent contributions are added to bias
and cast to `DTYPE_A` before adding the input projection and casting again.
`fused/slstm_fused_pointwise.cuh` performs pointwise operations with `DTYPE_A`
and casts every new local state to `DTYPE_S`. The new kernels currently add
recurrent output, input projection and bias in FP32.

The sLSTM initialization difference is observable. With two normalizers
`[0, 1]`, input preactivations `[0, 0]`, forget preactivations `[0, 0]` and
old stabilizers `[2, 2]`, vanilla uses `m'=2-log(2)` for both coordinates;
the elementwise paths use `m'=0` for the first coordinate. CUDA additionally
caps its first forget gate at one, while original Triton does not. Zero
normalizer does not by itself require zero cell/stabilizer for the public
tensor interface. Mixed-state r4 PASS therefore establishes the stated
vanilla contract; it does not prove native CUDA/Triton equivalence.

The r4 sLSTM prototype remains FAILED under its original absolute budget.
Its BF16 midpoint split is documented in [tiling results](tiling-results.md).
Before independent baseline calibration, select and record the state,
pointwise and initialization contract. Do not combine errors from these
different paths into one tolerance distribution or relabel the old failure.
