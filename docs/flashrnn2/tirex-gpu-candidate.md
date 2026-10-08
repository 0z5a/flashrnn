# TiRex sLSTM GPU candidate

This branch adds a TiRex-specific BF16 recurrence mode and an adapter for the
public [TiRex checkpoint](https://huggingface.co/NX-AI/TiRex/tree/63c740922493f5fbe60b277609ec62babfba2762).
The default FlashRNN2 sLSTM mode is unchanged. The variant follows the
[pinned Torch cell](https://github.com/NX-AI/tirex/blob/91b67bc6e5d4d5d1e68a302dd848dd1d612f4c97/src/tirex/models/slstm/cell.py):
BF16 recurrent projection and per-step state, FP32 pointwise bias, capped
forget preactivation and gates, and an unclamped normalizer.

The GPU qualification command uses the pinned checkpoint and source tree:

```sh
python tools/flashrnn2/tirex_gpu_gate.py \
  --model "$TIREX_MODEL_DIR" --source "$TIREX_SOURCE_DIR" \
  --output "$PRIVATE_RESULTS/tirex-gpu-gate.json"
```

It compares the official Torch cell and full 12-block, 64-step forecast with
the candidate on the same GPU. Batch sizes 1, 2 and 4 cover synthetic cell
inputs and the public forecast path; each block's output and four recurrent
states are checked after both forecast patches at the CPU reference's
unchanged `1e-4 × (1 + |reference|)` tolerance. The gate saves raw tensor
pairs for an independent audit. Passing this gate is required before paired
AB/BA E2E timing and high-concurrency throughput measurements.

| Checkpoint GPU comparison | Torch baseline | FlashRNN2 candidate | Speedup |
|---|---:|---:|---:|
| Per-cell BF16 state/output parity, r2 | Reference | Failed 15/15 tensor pairs | — |
| Full forecast B1/B2/B4 parity, r2 | Reference | Failed 366/366 tensor pairs | — |
| Repaired adapter isolated cell parity, r3 | Reference | Failed 5/15 tensor pairs | — |
| Repaired adapter full forecast parity, r3 | Reference | Failed 302/366 tensor pairs | — |
| CUDA-libdevice isolated cell parity, r4 | Reference | Passed 15/15 tensor pairs | — |
| CUDA-libdevice full forecast parity, r4 | Reference | Failed 206/366 tensor pairs | — |
| B2-only candidate, Torch fallback for other batches, r5 | Reference | GPU qualification pending | — |
| Paired full-model forecast E2E | Not timed | Not timed | — |

The [r2 failure report](tirex-gpu-r2-results.md) includes the checkpoint-weight
layout diagnosis. The [r3 failure report](tirex-gpu-r3-results.md) includes
the corrected-layout GPU comparison, independent 381-pair audit and raw
tensors. A subsequent candidate uses CUDA libdevice `exp` and `log1p` for the
TiRex pointwise path, matching the operation sequence in PyTorch's CUDA
[log-sigmoid](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/cuda/ActivationLogSigmoidKernel.cu)
and sigmoid kernels. The [r4 result](tirex-gpu-r4-results.md) shows that all
isolated-cell comparisons and the B2 full forecast now pass, while B1 and B4
full forecasts remain outside the original budget. This is not a passing
complete GPU gate.

The next adapter dispatches B2 to the CUDA-libdevice candidate and retains
the original TiRex Torch cell for all other batch sizes. B1 and B4 will not
be reported as FlashRNN2-accelerated. This source change has no GPU result
yet: the full 381-pair gate and independent audit must pass before any
forecast timing. The gate records the dispatch and adapter source hash so
the timing runner cannot reuse a qualification from another adapter.

The prior [CPU reference](tirex-slstm-reference.md) validates the checkpoint's
official Torch forecast. It does not establish GPU parity or a speedup.
