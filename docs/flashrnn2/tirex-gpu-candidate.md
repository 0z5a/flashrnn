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
states are checked after both forecast patches. The gate saves raw tensor
pairs for an independent audit. Passing this gate is required before paired
AB/BA E2E timing and high-concurrency throughput measurements.

| Checkpoint GPU comparison | Torch baseline | FlashRNN2 candidate | Speedup |
|---|---:|---:|---:|
| Per-cell BF16 state/output parity | Not run | Not run | — |
| Full forecast B1/B2/B4 parity | Not run | Not run | — |
| Paired full-model forecast E2E | Not run | Not run | — |

The prior [CPU reference](tirex-slstm-reference.md) validates the checkpoint's
official Torch forecast. It does not establish GPU parity or a speedup.
