# Pinned source map

Upstream: `0b84fc30120ecccbde97976bc5b086f3215572d2`.

| Symbol / behavior | Source |
| --- | --- |
| `flashrnn`, `_get_config`, `_get_kernel`, `FlashRNNConfig` | `flashrnn/flashrnn/flashrnn.py` |
| Cell/gate registry | `rnn_function_registry` in the same module |
| Public vanilla recurrence and pointwise cells | `flashrnn/flashrnn/vanilla/` |
| CUDA fused forward/backward and cell expressions | `flashrnn/flashrnn/fused/` |
| Alternating CUDA | `flashrnn/flashrnn/alternating/` |
| Original Triton LSTM/sLSTM forward/backward | `flashrnn/flashrnn/triton_fused/` |
| CUDA extension compilation | `cuda_init.py`, `cuda_init_parametric.py` |
| Constraint solver | `flashrnn/autotune/constrint.py` |
| Existing tests | `flashrnn/tests/test_{lstm,slstm,gru,elman}.py` |
| Existing benchmark | `flashrnn/speed_experiments/kernel_speed_benchmark.py` |

The actual Triton backend key is `triton_fused`; the README's `triton` label is not accepted by this pinned dispatcher. Gradient configuration fields are `gradient_recurrent_cut`, `gradient_recurrent_clipval`, and `forward_clipval`. CPU reference tests use the vanilla mathematical/autograd path, with no clipping or recurrent cut.

The new model campaign lists intended families and baseline categories only. It does not claim that FlashRNN implements Mamba, RWKV, mLSTM, GLA or DeltaNet, or that their state equations are interchangeable. Each requires its own official model and backend adapter.
