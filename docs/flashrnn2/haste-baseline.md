# Haste baseline qualification

Official [Haste source](https://github.com/lmnt-com/haste/tree/ceba32ecd3735062f25e56b08ef3143caa086753) was fetched as a source-only archive at revision `ceba32ecd3735062f25e56b08ef3143caa086753`; archive SHA-256 `c0d12f51dd1e60dbeb81aa4eab059f12b9675c55c47124ff248c580317032b90`. No package was installed. The gate extracts unchanged `LSTMScript` and `GRUScript` definitions only after checking their complete source hashes (`64dc001d…f4ed26` and `567e1835…51c3e4`). This qualifies equations on CPU, not the native CUDA library.

| Official binding | Native dtype dispatch at this revision | Equal-workload mapping | Result |
| --- | --- | --- | --- |
| LSTM | `AT_DISPATCH_FLOATING_TYPES`: FP32/FP64 | Haste `i,g,f,o` ↔ FlashRNN `i,f,z,o`; disable DropConnect/Zoneout, map initial h/c and all gradients | Source inspected; CPU Torch-equation gate pending |
| GRU | `AT_DISPATCH_FLOATING_TYPES_AND_HALF`: FP16/FP32/FP64 | Haste `z,r,h` ↔ FlashRNN update/reset/candidate; recurrent candidate bias stays behind reset; disable regularization | Source inspected; CPU Torch-equation gate pending |
| Existing BF16 L1 cohort | Neither binding dispatches BF16 | No same-dtype Haste control | Ineligible for a Haste speed ratio |

The pinned [PyTorch LSTM binding](https://github.com/lmnt-com/haste/blob/ceba32ecd3735062f25e56b08ef3143caa086753/frameworks/pytorch/lstm.cc) and [GRU binding](https://github.com/lmnt-com/haste/blob/ceba32ecd3735062f25e56b08ef3143caa086753/frameworks/pytorch/gru.cc) establish the dtype limits. The stock Makefile targets compute 3.7/6.0/7.0; a task-local build on current CUDA hardware needs a separately audited arch transport. Neither a source audit nor Torch-equation parity proves that native Haste compiles, dispatches or beats FlashRNN. Full-model E2E remains [unmeasured](full-model-e2e-speedups.md).
