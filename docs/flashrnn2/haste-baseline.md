# Haste baseline qualification

Official [Haste source](https://github.com/lmnt-com/haste/tree/ceba32ecd3735062f25e56b08ef3143caa086753) was fetched as a source-only archive at revision `ceba32ecd3735062f25e56b08ef3143caa086753`; archive SHA-256 `c0d12f51dd1e60dbeb81aa4eab059f12b9675c55c47124ff248c580317032b90`. No package was installed. The gate extracts unchanged `LSTMScript` and `GRUScript` definitions only after checking their complete source hashes (`64dc001d…f4ed26` and `567e1835…51c3e4`). Torch 2.14.1 completed the CPU equation gate; this does not qualify the native CUDA library.

| Official binding | Native dtype dispatch at this revision | Equal-workload mapping | Result |
| --- | --- | --- | --- |
| LSTM | `AT_DISPATCH_FLOATING_TYPES`: FP32/FP64 | Haste `i,g,f,o` ↔ FlashRNN `i,f,z,o`; disable DropConnect/Zoneout, map initial h/c and all gradients | CPU forward and gradient gate passed for both losses |
| GRU | `AT_DISPATCH_FLOATING_TYPES_AND_HALF`: FP16/FP32/FP64 | Haste `z,r,h` uses FlashRNN input gates `[1,0,2]`, recurrent gates `[2,1,0]`, and biases `[2,1,3]` plus recurrent candidate bias `[0]`; disable regularization | CPU forward and gradient gate passed for both losses |
| Existing BF16 L1 cohort | Neither binding dispatches BF16 | No same-dtype Haste control | Ineligible for a Haste speed ratio |

The pinned [PyTorch LSTM binding](https://github.com/lmnt-com/haste/blob/ceba32ecd3735062f25e56b08ef3143caa086753/frameworks/pytorch/lstm.cc) and [GRU binding](https://github.com/lmnt-com/haste/blob/ceba32ecd3735062f25e56b08ef3143caa086753/frameworks/pytorch/gru.cc) establish the dtype limits. The stock Makefile targets compute 3.7/6.0/7.0; a task-local build on current CUDA hardware needs a separately audited arch transport. Neither a source audit nor Torch-equation parity proves that native Haste compiles, dispatches or beats FlashRNN. Full-model E2E remains [unmeasured](full-model-e2e-speedups.md).

The input and recurrent permutations differ because the FlashRNN GRU reference reads input reset/update/candidate from gates 0/1/2, but recurrent candidate/reset/update from gates 0/1/2. Haste reads both projections as update/reset/candidate. The previous `[2,1,0]` input permutation fed candidate input to the update gate; it is invalid for the equal-workload gate.

The four raw CPU rows use B3/T7/H4/I5 FP64 with both history-plus-final and final-only losses. They passed `atol=rtol=1e-12` for outputs and `atol=rtol=1e-11` for gradients. The independent row and hash check found all four expected combinations, no extras, and SHA-256 `3c49a3fae2d0227198d03fee5ab1882c3ee3bb365a1b81afbdf93d0852efc9b9` for the raw file.

| Cell | Loss | Maximum output error | Maximum gradient error |
| --- | --- | ---: | ---: |
| LSTM | History and final | 6.94e-18 | 5.55e-17 |
| LSTM | Final only | 6.94e-18 | 1.39e-17 |
| GRU | History and final | 1.39e-17 | 4.44e-16 |
| GRU | Final only | 1.39e-17 | 1.11e-16 |

The [task-local H100 builder](../../tools/flashrnn2/build_haste_native.py) verifies 18 pinned source/header hashes and stages the original LSTM/GRU CUDA translation units under `.cu` filenames for PyTorch's extension builder. Its only generated C++ registers those two existing bindings; the recurrent kernels and wrappers are unchanged. The [native gate](../../tools/flashrnn2/haste_native_gate.py) saves FP32/FP64 LSTM and FP16/FP32/FP64 GRU outputs for one identical B3/T7/H4/I5 fixture, with the loaded extension binary SHA in producer metadata. A [separate NumPy auditor](../../tools/flashrnn2/audit_haste_native.py) replays all 20 tensor comparisons and binds its report to that binary SHA for later E2E. Source-hash checking and the earlier auditor's positive/negative CPU fixtures passed locally; the revised binary binding remains GPU-unverified. Native Haste compilation, GPU parity and paired full-model E2E remain unmeasured. Only GRU FP16 can enter a same-dtype comparison with the current FlashRNN2 Triton kernels; the Haste LSTM binding does not dispatch FP16/BF16.

The four-layer [stacked E2E runner](../../tools/flashrnn2/stacked_e2e.py) has a source-only Haste GRU FP16 route. It converts the same model weights to Haste's layout once at construction, includes every layer, embedding, readout and output transfer in both timed arms, and requires the independent native-gate PASS record, raw hash and the same extension binary SHA used by the gate before timing. No Haste CUDA or E2E result has been recorded.
