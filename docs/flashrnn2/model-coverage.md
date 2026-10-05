# Recurrent-model campaign coverage

The campaign remains incomplete. Fourteen language-model checkpoints and one
time-series forecasting checkpoint have CPU reference execution; these results
do not qualify their native accelerated
implementations or the requested high-concurrency GPU E2E matrix. The registry
now records completed Mamba3 SISO/MIMO, RWKV6, DeltaNet, RetNet, HGRN, HGRN2 and
Gated DeltaNet work rather than leaving stale not-downloaded entries. Original numerical
failures remain visible.

| Registered family | Existing complete-model evidence | Remaining GPU/native qualification |
|---|---|---|
| Mamba 130M | [CPU native-Torch/Script long generation](mamba-native-reference.md): 576 steps, 1,344 token choices; [queued serving control](mamba-native-serving.md): 1,022 request executions | [Same-platform short Torch-fallback/Script CUDA](native-runtime-results.md) passes 3/3; accelerated mamba_ssm and long/high-concurrency GPU E2E unfinished |
| Mamba2 130M | [CPU long Torch/Script](mamba2-generation-qualification.md): 576 steps, 1,344 choices exact; cached/full-prefix logits retain failures | Native accelerated and full GPU E2E unfinished |
| RWKV6 1.6B | [36-step/84-token reference](rwkv6-reference.md), nine BS=1 recurrent-state failures; [BlinkDL CPU crosscheck](rwkv6-blinkdl.md) retains cache failures; [chunk-prefix CPU gate](rwkv6-chunk-prefix.md) passes 12/20 FP32 and 20/20 algebraic FP64 controls | Native CUDA, backward and high-concurrency GPU E2E unfinished |
| RWKV7 0.4B | [Complete checkpoint](rwkv7-qualification.md): 36 steps/84 tokens; BS=2/4 state failures retained | Native CUDA and high-concurrency GPU E2E unfinished |
| GLA 1.3B | [Original 36-step reference](gla-reference-results.md) retains nine BS=1 state failures; [changed Linear arithmetic control](gla-numerics.md) passes 36/36 | Native FLA and high-concurrency GPU E2E unfinished |
| DeltaNet 1.3B | [Original reference](deltanet-reference.md) retains nine recurrent/six convolution failures; [changed Linear arithmetic control](deltanet-numerics.md) passes 36/36 | Native FLA and high-concurrency GPU E2E unfinished |
| RetNet 1.3B | [Original 36-step/84-token reference](retnet-reference.md) retains nine BS=1 state failures; [changed Linear arithmetic control](retnet-numerics.md) passes 36/36 with bitwise state/logit parity | Native FLA, long contexts and high-concurrency GPU E2E unfinished |
| xLSTM / mLSTM 7B | Immutable checkpoint metadata pinned; full checkpoint not downloaded | Complete-model and mlstm_kernels baseline execution unfinished |
| xLSTM / mLSTM 164M | [Complete checkpoint](mlstm-164m-reference.md): 36/36 logits and 84/84 tokens; original five cell-state failures retained, fixed-row Linear control bitwise pass | Native `mlstm_kernels` and high-concurrency GPU E2E unfinished |
| TiRex sLSTM 35M, time-series forecast | [Complete checkpoint](tirex-slstm-reference.md): nine forecasts, 21 series, 12,096 quantile values per path and 864 layer-state tensor comparisons bitwise | Native FlashRNN CUDA, ONNX and high-concurrency GPU forecast E2E unfinished; not a language-token baseline |
| sLSTM language model | Cell-level reference/gradient tests only | Complete checkpoint and model E2E unfinished |
| Gated DeltaNet 340M | [Complete checkpoint](gdn-reference.md): 36/36 logits, recurrent and convolution checks; 84/84 tokens; independent audit pass | Native FLA and high-concurrency GPU E2E unfinished |
| Monostich-2-base GDN-2/GQA 149M | [Complete checkpoint](monostich2-reference.md): 36/36 full-model batched/serial logits and 84/84 tokens; independent audit and pinned first GDN-2 layer crosscheck pass | Native FLA, cache, backward and high-concurrency GPU E2E unfinished |
| HGRN2 1.3B | [Complete checkpoint](hgrn2-reference.md): 36 steps/84 tokens, seven BS=1 state failures; BS=2/4 bitwise | Native FLA and high-concurrency GPU E2E unfinished |
| HGRN 1.3B | [Complete checkpoint](hgrn-reference.md): 36/36 logits and recurrent-state checks, 84/84 tokens; independent audit pass | Native FLA and high-concurrency GPU E2E unfinished |
| Mamba3 SISO 187M | [Complete checkpoint](mamba3-siso-reference.md): 36/36 logits, angle/SSM/key/value states; 84/84 tokens; independent audit pass | Native Mamba-3 CUDA and high-concurrency GPU E2E unfinished |
| Mamba3 MIMO 187M | [Complete checkpoint](mamba3-mimo-reference.md): 36/36 logits, 84/84 tokens; original strict state gate failed, calibrated gate and independent audit pass | Native Mamba-3 CUDA and high-concurrency GPU E2E unfinished |
| GatedDeltaNet2 | Hybrid checkpoint above exercises GDN-2; pure native baseline remains registered | Standalone GDN-2 checkpoint/native GPU E2E unfinished |
| KDA | Family registered | Complete pinned model/native baseline unfinished |

Changed-arithmetic diagnostics do not replace original failures or establish
training quality. RetNet's local weight was evicted after the completed CPU
diagnostic to make room for HGRN2 evidence; the pinned public weight needs
re-download before its GPU run. Gated DeltaNet, GLA, HGRN2, HGRN and Mamba3
SISO/MIMO, mLSTM 164M, TiRex sLSTM and Monostich-2-base local weights were likewise evicted after completed CPU work and no-reader checks;
their pinned versions need re-download before GPU use. Other model weights
remain needed for unfinished native/GPU consumers. Five redundant closed export
files were losslessly archived locally;
replay must restore those exact files from their retained delta manifests.

## Cell, layer and model results have different scopes

| Level | Evidence | Speed result |
|---|---|---|
| Four-cell functional API | [Portable CPU](portable-gate.md) and [Apple MPS](mps-gate.md), 38 cases per matrix | Correctness only |
| Small CPU compiled recurrence | [Inductor](compile-gate.md), B2/T4/H1/D8, full history/gradients | Local forward+loss+backward probe 1.61–3.44×; no model E2E claim |
| One LSTM GPU layer | [Paired layer pilot](layer-timing.md), B16/T128/H1/D64 | 6.3669× versus cuDNN in this layer pilot |
| Full recurrent models | Reports above | No full-model/high-concurrency GPU speedup qualified |

The required campaign still includes model batches 1/4/16/32/64, request
concurrency 1/8/32/64/128, native and accelerated baselines, comparable output
contracts, and paired measurements. SM90/Hopper, SM100/B200-specific kernels,
training, sanitizer/readiness tests, selector/holdouts and low-precision/Newton
extensions retain their separate acceptance gates. CPU/MPS results cannot
stand in for those hardware experiments.
