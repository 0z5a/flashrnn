# Recurrent-model campaign coverage

The campaign remains incomplete. Sixteen language-model checkpoints and one
time-series forecasting checkpoint have CPU reference execution; these results
do not qualify their native accelerated
implementations or the requested high-concurrency GPU E2E matrix. The registry
now records completed Mamba3 SISO/MIMO, RWKV6, DeltaNet, RetNet, HGRN, HGRN2 and
Gated DeltaNet work rather than leaving stale not-downloaded entries. Original numerical
failures remain visible.

The [full-model speedup matrix](full-model-e2e-speedups.md) lists the accelerated baselines still requiring same-device qualification; none of its pretrained checkpoints has a reportable GPU E2E ratio yet. Separate fixed-weight synthetic four-layer controls cover [34 LSTM/sLSTM comparisons with original FlashRNN1](h100-flashrnn1-results.md), [51 LSTM/GRU/Elman comparisons with packed cuDNN](h100-full-matrix-results.md), and eight H20 BF16 FlashRNN1 comparisons in the [stacked-model report](stacked-e2e.md).
The [competing-baseline ledger](competing-baselines.md) separately tracks original FlashRNN, cuDNN/Haste and historical or parallel-time papers with their comparability gates.

| Registered family | Existing complete-model evidence | Remaining GPU/native qualification |
|---|---|---|
| Mamba 130M | [CPU native-Torch/Script long generation](mamba-native-reference.md): 576 steps, 1,344 token choices; [high-batch trace gate](mamba-highbatch-cpu.md): B16/B32/B64, 576 steps and 21,504 choices; [separately loaded native oracle](mamba-highbatch-oracle.md): 288 steps, 10,752 choices, 36 boundary pairs pass; [queued serving control](mamba-native-serving.md): 1,022 request executions | [Same-platform short Torch-fallback/Script CUDA](native-runtime-results.md) passes 3/3; accelerated mamba_ssm and long/high-concurrency GPU E2E unfinished |
| Mamba2 130M | [CPU long Torch/Script](mamba2-generation-qualification.md): 576 steps, 1,344 choices exact; [high-batch B16 three-group and B32 one-group gates](mamba2-highbatch-cpu.md): 256 steps, 5,120 choices, 288 tensor checks and 12 audited boundary pairs pass; cached/full-prefix logits retain failures | [Queued-serving comparison harness](mamba2-serving-harness.md) prepared; B32/B64 three-group, native accelerated and full GPU E2E unfinished |
| RWKV6 1.6B | [36-step/84-token reference](rwkv6-reference.md), nine BS=1 recurrent-state failures; [BlinkDL CPU crosscheck](rwkv6-blinkdl.md) retains cache failures; [chunk-prefix CPU gate](rwkv6-chunk-prefix.md) passes 12/20 FP32 and 20/20 algebraic FP64 controls | Native CUDA, backward and high-concurrency GPU E2E unfinished |
| RWKV7 0.4B | [Complete checkpoint](rwkv7-qualification.md): 36 steps/84 tokens; BS=2/4 state failures retained | Native CUDA and high-concurrency GPU E2E unfinished |
| GLA 1.3B | [Original 36-step reference](gla-reference-results.md) retains nine BS=1 state failures; [changed Linear arithmetic control](gla-numerics.md) passes 36/36 | Native FLA and high-concurrency GPU E2E unfinished |
| DeltaNet 1.3B | [Original reference](deltanet-reference.md) retains nine recurrent/six convolution failures; [changed Linear arithmetic control](deltanet-numerics.md) passes 36/36 | Native FLA and high-concurrency GPU E2E unfinished |
| RetNet 1.3B | [Original 36-step/84-token reference](retnet-reference.md) retains nine BS=1 state failures; [changed Linear arithmetic control](retnet-numerics.md) passes 36/36 with bitwise state/logit parity | Native FLA, long contexts and high-concurrency GPU E2E unfinished |
| xLSTM / mLSTM 7B | Immutable checkpoint metadata pinned; full checkpoint not downloaded | Complete-model and mlstm_kernels baseline execution unfinished |
| xLSTM / mLSTM 164M | [Complete checkpoint](mlstm-164m-reference.md): 36/36 logits and 84/84 tokens; original five cell-state failures retained, fixed-row Linear control bitwise pass | Native `mlstm_kernels` and high-concurrency GPU E2E unfinished |
| TiRex sLSTM 35M, time-series forecast | [Complete checkpoint](tirex-slstm-reference.md): nine forecasts, 21 series, 12,096 quantile values per path and 864 layer-state tensor comparisons bitwise | [H100 BF16 GPU numerical qualification](tirex-gpu-r5-results.md) passes 381/381 with B2 FlashRNN2 only; B1/B4 retain Torch. Native author CUDA, ONNX and high-concurrency forecast E2E remain unmeasured; not a language-token baseline |
| Mixed mLSTM/sLSTM language model, 2.1M | [Complete checkpoint](lovecraft-xlstm-reference.md): 36/36 full-vocabulary batch/serial and cached/full-prefix checks, 84/84 tokens, 504/504 state tensor pairs; independent audit pass | Native xLSTM CUDA, larger-model baseline and high-concurrency GPU E2E unfinished |
| Gated DeltaNet 340M | [Complete checkpoint](gdn-reference.md): 36/36 logits, recurrent and convolution checks; 84/84 tokens; independent audit pass | Native FLA and high-concurrency GPU E2E unfinished |
| Monostich-2-base GDN-2/GQA 149M | [Complete checkpoint](monostich2-reference.md): 36/36 full-model batched/serial logits and 84/84 tokens; independent audit and pinned first GDN-2 layer crosscheck pass | Native FLA, cache, backward and high-concurrency GPU E2E unfinished |
| HGRN2 1.3B | [Complete checkpoint](hgrn2-reference.md): 36 steps/84 tokens, seven BS=1 state failures; BS=2/4 bitwise | Native FLA and high-concurrency GPU E2E unfinished |
| HGRN 1.3B | [Complete checkpoint](hgrn-reference.md): 36/36 logits and recurrent-state checks, 84/84 tokens; independent audit pass | Native FLA and high-concurrency GPU E2E unfinished |
| Mamba3 SISO 187M | [Complete checkpoint](mamba3-siso-reference.md): 36/36 logits, angle/SSM/key/value states; 84/84 tokens; independent audit pass | Native Mamba-3 CUDA and high-concurrency GPU E2E unfinished |
| Mamba3 MIMO 187M | [Complete checkpoint](mamba3-mimo-reference.md): 36/36 logits, 84/84 tokens; original strict state gate failed, calibrated gate and independent audit pass | Native Mamba-3 CUDA and high-concurrency GPU E2E unfinished |
| Pure GatedDeltaNet2 305M | [Complete checkpoint](gdn2-pure-305m-reference.md): 36/36 full-vocabulary steps, 84/84 tokens, 1,008/1,008 cached/serial and 432/432 cached/full state pairs; independent audit pass | Native FLA fused recurrent/chunk, backward and high-concurrency GPU E2E unfinished |
| KDA / signed Complex KDA 1.3B | [Two complete checkpoints screened](kda-paired-numerics.md): BF16 B1/P5 and B1/P16 full-vocabulary logits fail in both models, FP32 controls and all 96 final-state pairs per run pass; eight raw-tensor audits pass | Full batch/generation qualification, native author kernels, backward and high-concurrency GPU E2E unfinished |

Changed-arithmetic diagnostics do not replace original failures or establish
training quality. The KDA screens do not add to the sixteen complete CPU model
qualifications above. RetNet's local weight was evicted after the completed CPU
diagnostic to make room for HGRN2 evidence; the pinned public weight needs
re-download before its GPU run. Gated DeltaNet, GLA, HGRN2, HGRN and Mamba3
SISO/MIMO, mLSTM 164M, TiRex sLSTM, Monostich-2-base, the mixed mLSTM/sLSTM language model and pure GDN-2 305M local weights were likewise evicted after completed CPU work and no-reader checks. The two KDA weights were evicted after their numerical screens and audits;
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
| Full recurrent models | Reports above; [synthetic stacked controls](stacked-e2e.md) | Pretrained high-concurrency GPU speedups unmeasured; H100 FP16 FlashRNN1 34/34 at 1.177×–1.808×, H100 packed cuDNN 51/51 at 1.422×–1.655×, and H20 BF16 FlashRNN1 eight rows at 0.8165×–0.9324× |

The required campaign still includes model batches 1/4/16/32/64, request
concurrency 1/8/32/64/128, native and accelerated baselines, comparable output
contracts, and paired measurements. SM90/Hopper, SM100/B200-specific kernels,
training, sanitizer/readiness tests, selector/holdouts and low-precision/Newton
extensions retain their separate acceptance gates. CPU/MPS results cannot
stand in for those hardware experiments.
