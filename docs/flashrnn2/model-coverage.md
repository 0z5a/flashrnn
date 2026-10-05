# Recurrent-model campaign coverage

The campaign remains incomplete. Ten complete checkpoint families have CPU
reference execution; these results do not qualify their native accelerated
implementations or the requested high-concurrency GPU E2E matrix. The registry
now records completed RWKV6, DeltaNet, RetNet, HGRN, HGRN2 and Gated DeltaNet
work rather than leaving stale not-downloaded entries. Original numerical
failures remain visible.

| Registered family | Existing complete-model evidence | Remaining GPU/native qualification |
|---|---|---|
| Mamba 130M | [CPU native-Torch/Script long generation](mamba-native-reference.md): 576 steps, 1,344 token choices; [queued serving control](mamba-native-serving.md): 1,022 request executions | [Same-platform short Torch-fallback/Script CUDA](native-runtime-results.md) passes 3/3; accelerated mamba_ssm and long/high-concurrency GPU E2E unfinished |
| Mamba2 130M | [CPU long Torch/Script](mamba2-generation-qualification.md): 576 steps, 1,344 choices exact; cached/full-prefix logits retain failures | Native accelerated and full GPU E2E unfinished |
| RWKV6 1.6B | [36-step/84-token reference](rwkv6-reference.md), nine BS=1 recurrent-state failures; [BlinkDL CPU crosscheck](rwkv6-blinkdl.md) retains cache failures | Native CUDA and high-concurrency GPU E2E unfinished |
| RWKV7 0.4B | [Complete checkpoint](rwkv7-qualification.md): 36 steps/84 tokens; BS=2/4 state failures retained | Native CUDA and high-concurrency GPU E2E unfinished |
| GLA 1.3B | [Original 36-step reference](gla-reference-results.md) retains nine BS=1 state failures; [changed Linear arithmetic control](gla-numerics.md) passes 36/36 | Native FLA and high-concurrency GPU E2E unfinished |
| DeltaNet 1.3B | [Original reference](deltanet-reference.md) retains nine recurrent/six convolution failures; [changed Linear arithmetic control](deltanet-numerics.md) passes 36/36 | Native FLA and high-concurrency GPU E2E unfinished |
| RetNet 1.3B | [Original 36-step/84-token reference](retnet-reference.md) retains nine BS=1 state failures; [changed Linear arithmetic control](retnet-numerics.md) passes 36/36 with bitwise state/logit parity | Native FLA, long contexts and high-concurrency GPU E2E unfinished |
| xLSTM / mLSTM 7B | Immutable checkpoint metadata pinned; full checkpoint not downloaded | Complete-model and mlstm_kernels baseline execution unfinished |
| sLSTM language model | Cell-level reference/gradient tests only | Complete checkpoint and model E2E unfinished |
| Gated DeltaNet 340M | [Complete checkpoint](gdn-reference.md): 36/36 logits, recurrent and convolution checks; 84/84 tokens; independent audit pass | Native FLA and high-concurrency GPU E2E unfinished |
| HGRN2 1.3B | [Complete checkpoint](hgrn2-reference.md): 36 steps/84 tokens, seven BS=1 state failures; BS=2/4 bitwise | Native FLA and high-concurrency GPU E2E unfinished |
| HGRN 1.3B | [Complete checkpoint](hgrn-reference.md): 36/36 logits and recurrent-state checks, 84/84 tokens; independent audit pass | Native FLA and high-concurrency GPU E2E unfinished |
| Mamba3 | Family registered | Complete pinned model/native baseline unfinished |
| GatedDeltaNet2 | Family registered | Complete pinned model/native baseline unfinished |
| KDA | Family registered | Complete pinned model/native baseline unfinished |

Changed-arithmetic diagnostics do not replace original failures or establish
training quality. RetNet's local weight was evicted after the completed CPU
diagnostic to make room for HGRN2 evidence; the pinned public weight needs
re-download before its GPU run. Gated DeltaNet, GLA, HGRN2 and HGRN local
weights were likewise evicted after completed CPU work and no-reader checks;
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
