# Full-model E2E speedup matrix

No full-model, high-concurrency GPU speedup has passed the paired measurement gate yet. The [single-layer GPU pilot](layer-timing.md) and [small CPU compile probe](compile-gate.md) have narrower contracts and are excluded from this table. Each future result must use the same checkpoint, inputs, output contract, batch, concurrency and device for baseline and candidate; report paired AB/BA blocks and a 95% interval. Language-model throughput is generated tokens/s; TiRex uses completed forecasts/s.

| Model family | Accelerated baseline to qualify | Full-model GPU baseline throughput | Candidate throughput | Paired speedup [95% CI] |
| --- | --- | ---: | ---: | ---: |
| LSTM / GRU / Elman / sLSTM representative stacks | cuDNN / `torch.nn` where the output contract matches; upstream FlashRNN | — | — | Unmeasured |
| Mamba 130M | `mamba_ssm` fused scan and native Torch control | — | — | Unmeasured |
| Mamba2 130M | Author fused backend and [native Torch control](mamba2-serving-harness.md) | — | — | Unmeasured |
| Mamba3 SISO / MIMO 187M | Author Mamba-3 CUDA | — | — | Unmeasured |
| RWKV6 1.6B | BlinkDL recurrent and FLA chunk | — | — | Unmeasured |
| RWKV7 0.4B | Author recurrent and FLA chunk | — | — | Unmeasured |
| GLA 1.3B | FLA recurrent and chunk | — | — | Unmeasured |
| DeltaNet 1.3B | FLA recurrent and chunk | — | — | Unmeasured |
| RetNet 1.3B | Author / FLA recurrent and chunk | — | — | Unmeasured |
| HGRN / HGRN2 1.3B | FLA recurrent and chunk | — | — | Unmeasured |
| Gated DeltaNet 340M | FLA recurrent and chunk | — | — | Unmeasured |
| Gated DeltaNet2 305M | FLA fused recurrent and chunk | — | — | Unmeasured |
| Monostich-2 149M GDN-2/GQA | Author / FLA GDN-2 and attention | — | — | Unmeasured |
| xLSTM mLSTM 164M | `mlstm_kernels` | — | — | Unmeasured |
| xLSTM mLSTM 7B | `mlstm_kernels`; full checkpoint not downloaded | — | — | Unmeasured |
| Mixed mLSTM/sLSTM 2.1M | Author xLSTM CUDA | — | — | Unmeasured |
| TiRex sLSTM 35M forecast | Author FlashRNN CUDA and ONNX; forecast/s, not tok/s | — | — | Unmeasured |
| KDA / signed Complex KDA 1.3B | Author fused kernels; BF16 numerical gate pending | — | — | Unmeasured |

The [model coverage registry](model-coverage.md) links each checkpoint's completed CPU evidence and retained numerical failures. Baselines in this table are targets for qualification, not claims that those kernels already ran. Request concurrency and model batch are separate axes; unsupported pairs must be recorded rather than silently replaced by another workload.
