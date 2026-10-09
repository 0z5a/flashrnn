# Full-model E2E speedup matrix

No pretrained-checkpoint, high-concurrency GPU speedup has passed the paired measurement gate yet. The single-layer GPU pilot and small CPU compile probe have narrower contracts and are excluded from this table. Each future result must use the same checkpoint, inputs, output contract, batch, concurrency and device for baseline and candidate; report paired AB/BA blocks and a 95% interval. Language-model throughput is generated tokens/s; TiRex uses completed forecasts/s.

Separate fixed-weight synthetic controls report [34 H100 FP16 LSTM/sLSTM rows against FlashRNN1](h100-flashrnn1-results.md) at 1.177×–1.808× and [51 H100 FP16 LSTM/GRU/Elman rows against packed cuDNN](h100-full-matrix-results.md) at 1.422×–1.655×. These fixed-weight sequence-responses/s controls do not fill any pretrained-model row below.

| Model family | Accelerated baseline to qualify | Full-model GPU baseline throughput | Candidate throughput | Paired speedup [95% CI] |
| --- | --- | ---: | ---: | ---: |
| LSTM / GRU / Elman / sLSTM representative pretrained stacks | FlashRNN1 backends, cuDNN and Haste where the cell/output contract matches; synthetic controls linked above | — | — | Pretrained checkpoint unmeasured |
| Mamba 130M | `mamba_ssm` fused scan and native Torch control | — | — | Unmeasured |
| Mamba2 130M | Author fused backend and native Torch control | — | — | Unmeasured |
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
| TiRex sLSTM 35M forecast | Official Torch; B2-only GPU numerical gate passed, B1/B4 use Torch fallback; forecast/s, not tok/s | — | — | E2E unmeasured |
| KDA / signed Complex KDA 1.3B | Author fused kernels; BF16 numerical gate pending | — | — | Unmeasured |

The model coverage registry links each checkpoint's completed CPU evidence and retained numerical failures. Baselines in this table are targets for qualification, not claims that those kernels already ran. Request concurrency and model batch are separate axes; unsupported pairs must be recorded rather than silently replaced by another workload.

The Mamba 130M high-batch serving control also requires restoration of its three original CPU oracle tensors and verification against their recorded hashes before GPU timing. Its source-only eight-case B16/B32/B64 × C32/C64/C128 plan is conditional on a separate numerical gate for each batch; no throughput row is available yet.

Persistent RNN, Sparse Persistent RNNs, DEER and ParaRNN are additional paper controls. Their sparse, adapted-cell or training-time results need their own matched workload and must not be inserted as same-checkpoint inference speedups. No pretrained-checkpoint complete-model GPU result from these papers or FlashRNN1 has been measured here.
