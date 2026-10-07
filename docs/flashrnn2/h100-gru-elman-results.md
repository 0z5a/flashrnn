# GRU and Elman complete-model E2E on H100

FlashRNN2's single-CTA GRU and Elman recurrences passed FP16 GPU correctness checks against an independent reference at B16/B32, T32, D64. The largest hidden-state error was 0.000122. Matched PyTorch cuDNN LSTM, GRU and Elman mapping checks passed: each used packed, acceptable weights, dispatched `aten::_cudnn_rnn`, and stayed within 0.000092 of the reference. These checks precede the complete-model measurements below.

The timed workload is a four-layer D64 sequence classifier with a 512-token embedding and readout, T32 input IDs, FP16 weights, and fixed random fixtures. It includes IDs transfer, embedding, all recurrent layers, readout, host-visible logits and queued B-sized groups. Model construction, compilation, warmup and numerical qualification are excluded. Each baseline and candidate uses the same weights and inputs. The cuDNN baseline is in eval mode, has one packed weight storage per layer, passes the weight-acceptability gate and dispatches `aten::_cudnn_rnn`.

| Cell | Batch | Concurrent requests | cuDNN responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 16 | 32 | 26,562.871 | 41,205.729 | 1.544× [1.509, 1.577] |
| Elman | 16 | 32 | 27,373.942 | 42,328.855 | 1.548× [1.513, 1.581] |
| GRU | 16 | 128 | 31,049.260 | 48,998.364 | 1.572× [1.549, 1.594] |
| Elman | 16 | 128 | 31,181.102 | 49,837.699 | 1.582× [1.557, 1.599] |
| GRU | 32 | 128 | 56,613.749 | 88,676.117 | 1.566× [1.552, 1.578] |
| Elman | 32 | 128 | 57,980.711 | 91,342.809 | 1.583× [1.568, 1.602] |
| GRU | 64 | 128 | 104,001.387 | 163,519.777 | 1.565× [1.538, 1.590] |
| Elman | 64 | 128 | 105,540.339 | 162,314.354 | 1.535× [1.511, 1.558] |

All eight cases completed 20 alternating AB/BA paired blocks and passed the per-layer hidden-history, final-state and logit gates. An independent CPU pass compared all 288 saved baseline/candidate tensor pairs and verified their qualification hashes. The [raw rows, redacted metadata, saved qualification tensors and audit](../../evidence/flashrnn2-h100-gru-elman-r1/audit.json) are committed; each case has a matching `-summary.md` and `-summary.json` derived by the [analyzer](../../tools/flashrnn2/analyze_stacked_e2e.py).

The first four B16/C32 and B32/C128 cases ran on one H100. The remaining four ran simultaneously on four H100s, with one GPU assigned to each case. Per-row throughput is the median completed sequence responses/s on one GPU and is not aggregate four-GPU throughput. The speedup is the geometric mean of paired block ratios; the interval bootstraps those blocks 10,000 times within one process. Independent-start variance was not measured. C requests arrive at time zero and one worker serves them in B-sized groups; C does not mean C simultaneous CUDA streams.

The runtime was PyTorch 2.4.0a0, CUDA 11.8, cuDNN 9.0 and Triton 3.0.0. These are synthetic complete-model sequence-response results, not pretrained-checkpoint generation tokens/s.
