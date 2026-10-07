# H100 complete-model E2E: full legal batch × concurrency matrix

FlashRNN2 was compared with matched packed cuDNN for synthetic FP16 LSTM, GRU and Elman sequence classifiers. All 51 legal configurations (17 per cell) completed 20 alternating AB/BA paired blocks. Each passed an independent CPU comparison of saved logits, layer histories and final states. FlashRNN2 was faster in all 51 configurations; this is a qualified cuDNN comparison, not a pretrained-model result.

The model has four D64 recurrent layers, a 512-token embedding and readout, and T32 inputs. The timed path includes host-to-device IDs, embedding, recurrence, readout, host-visible logits and queued B-sized groups. Both arms use identical random weights and fixtures. Construction, compilation, warmup, fixture generation and qualification are excluded. cuDNN ran in eval mode, one packed storage per layer, with `aten::_cudnn_rnn` dispatch. Throughput is completed sequence responses/s on one H100, including for cases run concurrently on separate H100s.

B is the batch per queued group; C requests arrive at time zero and one worker processes them in B-sized groups. C does not mean C simultaneous CUDA streams. The 95% intervals bootstrap the 20 paired blocks within one process; independent-start variance was not measured. Runtime: PyTorch 2.4.0a0, CUDA 11.8, cuDNN 9.0, Triton 3.0.0.

## LSTM

| Batch | Concurrent requests | cuDNN responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1,508.3 | 2,237.0 | 1.622× [1.468, 1.911] |
| 1 | 8 | 1,851.0 | 2,837.8 | 1.533× [1.525, 1.540] |
| 1 | 32 | 1,915.5 | 2,977.3 | 1.568× [1.551, 1.593] |
| 1 | 64 | 1,912.9 | 2,993.5 | 1.568× [1.560, 1.577] |
| 1 | 128 | 1,978.0 | 3,063.4 | 1.546× [1.526, 1.564] |
| 4 | 8 | 6,335.3 | 10,144.2 | 1.595× [1.566, 1.623] |
| 4 | 32 | 7,021.9 | 11,396.6 | 1.617× [1.602, 1.630] |
| 4 | 64 | 7,055.0 | 11,641.3 | 1.646× [1.638, 1.653] |
| 4 | 128 | 7,345.0 | 12,000.8 | 1.623× [1.598, 1.642] |
| 16 | 32 | 24,815.2 | 39,721.5 | 1.607× [1.579, 1.632] |
| 16 | 64 | 26,192.3 | 42,910.9 | 1.643× [1.624, 1.662] |
| 16 | 128 | 26,893.6 | 44,684.7 | 1.655× [1.643, 1.666] |
| 32 | 32 | 43,954.5 | 68,713.3 | 1.572× [1.528, 1.613] |
| 32 | 64 | 48,249.2 | 77,588.2 | 1.615× [1.589, 1.640] |
| 32 | 128 | 52,175.2 | 84,604.6 | 1.620× [1.606, 1.637] |
| 64 | 64 | 85,431.2 | 132,613.0 | 1.554× [1.517, 1.591] |
| 64 | 128 | 94,141.3 | 151,473.8 | 1.614× [1.587, 1.641] |

## GRU

| Batch | Concurrent requests | cuDNN responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1,661.4 | 2,335.9 | 1.422× [1.374, 1.469] |
| 1 | 8 | 2,076.2 | 3,115.7 | 1.497× [1.489, 1.505] |
| 1 | 32 | 2,202.3 | 3,317.1 | 1.508× [1.501, 1.517] |
| 1 | 64 | 2,220.4 | 3,354.0 | 1.497× [1.468, 1.517] |
| 1 | 128 | 2,247.6 | 3,274.8 | 1.457× [1.437, 1.475] |
| 4 | 8 | 6,980.7 | 10,895.6 | 1.550× [1.518, 1.580] |
| 4 | 32 | 7,833.0 | 12,380.2 | 1.578× [1.568, 1.586] |
| 4 | 64 | 8,081.0 | 12,768.1 | 1.578× [1.572, 1.584] |
| 4 | 128 | 8,289.8 | 13,052.4 | 1.581× [1.562, 1.601] |
| 16 | 32 | 26,562.9 | 41,205.7 | 1.544× [1.509, 1.577] |
| 16 | 64 | 29,511.4 | 46,104.8 | 1.614× [1.539, 1.754] |
| 16 | 128 | 31,049.3 | 48,998.4 | 1.572× [1.549, 1.594] |
| 32 | 32 | 48,720.9 | 74,350.7 | 1.524× [1.480, 1.566] |
| 32 | 64 | 52,674.2 | 81,672.0 | 1.554× [1.523, 1.582] |
| 32 | 128 | 56,613.7 | 88,676.1 | 1.566× [1.552, 1.578] |
| 64 | 64 | 91,433.0 | 138,654.2 | 1.509× [1.474, 1.538] |
| 64 | 128 | 104,001.4 | 163,519.8 | 1.565× [1.538, 1.590] |

## Elman

| Batch | Concurrent requests | cuDNN responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1,684.1 | 2,362.9 | 1.468× [1.375, 1.631] |
| 1 | 8 | 2,137.4 | 3,134.4 | 1.470× [1.457, 1.484] |
| 1 | 32 | 2,251.0 | 3,317.0 | 1.455× [1.432, 1.473] |
| 1 | 64 | 2,264.8 | 3,371.4 | 1.506× [1.489, 1.526] |
| 1 | 128 | 2,283.6 | 3,397.9 | 1.492× [1.486, 1.501] |
| 4 | 8 | 7,038.9 | 10,884.8 | 1.558× [1.522, 1.592] |
| 4 | 32 | 8,028.8 | 12,630.1 | 1.576× [1.565, 1.585] |
| 4 | 64 | 8,199.4 | 13,262.0 | 1.617× [1.610, 1.623] |
| 4 | 128 | 8,370.5 | 13,437.7 | 1.619× [1.566, 1.673] |
| 16 | 32 | 27,373.9 | 42,328.9 | 1.548× [1.513, 1.581] |
| 16 | 64 | 30,133.9 | 47,610.3 | 1.585× [1.563, 1.606] |
| 16 | 128 | 31,181.1 | 49,837.7 | 1.582× [1.557, 1.599] |
| 32 | 32 | 49,547.4 | 73,977.7 | 1.498× [1.452, 1.539] |
| 32 | 64 | 54,853.6 | 84,635.4 | 1.544× [1.512, 1.572] |
| 32 | 128 | 57,980.7 | 91,342.8 | 1.583× [1.568, 1.602] |
| 64 | 64 | 97,140.0 | 145,128.2 | 1.509× [1.477, 1.537] |
| 64 | 128 | 105,540.3 | 162,314.4 | 1.535× [1.511, 1.558] |

The [51-case independent audit](../../evidence/flashrnn2-h100-full-matrix-r1/audit.json) covers all saved tensor pairs and timing hashes. This PR adds 43 cases with [raw timing, redacted metadata, saved qualification tensors and per-case analyses](../../evidence/flashrnn2-h100-full-matrix-r1/); the preceding [GRU/Elman evidence](../../evidence/flashrnn2-h100-gru-elman-r1/audit.json) contains the other eight. The one-GPU pilot and later one-case-per-GPU runs are pooled as distinct configurations, not as repeated independent starts. These synthetic sequence responses/s must not be read as autoregressive tokens/s.
