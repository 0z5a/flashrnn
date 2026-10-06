# H20 FlashRNN2 R7: C128 stacked E2E

Seven children and the original controller/wrapper naturally exited 0. BF16, T128, four layers, D64, C128; fixed random model weights and request IDs.

| Cell | B / C | FlashRNN1 median ms | FlashRNN2 median ms | Paired speedup [95% CI] | Throughput change | Maximum absolute error |
|---|---|---:|---:|---:|---:|---:|
| lstm | 16 / 128 | 8.0747 | 9.8938 | 0.8165x [0.8136, 0.8197] | -18.35% | 0.000244141 |
| slstm | 16 / 128 | 8.9437 | 9.7532 | 0.9185x [0.9156, 0.9217] | -8.15% | 0.000488281 |
| lstm | 32 / 128 | 4.1076 | 5.0354 | 0.8167x [0.8128, 0.8224] | -18.33% | 0.000244141 |
| slstm | 32 / 128 | 4.5580 | 4.8876 | 0.9324x [0.9289, 0.9369] | -6.76% | 0.007812500 |
| lstm | 64 / 128 | 2.1606 | 2.5928 | 0.8966x [0.8415, 0.9966] | -10.34% | 0.000244141 |
| slstm | 64 / 128 | 2.3892 | 2.5658 | 0.9315x [0.9282, 0.9350] | -6.85% | 0.000488281 |

All 120 AB/BA blocks, including anomalies, are retained. Intervals are within-process paired bootstrap; independent restarts remain untested. NumPy CPU independently recomputed 252 layer-state/logit tensor pairs and all fixture hashes at the unchanged numerical budgets.

C128 is a simultaneous-arrival queue served by one worker in 8/4/2 sequential B16/B32/B64 groups; it is not 128 concurrent CUDA streams. Timing includes IDs H2D, embedding, four recurrent layers, readout and logits D2H/host delivery. Initialization, compilation and qualification are excluded. This is not a pretrained checkpoint, generation tok/s, HTTP serving or the complete model matrix. Both arms remain resident; process peak memory cannot establish per-arm savings.

## cuDNN packing diagnostic

Mapped and plain BF16 nn.LSTM both retain four weight storages before/after flatten_parameters and report false is_acceptable. Two forwards produce eight mapped warnings and two plain warnings. Installed Torch2.12.1 CUDNN_TENSOR_DTYPES contains half/float/double; BF16 is rejected by the dtype guard. The environment was not changed. This diagnostic does not qualify optimized cuDNN performance.

Raw archive SHA256: `be8444c01bb97cb7f9e18c05317035c82192ee75d39c22a5371727cb2cc8792d`, 51152692 bytes. All 159 payload and 123 frozen-source hashes pass. Nine execution actors are absent; original GPU/IO locks pass nonblocking exclusive checks and compute apps are empty. Raw evidence is retained. No model downloads, environment updates, process signals or lcpu NFS access.

Client/remote grant JSON formatting differs; all field values match. The controller binds the actual remote grant SHA `8be059a32bdcae8118714a6ef98cfd894a99e975d37fe70d81fa82586c4e2d79`. No scientific job was rerun.
