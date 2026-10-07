# H100 FlashRNN1 complete-model comparison

FlashRNN2 was compared with the pinned original FlashRNN1 Triton LSTM and sLSTM forward kernels over all 34 legal batch × concurrency cells. The [baseline adapter](../../flashrnn/flashrnn2/upstream_triton.py) converts layouts explicitly and invokes the unchanged [LSTM](../../flashrnn/flashrnn/triton_fused/lstm_fw.py) or [sLSTM](../../flashrnn/flashrnn/triton_fused/slstm_fw.py) kernel. This qualifies the original kernels through this adapter; it does not measure the stock FlashRNN1 wrapper.

Each arm ran the same FP16 four-layer sequence classifier with fixed random weights and request IDs, D64, T32, and a 512-token embedding and readout. All C requests arrived at time zero; one worker served them in B-sized groups. Timed work includes IDs transfer, embedding, every recurrent layer, readout, host-visible logits and queued groups. Model construction, compilation, warmup and numerical qualification are excluded. Throughput is complete sequence responses/s.

| Cell | Batch | Concurrent requests | FlashRNN1 responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstm | 1 | 1 | 1,594.928 | 2,710.621 | 1.685× [1.620, 1.743] |
| lstm | 1 | 8 | 1,825.251 | 3,193.082 | 1.744× [1.731, 1.756] |
| lstm | 1 | 32 | 1,890.598 | 3,343.081 | 1.766× [1.759, 1.773] |
| lstm | 1 | 64 | 1,903.812 | 3,382.557 | 1.776× [1.771, 1.780] |
| lstm | 1 | 128 | 1,912.634 | 3,388.375 | 1.771× [1.768, 1.774] |
| lstm | 4 | 8 | 6,457.134 | 11,491.372 | 1.771× [1.732, 1.803] |
| lstm | 4 | 32 | 7,100.782 | 12,650.444 | 1.780× [1.767, 1.792] |
| lstm | 4 | 64 | 7,210.320 | 13,052.688 | 1.808× [1.794, 1.821] |
| lstm | 4 | 128 | 7,436.108 | 13,352.404 | 1.792× [1.787, 1.797] |
| lstm | 16 | 32 | 32,603.687 | 45,890.336 | 1.403× [1.381, 1.424] |
| lstm | 16 | 64 | 34,444.360 | 48,258.836 | 1.395× [1.375, 1.413] |
| lstm | 16 | 128 | 36,264.474 | 50,451.005 | 1.392× [1.384, 1.400] |
| lstm | 32 | 32 | 59,495.949 | 84,396.632 | 1.405× [1.353, 1.453] |
| lstm | 32 | 64 | 63,636.433 | 89,218.368 | 1.382× [1.351, 1.409] |
| lstm | 32 | 128 | 68,011.129 | 94,148.907 | 1.375× [1.358, 1.392] |
| lstm | 64 | 64 | 115,269.454 | 157,049.643 | 1.362× [1.316, 1.408] |
| lstm | 64 | 128 | 125,453.522 | 172,550.048 | 1.362× [1.339, 1.382] |
| slstm | 1 | 1 | 1,513.747 | 2,159.681 | 1.422× [1.364, 1.477] |
| slstm | 1 | 8 | 1,737.804 | 2,616.056 | 1.504× [1.493, 1.513] |
| slstm | 1 | 32 | 1,816.920 | 2,748.474 | 1.520× [1.501, 1.550] |
| slstm | 1 | 64 | 1,836.252 | 2,767.152 | 1.504× [1.500, 1.508] |
| slstm | 1 | 128 | 1,828.234 | 2,767.277 | 1.512× [1.508, 1.517] |
| slstm | 4 | 8 | 6,114.119 | 9,338.252 | 1.524× [1.500, 1.547] |
| slstm | 4 | 32 | 6,760.853 | 10,482.280 | 1.549× [1.539, 1.560] |
| slstm | 4 | 64 | 6,861.807 | 10,728.667 | 1.565× [1.558, 1.573] |
| slstm | 4 | 128 | 7,020.592 | 10,987.480 | 1.559× [1.550, 1.566] |
| slstm | 16 | 32 | 30,998.198 | 37,108.511 | 1.177× [1.150, 1.200] |
| slstm | 16 | 64 | 32,666.928 | 39,822.375 | 1.215× [1.201, 1.229] |
| slstm | 16 | 128 | 33,769.200 | 41,125.487 | 1.217× [1.209, 1.225] |
| slstm | 32 | 32 | 57,234.031 | 67,569.462 | 1.185× [1.158, 1.221] |
| slstm | 32 | 64 | 60,356.275 | 73,020.948 | 1.206× [1.194, 1.216] |
| slstm | 32 | 128 | 64,164.168 | 78,247.300 | 1.218× [1.206, 1.230] |
| slstm | 64 | 64 | 110,909.685 | 130,491.058 | 1.178× [1.151, 1.212] |
| slstm | 64 | 128 | 116,133.407 | 138,098.801 | 1.188× [1.177, 1.199] |

All 34 cases passed 20 alternating AB/BA paired blocks each (680 blocks), with FlashRNN2 faster in every case. Paired speedups span 1.177×–1.808×. Independent CPU NumPy replay verified 5,670 saved layer-history, final-state and logit tensor pairs; maximum absolute error was 0.00195312. The first sLSTM B16/C32 case ran in a separate earlier H100 window; each ratio and interval uses its own paired run.

[First-case evidence](../../evidence/flashrnn2-h100-flashrnn1-r1/slstm-b16-c32.jsonl) and the [33-case evidence index](../../evidence/flashrnn2-h100-flashrnn1-r2/aggregate.json) include raw paired times, redacted metadata, saved qualification tensors, per-case numerical audits and SHA256 manifest. The [NumPy auditor](../../tools/flashrnn2/audit_stacked_numpy.py) replays tensors without importing PyTorch or using a GPU; the [timing analyzer](../../tools/flashrnn2/analyze_stacked_e2e.py) reproduces each speedup and within-process 95% bootstrap interval.

Hardware: one NVIDIA H100 per measured case. Runtime: PyTorch 2.4.0a0, CUDA 11.8, Triton 3.0.0. The intervals bootstrap paired blocks within one process; independent-start variance was not measured. These fixed-weight synthetic sequence responses are not pretrained-checkpoint generation tokens/s, and concurrent requests do not imply simultaneous CUDA streams.
