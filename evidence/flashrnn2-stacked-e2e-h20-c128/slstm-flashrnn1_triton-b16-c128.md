# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| slstm | flashrnn1_triton | 16 | 128 | 14311.674 | 13123.953 | 0.919× [0.916, 0.922] |
