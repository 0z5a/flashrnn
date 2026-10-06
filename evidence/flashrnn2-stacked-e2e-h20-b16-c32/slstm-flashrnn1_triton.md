# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| slstm | flashrnn1_triton | 16 | 32 | 14112.876 | 12962.632 | 0.881× [0.807, 0.922] |
