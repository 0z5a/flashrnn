# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lstm | cudnn | 16 | 32 | 2596.912 | 12379.678 | 4.764× [4.714, 4.819] |
