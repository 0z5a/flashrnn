# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs in torch.float16; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Qualification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gru | cudnn | 16 | 64 | 29511.403 | 46104.803 | 1.614× [1.539, 1.754] | Qualified |
