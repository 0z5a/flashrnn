# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs in torch.float16; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Qualification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| elman | cudnn | 4 | 128 | 8370.485 | 13437.679 | 1.619× [1.566, 1.673] | Qualified |
