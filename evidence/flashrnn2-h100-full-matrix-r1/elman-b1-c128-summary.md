# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs in torch.float16; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Qualification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| elman | cudnn | 1 | 128 | 2283.641 | 3397.879 | 1.492× [1.486, 1.501] | Qualified |
