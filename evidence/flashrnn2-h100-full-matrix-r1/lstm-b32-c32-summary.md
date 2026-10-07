# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs in torch.float16; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Qualification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| lstm | cudnn | 32 | 32 | 43954.543 | 68713.296 | 1.572× [1.528, 1.613] | Qualified |
