# Complete synthetic stacked-model sequence inference

Fixed random weights and request IDs in torch.float16; this is a full 4-layer model workload, not a pretrained checkpoint or autoregressive generation result. Throughput is completed sequence responses/s. Timed scope includes host-to-device IDs, embedding, recurrence, readout, device-to-host logits and queued groups. Compilation, model construction and qualification are excluded.

| Cell | Baseline | Batch | Concurrent requests | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Qualification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gru | cudnn | 32 | 128 | 56613.749 | 88676.117 | 1.566× [1.552, 1.578] | Qualified |
