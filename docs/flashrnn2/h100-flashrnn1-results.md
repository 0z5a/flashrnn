# H100 FlashRNN1 sLSTM complete-model gate

One FP16 four-layer sLSTM sequence classifier compared FlashRNN2 with the pinned original FlashRNN1 Triton forward kernel. The [baseline adapter](../../flashrnn/flashrnn2/upstream_triton.py) performs explicit layout conversion and invokes the unchanged [upstream kernel](../../flashrnn/flashrnn/triton_fused/slstm_fw.py); this is an original-kernel comparison, not a claim about the stock FlashRNN1 public wrapper.

Both arms used the same fixed random weights and request IDs, D64, T32, a 512-token embedding and readout, batch 16 and 32 queued requests. Timed work includes IDs transfer, embedding, four recurrent layers, readout, host-visible logits and queued groups. Construction, compilation, warmup and numerical qualification were excluded.

| Cell | Batch | Concurrent requests | FlashRNN1 responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| sLSTM | 16 | 32 | 30,998.198 | 37,108.511 | 1.177× [1.150, 1.200] |

The run completed 20 alternating AB/BA paired blocks. An independent CPU NumPy audit compared all 18 saved logit, layer-history and final-state tensor pairs against the qualification file's SHA256; the largest absolute error was 0.00006104. The [raw timing, redacted metadata, qualification tensor and per-case analysis](../../evidence/flashrnn2-h100-flashrnn1-r1/slstm-b16-c32.jsonl) are committed with the [numerical audit](../../evidence/flashrnn2-h100-flashrnn1-r1/slstm-b16-c32-audit.json) and [natural-exit receipt](../../evidence/flashrnn2-h100-flashrnn1-r1/control-audit.json). The [NumPy auditor](../../tools/flashrnn2/audit_stacked_numpy.py) can replay the saved tensor comparison without importing PyTorch or using a GPU.

The runtime was PyTorch 2.4.0a0, CUDA 11.8 and Triton 3.0.0. C requests arrived at time zero and one worker served them in B-sized groups; C does not mean C simultaneous CUDA streams. The interval bootstraps paired blocks within one process, so independent-start variance is unmeasured. These are synthetic sequence responses/s, not pretrained-checkpoint generation tokens/s. The remaining legal LSTM and sLSTM batch × concurrency cases are still unmeasured.
