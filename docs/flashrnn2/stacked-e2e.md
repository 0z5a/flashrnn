# Stacked recurrent-model E2E comparison

The [paired runner](../../tools/flashrnn2/stacked_e2e.py) is prepared for a
four-layer D64 BF16 sequence model with a 512-token embedding and full-vocabulary
readout. All arms use the same deterministic weights and request IDs. It times
queued sequence classification from host token IDs to host-visible logits:
transfer, embedding, all recurrent layers, readout and output transfer count.
Model construction, JIT, warmup and numerical qualification are reported
separately. Throughput is completed sequence responses/s, not generated tokens/s.
This synthetic model is a complete-model control and does not replace any
pretrained-checkpoint result in the [18-family matrix](full-model-e2e-speedups.md).

The first H20 window completed B16/C32, T128, four layers and D64 in BF16.
All three processes exited naturally with code 0. The independent NumPy audit
recomputed all 54 hidden-history, final-state and logit tensor pairs from the
saved qualification tensors. Each row retains all 20 AB/BA paired blocks.

| Cell / baseline | Baseline responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Throughput change | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| LSTM / PyTorch cuDNN | 2,596.912 | 12,379.678 | 4.7643× [4.7136, 4.8187] | +376.43% | Diagnostic only: cuDNN weight-compaction warning |
| LSTM / original FlashRNN Triton | 15,590.162 | 12,789.009 | 0.8222× [0.8192, 0.8252] | −17.78% | FlashRNN2 slower at this shape |
| sLSTM / original FlashRNN Triton | 14,112.876 | 12,962.632 | 0.8810× [0.8067, 0.9222] | −11.90% | FlashRNN2 slower at this shape |

The speedup is the geometric mean of the 20 paired latency ratios, so it need
not equal the ratio of the two displayed median response rates. The intervals
bootstrap paired blocks within one process; independent-start variance was not
measured. The sLSTM block 3 ratio of 0.387053 remains in the analysis. C32
means two B16 groups arriving together and served by one worker in order; it
does not mean 32 concurrent CUDA streams. Both arms' weights remain resident,
so the recorded process peak is not a single-arm memory comparison.

The cuDNN baseline ran in eval mode and dispatched `aten::_cudnn_rnn`, but its
log warns that non-contiguous weights may be compacted at every call. This
invalidates the 4.7643× ratio as a comparison against a qualified optimized
cuDNN baseline. A later run must prove packed-weight reuse before reporting
that comparison as fair.

The [raw offbox archive](../../evidence/flashrnn2-stacked-e2e-h20-b16-c32/own-offbox-raw.tar.gz)
has SHA256 `2b77b63631e8a06071325dea3b2816d256886863fdbbfbd9d783b6d252d0a75c`.
Its 143 payload files and the frozen 122-file source manifest passed bytewise
SHA checks. The [independent review](../../evidence/flashrnn2-stacked-e2e-h20-b16-c32/independent-review-r6.json)
and [whole-device handback](../../evidence/flashrnn2-stacked-e2e-h20-b16-c32/WHOLE.json)
record the numerical gate, natural exits, empty compute-app list and original
GPU/IO lock checks. Per-case JSON summaries and Markdown rows are alongside
the archive.

The 17 legal `(batch, concurrent requests)` pairs are `(1,1/8/32/64/128)`,
`(4,8/32/64/128)`, `(16,32/64/128)`, `(32,32/64/128)` and `(64,64/128)`.
Each run first compares every layer's hidden history and final state plus final
logits, then uses at least 20 paired AB/BA blocks. A result is reportable only
when the complete JSONL and metadata say `PASS`, the actual cuDNN dispatch is
observed in eval mode where applicable, both arms use the admitted GPU UUID, and the
[analyzer](../../tools/flashrnn2/analyze_stacked_e2e.py) produces a 95% paired
bootstrap interval. The untimed qualification saves every request group's IDs,
both arms' per-layer hidden/final tensors and logits in a SHA-bound `.pt` file
for independent numerical recomputation. Each group has distinct token IDs.

The other 16 legal B × C pairs per row remain unmeasured. Haste requires a
separate FP32/FP16 model contract; BF16 results here cannot be assigned to
Haste. GRU and Elman need their own FlashRNN2 GPU candidates before a
same-cell speedup exists. These fixed-weight synthetic results do not establish
pretrained-checkpoint generation throughput for the 18-family matrix.

For one admitted run, the command shape is:

```sh
PYTHONPATH=application python application/tools/flashrnn2/stacked_e2e.py \
  --cell lstm --baseline cudnn --batch 16 --concurrency 32 \
  --expected-gpu-uuid GPU-d952768b-40d1-0de6-38f2-6725427f8214 \
  --output results/lstm-cudnn-b16-c32.jsonl
```

The reviewer should use the machine receipt's current boot, GPU UUID, lock
identity and source SHA, not infer validity from this example command. Further
measurements require new finite GPU+IO grants, complete offbox and independent
audit; the first window was handed back.
