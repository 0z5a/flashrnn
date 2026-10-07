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

A second H20 window completed six more FlashRNN1 comparisons with C128
requests. All six processes exited naturally with code 0, and a second local
NumPy run reproduced the independent audit of all 252 saved tensor pairs.
Each row retains its 20 AB/BA blocks. The throughput is still completed
sequence responses/s for the same BF16 T128, four-layer D64 model.

| Cell | B | C | FlashRNN1 responses/s | FlashRNN2 responses/s | Paired speedup [95% CI] | Throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 16 | 128 | 15,851.919 | 12,937.361 | 0.8165× [0.8136, 0.8197] | −18.35% |
| LSTM | 32 | 128 | 31,161.451 | 25,419.915 | 0.8167× [0.8128, 0.8224] | −18.33% |
| LSTM | 64 | 128 | 59,242.892 | 49,368.017 | 0.8966× [0.8415, 0.9966] | −10.34% |
| sLSTM | 16 | 128 | 14,311.674 | 13,123.953 | 0.9185× [0.9156, 0.9217] | −8.15% |
| sLSTM | 32 | 128 | 28,082.347 | 26,188.495 | 0.9324× [0.9289, 0.9369] | −6.76% |
| sLSTM | 64 | 128 | 53,573.625 | 49,886.388 | 0.9315× [0.9282, 0.9350] | −6.85% |

The LSTM B64/C128 baseline took 5.751836 ms in block 6, producing a
2.156187× block ratio; that block was not removed. The table's geometric-mean
speedup and within-process interval include it. Across the six C128 cases,
the largest qualified tensor difference was 0.0078125 in sLSTM B32/C128.
The [raw rows, audits and paired summaries](../../evidence/flashrnn2-stacked-e2e-h20-c128/RAW_ARCHIVE.md)
are committed with the complete offbox in three SHA-verified parts, including
the qualification tensors.

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
cuDNN baseline. The [packing probe](../../tools/flashrnn2/cudnn_pack_probe.py)
ran on the same H20: mapped and plain BF16 PyTorch LSTMs both retained four
weight storages after `flatten_parameters()`, all their weights failed the
installed `is_acceptable` check, and both triggered compaction warnings.
The installed PyTorch 2.12.1+cu130 dtype guard lists FP16, FP32 and FP64 but
omits BF16. This explains the warning on this runtime; retrying the same
flatten call is not a fix. A matched-FP16 baseline needs a separate numerical
and packed-weight gate before any fair cuDNN speedup is reported.

The next finite run uses `--dtype fp16 --require-packed-cudnn` for both model
arms. Before timing, the runner requires one shared weight storage per cuDNN
layer, accepted cuDNN weights, eval-mode `aten::_cudnn_rnn` dispatch, no
weight-compaction warning and the same per-layer hidden/final-state and logit
numerical gate. The analyzer marks the earlier BF16 row diagnostic even when
replayed from its archived metadata. No FP16 GPU result has been measured yet.

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

Thirteen legal B × C pairs per FlashRNN1 cell remain unmeasured. The cuDNN
B16/C32 row remains diagnostic, and its other 16 pairs are unrun. Haste requires a
separate FP32/FP16 model contract; BF16 results here cannot be assigned to
Haste. A single-CTA GRU/Elman candidate and matched cuDNN model route are now
prepared for the same four-layer D64 workload. They have no GPU correctness or
speed result yet; the source-only addition does not qualify a same-cell
speedup. These fixed-weight synthetic results do not establish
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
