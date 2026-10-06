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

| Cell | Same-equation baseline | FlashRNN2 candidate | Valid B × C pairs | H20 measured speedup |
| --- | --- | --- | ---: | ---: |
| LSTM | PyTorch/cuDNN | Triton persistent | 17 | Unmeasured |
| LSTM | Original FlashRNN `triton_fused` | Triton persistent | 17 | Unmeasured |
| sLSTM | Original FlashRNN `triton_fused` | Triton persistent | 17 | Unmeasured |

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

The first bounded H20 package targets B16/C32 for the three rows above. It must
receive a fresh GPU+IO window and be staged with an immutable source manifest;
the earlier 38-case correctness grant was consumed. This runner's GPU compile,
model parity and timing remain untested. Haste requires a separate FP32/FP16
model contract; BF16 results here cannot be assigned to Haste. GRU and Elman
need their own FlashRNN2 GPU candidates before a same-cell speedup exists.

For one admitted run, the command shape is:

```sh
PYTHONPATH=application python application/tools/flashrnn2/stacked_e2e.py \
  --cell lstm --baseline cudnn --batch 16 --concurrency 32 \
  --expected-gpu-uuid GPU-d952768b-40d1-0de6-38f2-6725427f8214 \
  --output results/lstm-cudnn-b16-c32.jsonl
```

The reviewer should use the machine receipt's current boot, GPU UUID, lock
identity and source SHA, not infer validity from this example command. Real
measurements will replace `Unmeasured` only after offbox and independent audit.
