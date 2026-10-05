# RWKV7 complete-checkpoint qualification

The complete RWKV7 Goose World2.9 0.4B checkpoint executes all 24 layers in
native FP32 recurrent inference. Its batched TorchScript implementation matches
the native CPU oracle exactly at B1. B2/B4 match every greedy token and pass
the logit budget, but fail the existing shift and matrix cache budgets.
GPU acceleration, longer generation and high-concurrency E2E remain untested.

The checkpoint is
`RWKV/RWKV7-Goose-World2.9-0.4B-HF@e94655a9fad2c8da9f25aba575d8f0fdedc05931`.
All 11 functional files (902,729,415 bytes) passed their immutable Git-blob or
LFS checksums. The weights contain 795 BF16 tensors / 450,767,872 elements;
the config dtype string does not describe their actual storage dtype.
Inference converts these weights to FP32. The two layer-0 pre-norm vectors
are folded into the embedding exactly as in the official constructor;
450,765,824 elements remain as inference buffers.

The native adapter selects the unchanged arithmetic bodies of the official
`forward`, `time_mixing__` and `channel_mixing__` by hash-checked AST extraction.
Only the forward name/decorator and execution wrapper change. It preserves
the native coefficient `0.606531`, state orientation and layer-0 unused-v
aliases. All 795 inverse-mapped keys and transpose flags round-trip through
the pinned FLA converter; there are no collisions. The batched implementation
performs batch-shaped projections, with private functional shift and matrix
states. It does not serially invoke the scalar model once per batch row.

Source pins:
[RWKV-LM native RNN](https://github.com/BlinkDL/RWKV-LM/blob/20be0f8cf0bdf506d8c0030c87a9a3416f95b143/RWKV-v7/rwkv_v7_demo_rnn.py),
[FLA converter](https://github.com/fla-org/flash-linear-attention/blob/17dd5662554d46b6bcb1d1ff728cebb461c9aef9/utils/convert_from_rwkv7.py).
FLA recurrent/chunk execution is not qualified by this native CPU comparison.

Three independently executed native prompts come from the pinned WikiText
validation rows 1, 3 and 5, tokenized with the checkpoint's tokenizer.
P5/G4 uses fixed-count generation without EOS stopping: prefill emits the
first token, then three recurrent calls emit the remainder. The native oracle
retains full-vocabulary logits and all 72 state tensors at prefill/final.
Three groups per batch circularly select these prompts; B4 repeats one prompt
in a separate row, so it is not four distinct prompt samples.

| Batch | Batch-step comparisons | Token choices | Failed batch-steps | Max logit error | Max shift error | Max matrix error | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 12 | 12 | 0 | 0 | 0 | 0 | PASS |
| 2 | 12 | 24 | 6 | 1.01089e-4 | 1.52588e-4 | 3.05176e-5 | Cache parity FAIL |
| 4 | 12 | 48 | 6 | 1.01089e-4 | 1.52588e-4 | 3.05176e-5 | Cache parity FAIL |

All 84 token choices agree. Complete vocabulary logits are compared at every
step; caches are compared only at prefill/final. For each of B2/B4, six shift
checks and three matrix checks fail. Budgets were frozen before execution:
logits `atol=rtol=1e-3`, caches `atol=rtol=1e-5`. They are unchanged.
The model is serialized despite the failure so the exact artifact can be
diagnosed; successful serialization is not a qualification pass.

The additional B2 diagnosis uses the first two prompts. At prefill, 5 shift
elements fail and no matrix elements fail. At the final state, 161 shift and
3 matrix elements fail. The worst normalized cache error is 2.8643 times its
budget at shift coordinate `[1,18,0,15]`.

| First-layer projection, same row input | B1 versus B2 max error | B1 versus isolated FP64 | B2 versus isolated FP64 |
| --- | ---: | ---: | ---: |
| Receptance | 4.17233e-7 | 4.86290e-8 | 4.30653e-7 |
| Key | 8.34465e-7 | 1.24387e-7 | 9.58852e-7 |
| Value | 4.17233e-7 | 5.79331e-8 | 4.63275e-7 |

Batch-shaped linear projections already differ on identical inputs in the
first layer. B1 is closer to FP64 in these three probes. This is evidence of
batch-dependent rounding, not proof that it explains every later state
difference. FP64 here covers only three projections, not the whole model;
the underlying BLAS dispatch was not profiled. No revised acceptance or
speed claim follows from the diagnosis.

The first diagnostic attempt failed before forward because indexing a loaded
ScriptModuleList was unsupported. That source, log and natural exit 1 remain
in the evidence. The second attempt changes only layer access to
`next(model.layers.children())` and naturally exits 0. The export itself
naturally exits 1, reflecting its 12 failed comparisons.

| Complete-model GPU comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Native versus batched RWKV7, multiple BS/high concurrency | Not measured | Not measured | N/A |
| FLA recurrent/chunk and other accelerator baselines | Not measured | Not measured | N/A |

The native oracle artifact is 42,201,146 bytes, SHA256
`8eeb8d4a0764f3ae27b95d3b5092141cc1eea5abb77846a31830f3aeae5eafd8`.
The scripted artifact is 1,803,293,512 bytes, SHA256
`54dd315e61a65a3e05b9d7744fb6ed9b40812475bffcba13ac86f5391eb3ba8a`;
its size and hash were independently checked after execution. Existing
Torch 2.13 CPU was reused without environment changes. Model files remain
needed by the unfinished campaign.

To reproduce with the pinned local checkpoint, sources and dataset:

```bash
python tools/flashrnn2/rwkv7_native_goldens.py \
  --model "$MODEL" --native-source "$NATIVE_SOURCE" \
  --converter-source "$CONVERTER_SOURCE" --dataset "$WIKITEXT" \
  --output "$RESULTS/native-generation-b1-p5-g4.pt"
python tools/flashrnn2/export_rwkv7.py \
  --model "$MODEL" --native-source "$NATIVE_SOURCE" \
  --converter-source "$CONVERTER_SOURCE" \
  --goldens "$RESULTS/native-generation-b1-p5-g4.pt" \
  --output "$RESULTS/prefill-decode-fp32-r1.pt" \
  --results "$RESULTS/rwkv7-batched-script-r1.jsonl"
# The export is expected to exit 1 for the retained CPU parity failures.
python tools/flashrnn2/rwkv7_batch_diagnostic.py \
  --model "$RESULTS/prefill-decode-fp32-r1.pt" \
  --goldens "$RESULTS/native-generation-b1-p5-g4.pt" \
  --output "$RESULTS/rwkv7-batch-diagnostic-r2.json"
```

`$MODEL/verified-manifest.json` is the published
[checkpoint manifest](evidence/rwkv7-checkpoint-manifest.json). The native and
converter source links above supply the two hash-pinned files. `$WIKITEXT`
uses revision `b08601e04326c79dfdd32d625aee71d232d685c3` and SHA256
`204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`.
Use a fresh results directory; completed outputs are not overwritten.

Evidence: [independent result audit](evidence/rwkv7-result-audit-r1.json),
[raw comparison rows](evidence/rwkv7-batched-script-r1.jsonl),
[native metadata](evidence/rwkv7-native-oracles-r1.summary.json),
[export metadata](evidence/rwkv7-batched-script-r1.summary.json),
[numerical diagnosis](evidence/rwkv7-batch-diagnostic-r2.json),
[all key mappings](evidence/rwkv7-native-key-map.jsonl),
[evidence index](evidence/rwkv7-evidence-index.json).
The index identifies unchanged raw files and explicitly described metadata
normalization, with original hashes; original metadata remains retained.
