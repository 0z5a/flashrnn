# RetNet projection-row numerical control

The [original full-checkpoint RetNet run](retnet-reference.md) retains nine
BS=1 recurrent-state failures. Keeping the same 1,351,727,104-parameter
checkpoint, prompts, retention update, rotary positions, and comparison budgets,
this control applies each `torch.nn.functional.linear` operation to one input
row at a time. It changes floating-point reduction order. Across BS=1/2/4,
all 36 cached-versus-full-prefix steps and 84 generated token choices pass;
both logits and every checked recurrent state are bitwise identical.

| Batch | Saved steps | Logit pairs passing | All-layer state steps passing | Matching tokens | Maximum logit/state error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 12/12 | 12/12 | 12/12 | 0 / 0 |
| 2 | 12 | 12/12 | 12/12 | 24/24 | 0 / 0 |
| 4 | 12 | 12/12 | 12/12 | 48/48 | 0 / 0 |

The [independent audit](evidence/retnet-numerics/retnet-row-linear-recovery-complete-audit.json)
recomputes 36 full-vocabulary logit pairs, 84 token choices, and both complete
case-0 final caches for each batch: 72 layer-state pairs. It verifies all nine
saved snapshot hashes and sizes, source and checkpoint-source hashes, prompts,
row coverage, per-layer counters and token IDs against the original run. The
remaining intermediate states are runner comparisons. Fixed budgets remain
`atol=rtol=1e-3` for logits and `atol=rtol=1e-5` for states.

The first diagnostic process naturally exited 1 after writing 16 passing rows:
`torch.save` failed with an iostream error while saving the BS=2/case-0
snapshot. Its three complete BS=1 snapshots were independently audited (12
logit pairs, 12 tokens, 24 final layer states). The unregistered, invalid ZIP
partial file had no readers; its size and SHA256 were recorded before removing
only that file. The underlying I/O cause is undetermined. A fresh run restricted
to BS=2/4 naturally exited 0 and saved six complete snapshots. The combined
audit excludes the four BS=2 rows from the incomplete first run and uses all
24 rows of the second run. Both run receipts and the write error remain in the
[evidence inventory](evidence/retnet-numerics/manifest.json).

The diagnostic intentionally alters arithmetic. It does not replace the
original nine failures, qualify native FLA kernels, establish a fix for model
quality, or measure performance. The original reference was published
separately and retains its failed exit. The new gate accepts a selected list of
batches and writes snapshots through a `.pt.partial` file before renaming, so a
failed write cannot appear under a complete snapshot name.

```sh
python tools/flashrnn2/retnet_row_linear_reference.py \
  --model "$RETNET_CHECKPOINT" --source "$PINNED_RETNET_SOURCES" \
  --common "$PINNED_GLA_SOURCES" --output "$RESULTS/retnet-row-linear.jsonl" \
  --batches 1 2 4
```

The executed audit script is in the inventory; it expects the local execution
workspace's checkpoint, original snapshots and diagnostic snapshots. Binary
snapshots remain there with per-file hashes in the published metadata. All
weight files remain needed for native GPU and model E2E qualification.

| Required performance comparison | Original tok/s | Changed tok/s | Speedup |
| --- | ---: | ---: | ---: |
| CPU numerical control | — | — | Not measured |
| Native accelerated RetNet GPU E2E | — | — | Not measured |
| High-concurrency, multi-batch GPU E2E | — | — | Not measured |
