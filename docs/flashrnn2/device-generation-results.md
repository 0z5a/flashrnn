# Full-model generation with device-local tracing

The complete native Mamba130M model can generate and validate its traced
candidate on the execution device using the already resident checkpoint
transport. This avoids uploading separate large P128 artifacts for every batch.
The initial CPU run passes B1/B2/B4, P128/G32 with independent text prompts.
CUDA execution is queued and has not yet run.

| Batch | Serial + interleaved batch steps | Compared token choices | Logits and both caches | Saved boundary tensors vs original native oracle |
| ---: | ---: | ---: | --- | --- |
| 1 | 192 | 192 | All bitwise equal | 12/12 bitwise |
| 2 | 192 | 384 | All bitwise equal | 12/12 bitwise |
| 4 | 192 | 768 | All bitwise equal | 12/12 bitwise |
| Total | 576 | 1,344 | 1,728 tensor checks pass | 36/36 bitwise |

All 24 layers and 129,135,360 parameters execute in FP32. Each batch starts a
fresh process and constructs its own fixed-shape prefill/decode trace. A seeded
random trace input differs from the three WikiText prompt groups used for
validation. The validation runs serial groups and round-robin interleaving on
one execution stream, with separately allocated native/candidate request caches.
Both paths generate their own greedy tokens; every emitted full-vocabulary
logit vector and both complete cache tensors are compared at every step.
This checks three resident request groups, not a high-concurrency service.

The prompts are copied exactly from the earlier hash-verified native generation
oracles. The independent audit checks all emitted tokens against those original
oracles, every raw case/schedule/step, all source hashes, and both arms' saved
first-group prefill/final tensors. Boundary snapshots cover one group per batch;
other state rows remain runner checks. Serial/interleaved tokens also agree
within each arm. No reference failure or acceptance budget is replaced.

The trace is specific to its batch, prompt length and prefill/decode branch.
The native source's Python condition produces the expected tracer warning,
retained in the logs. This does not establish dynamic-shape trace correctness.
The pinned Transformers slow fallback remains the native reference; its fused
Mamba kernels and the strongest accelerated baselines are still unqualified.

| Required full-model comparison | Native tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| CUDA B1/B2/B4 P128/G32 | Not measured | Not measured | N/A |
| Multi-batch/high-concurrency CUDA service | Not measured | Not measured | N/A |

The controller and all three model children naturally exit 0. Model weights and
the 88,221,625 bytes of boundary snapshots remain task-owned for the unfinished
GPU campaign. A 14,227-byte transport containing the runner, exact prompt inputs
and a separate PTX function diagnostic is frozen but not admitted or uploaded.
Its numerical runs will precede any performance claim.

Reproduction on an existing Torch runtime:

```bash
python tools/flashrnn2/mamba_device_generation.py \
  --model "$MAMBA/prefill-decode-b1-t5-cuda.pt" \
  --source "$PINNED/modeling_mamba.py" --config "$MAMBA/config.json" \
  --inputs "$INPUTS/mamba-device-generation-inputs-r1.json" \
  --batch 1 --device cpu --output "$RESULTS/device-generation-b1.jsonl"
```

Repeat separately for batches 2 and 4. Evidence:
[commands and exits](evidence/mamba-device-generation-cpu-r1-controller.json),
[input provenance](evidence/mamba-device-generation-inputs-r1.json),
[independent audit](evidence/mamba-device-generation-cpu-r1-audit.json),
and [file inventory](evidence/device-generation-evidence-manifest.json).
