# Mamba2 queued-serving comparison

The [Mamba2-130M B16/B32 CPU gates](mamba2-highbatch-cpu.md) establish complete-model numerical parity for the pinned B1/P128 TorchScript artifact. The B32 CPU run has one request group and does not satisfy this harness's three-group timing prerequisite. The queued-serving harness is prepared for a same-GPU comparison of the native Transformers Torch path and that artifact; it has not produced GPU timings. It is not a measurement of `mamba_ssm` fused kernels or a FlashRNN candidate.

Before timing, `mamba2_highbatch_gate.py` must pass on the target GPU for all three fixed WikiText request groups at the selected B16, B32 or B64 batch. It compares every full-vocabulary logit step and token, plus all-layer convolution and SSM caches at prefill and final boundaries. `mamba2_serving.py` binds that qualification, checkpoint, input and script hashes, then generates the native golden outputs on the same device. Each arm must match all generated tokens and final caches during its qualification cohort.

The timed cohort starts with all C requests queued at time zero. One worker admits B-sized groups and round-robin decodes 32 tokens. State allocation, prompt transfer, full-model prefill and decode, argmax, queueing and host-visible token delivery are timed. Weight loading, golden generation, tokenization, network and JSON writing are excluded. Twenty or more AB/BA blocks feed the existing paired analyzer. Cohort p99 is not a steady-state service SLO.

| Model and shape | Native Torch tok/s | TorchScript tok/s | Paired speedup [95% CI] | Status |
| --- | ---: | ---: | ---: | --- |
| Mamba2-130M, B16/32/64, C ≥ B and divisible by B, P128/G32 | — | — | — | Same-GPU qualification and E2E timing not run |

The harness records `fast_kernel_qualified=false`. A fused Mamba2, FlashRNN, Hopper or B200 speed claim needs its own correctly qualified baseline and candidate on that device.
