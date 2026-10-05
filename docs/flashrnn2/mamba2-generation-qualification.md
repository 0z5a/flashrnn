# Mamba2 full-model long generation

The complete pinned Mamba2-130M checkpoint (24 layers,128,989,632 parameters)
passes native-versus-TorchScript CPU generation at P128/G32 for B1/B2/B4.
All576 batch-step comparisons and1,344 greedy token choices agree exactly.
This qualifies the finite CPU cases below; CUDA portability, accelerator
baselines and GPU service performance remain untested.

| Actual batch | Trace batch | Prompt / generated | Serial + interleaved batch-steps | Token choices | Max logits/cache error | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 1 | 128 /32 | 192 | 192 | 0 | PASS |
| 2 | 1 | 128 /32 | 192 | 384 | 0 | PASS |
| 4 | 1 | 128 /32 | 192 | 768 | 0 | PASS |

One unchanged B1-traced artifact supplies all three batch sizes. B2/B4 use
an explicit cross-batch probe flag; arbitrary batch sizes and other prompt
lengths are not qualified. The artifact SHA256 is
`17a501f69a8b2457bca46b114066b1dfaf527ad3c90605097939ee493eb0d81a`.
The separate P128 export also passed three independent random-input cases
against native prefill and one-step decode, all outputs bitwise equal.

Native oracles use the existing strict Mamba2 checkpoint mapping and unchanged
Transformers4.54.1 Torch fallback on Torch2.13 CPU. They retain the original
chunk256 and norm-before-gate=false configuration; this is not mamba_ssm CUDA.
The tokenizer is the pinned Mamba130M tokenizer, verified separately from its
weights. Dataset revision, exact selected rows, token hashes, source hashes
and checkpoint provenance are recorded in the linked evidence.

For each B, three fixed WikiText groups run serially and in round-robin order
on one execution stream. Prefill emits token1 and31 decode calls emit the
remaining tokens; generation uses a fixed count without EOS stopping. Every
step compares the complete vocabulary. Convolution/SSM states for all layers
are compared immediately after prefill and at the final step; intermediate
state tensors are not retained. Interleaving tests request isolation, not
parallel CUDA streams or a network service.

The frozen budgets remain logits atol/rtol1e-3 and cache atol/rtol1e-5.
Earlier cached-versus-full-prefix failures remain in
[mamba2-qualification.md](mamba2-qualification.md); these trace/native
comparisons do not replace them.

B1 first-decode cache controls check that the oracle rejects incorrect state:

| Cache input | Largest logit error | Expected acceptance | Control result |
| --- | ---: | --- | --- |
| Correct request | 0 | Accept | PASS |
| Other request convolution cache | 159.827087 | Reject | PASS |
| Other request SSM cache | 164.669586 | Reject | PASS |
| Both caches reset to zero | 247.101166 | Reject | PASS |

| Full-model GPU comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Mamba2 multi-BS/high-concurrency | Not measured | Not measured | N/A |

No CPU diagnostic time is used as a performance result. The CPU controllers
and children naturally returned0. Model files remain needed by the unfinished
GPU, accelerator-baseline and concurrency campaign.

Evidence: [aggregate](evidence/mamba2-generation-r1-summary.json),
[controller](evidence/mamba2-generation-r1-controller.json),
[B1](evidence/mamba2-generation-b1-r1.jsonl),
[B2](evidence/mamba2-generation-b2-r1.jsonl),
[B4](evidence/mamba2-generation-b4-r1.jsonl),
[cache controls](evidence/mamba2-cache-negative-r1.json),
[P128 export](evidence/mamba2-export-b1-p128.meta.json).
Each JSONL has matching `.meta.json` provenance and raw `.log`.
The CPU followup receipt also records a separate RWKV native-oracle job;
that model's implementation and qualification are a subsequent workstream.
