# FP8 recurrent-weight exploration — long-sequence error

This independent experiment packs the complete dense R matrix into E4M3FN
with one FP32 amax scale per gate/head. It unpacks R to BF16 once before
the recurrence. The existing CPU reference keeps FP32 local/history states,
rounds the recurrent hidden operand to BF16, and evaluates BF16-valued input
projections and biases in FP32. Cell equations and dense mixing are unchanged.

There is no FP8 MMA execution, GPU kernel or timing result in this experiment.
The unpacked BF16 weights also remain allocated: packed size is not a claim
about resident memory or register savings for the complete implementation.

All four cells run complete T4096 and T16384 sequences, B1/H1/D64, with two
independent random recurrent-gain configurations (0.1 and 1.0), seed 20261011.
Each baseline/quantized pair has identical inputs and initial states.
LSTM/sLSTM forget biases and GRU update biases include +2 to exercise
persistent memory. Different gain configurations use different random
draws; their comparison does not isolate gain alone.

## Actual long-sequence errors

The table shows T16384. All 16 trajectories are finite; accuracy is
unqualified. The raw JSONL also records prefix lengths 16/128/1024/4096/16384,
per-state relative L2 and final-step errors, including the T4096 cases.

| Cell | R gain | Max hidden error | Hidden relative L2 | Other state maximum absolute errors |
| --- | ---: | ---: | ---: | --- |
| LSTM | 0.1 | 0.002294 | 0.004368 | c: 0.004069 |
| LSTM | 1.0 | 1.329767 | 0.348804 | c: 6.132958 |
| sLSTM | 0.1 | 0.000137 | 0.001142 | c: 0.002566; n: 0.002251; m: 0.000363 |
| sLSTM | 1.0 | 0.001702 | 0.014521 | c: 0.028995; n: 0.032612; m: 0.004528 |
| GRU | 0.1 | 0.000237 | 0.001207 | — |
| GRU | 1.0 | 0.003790 | 0.014163 | — |
| Elman | 0.1 | 0.001720 | 0.002624 | — |
| Elman | 1.0 | 0.163255 | 0.090078 | — |

For the LSTM shape, BF16 R occupies 32,768 bytes; encoded FP8 R plus scales
occupies 16,400 bytes, a **1.998× packed-weight storage ratio**. The same ratio
holds for the other gate counts. The zero-weight packing control reproduces
exact zero without a zero scale. The finite completed run naturally exits 0;
it does not issue a numerical-accuracy PASS.

The large retained LSTM state error demonstrates that weight quantization
alone needs a recurrence-level error contract. It is not evidence about a
trained checkpoint's task quality, nor a reason to widen the earlier BF16
candidate's frozen threshold.

| Scope | Baseline | Candidate | Speedup |
| --- | --- | --- | --- |
| Complete CPU mathematical recurrence | BF16-valued R, FP32 states | FP8-packed R unpacked once to BF16, FP32 states | Not measured |
| GPU FP8 MMA / complete API | Pending | Pending | N/A |
| Trained models / training / high concurrency | Pending | Pending | N/A |

Remaining extension work includes quantized MMA operands, actual register
and shared-memory ownership, packing/scale/launch costs, other scale schemes,
NVFP4, gradients, trained-model quality and complete GPU/E2E comparisons.
Reproduce the fixed 16-case campaign with
`python tools/flashrnn2/fp8_weight_gate.py --output fp8-weight.jsonl` in the
existing Torch runtime. Source/reference hashes and contracts are recorded
in `evidence/fp8-weight-r1.meta.json`.
