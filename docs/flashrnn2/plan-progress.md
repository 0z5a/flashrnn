# Execution-plan coverage

This ledger tracks the 2026-10-05 Plan A and the broader model campaign.
Partial coverage is not phase completion. The execution plan's proposed
commands and file organization remain targets where the corresponding
implementation is absent.

| Workstream | Verified evidence | Remaining work |
| --- | --- | --- |
| P0 source and capability pinning | Pinned upstream, source map, existing-runtime probes, CUDA13 GPU-info and original seven-unit alternating LSTM BF16 build/import; five native Triton cases executed (four PASS, one retained failure), failed normalizer midpoint localized | Native CUDA first-forward error218 InvalidPtx confirmed; exact driver JIT diagnosis, device query and recurrence qualification; native Triton internal FP32 diagnosis; complete backend support matrix |
| P0 numerical contracts | FP64 reference and all-input CPU gradients; explicit backend dtype/initialization differences; full-model Mamba FP64 rounding diagnosis | Independent per-state low-precision budgets, gradient budgets, trained-checkpoint tolerances |
| P1 single CTA | Inferred Triton and explicit-layout Gluon forward; D64/D128 tested as supported | Broader inputs, long T, graph replay, resource/performance sweeps |
| P2 output R tiling | Stepwise kernel-boundary synchronization; `persistent=false` recorded | Larger-D campaign and legally synchronized persistent cross-CTA implementation |
| P3 Hopper overlap | No implementation or hardware measurement | Implement and validate on appropriate hardware |
| P4 Blackwell operand/TMEM | SM120 R-as-A offline exploration and register-layout improvement | SM100/103 complete recurrences, two orientations, TMEM ledger and measured ablations; SM120 is not substitute evidence |
| P5 readiness/reuse | Repeat, independent streams and changed-R checks on existing prototypes | Conservative readiness protocol, generations, reuse hazards, negative controls, sanitizer |
| P6 backward/training | CPU mathematical/autograd reference and torch.nn gradient mappings | Candidate CUDA dWx/dR/db/dinitial, both losses, intermediate checkpoints, optimizer/loss/task trajectories |
| P7 selector | Fixed cases and holdout definitions | Finite legal search, selector/cache implementation, holdout and oracle regret |
| P8 performance/evidence | One L1 LSTM shape, 20 paired blocks against actual cuDNN; longer generation published in Draft11; queued-request CPU qualification and CUDA failure evidence | Native strongest baselines, fresh starts, all shapes, ablations and full model/service evidence |
| C low precision | Draft10: four cells, 16 complete CPU T4096/T16384 error comparisons, FP32 states and FP8-packed R; large LSTM state errors retained, no accuracy acceptance | Actual GPU quantized MMA/ownership/cost, NVFP4, gradients, trained-model quality and complete E2E |
| D Newton/DEER/ParaRNN | Independent Draft9: all four cells retain dense mixing; 12 normal CPU cases, four sLSTM initial-state cases and four pinned ParaRNN dense-reduction forward cases pass; four convergence failures retained | GPU complete recurrence/training, scalable Jacobian organization, full DEER/ParaRNN package baselines, trained-model quality and E2E |
| Full model campaign | Complete Mamba130M and Mamba2 CPU checkpoint execution at B1/2/4; retained logit failures; full Mamba130M TorchScript CPU parity on B1/2/4 P128 with 576 serial/interleaved generation comparisons and three rejected cache faults; three Mamba CUDA cases execute, with two retained prefill SSM failures; Mamba2 CPU P128/G32 B1/2/4 also passes576 serial/interleaved comparisons and three rejected cache faults; complete RWKV7 CPU native P5/G4 executes, B1 batched parity exact but B2/B4 caches fail; Mamba remote CPU also retains one native-oracle failure; independent native Torch adapter short3 and P128/G32 B1/2/4 CPU576 steps are bitwise exact | Native GPU backends, CUDA numerical qualification, remaining families, multi-BS/high-concurrency GPU E2E and completed-model cleanup |

The current `bench.py` implements an L1 forward pilot. It does not yet
implement all K1/A1/L1/T1/S1 modes or the entire CLI contract in plan section
14. `analyze.py` implements paired statistics, not selector regret/ablation
reporting. `test_reference.py` and `test_torch_layer.py` cover part of the
proposed layout/forward/backward tests; they do not replace the remaining
dispatch, lifecycle, GPU backward and sanitizer suites.

The downloaded checkpoints remain task-owned until their GPU and concurrency
campaigns complete. Current resources are shared with finite peer jobs;
there is no process termination, package installation or lcpu NFS access.

Full-model [FP64 rounding diagnosis](mamba-fp64-diagnosis.md) finds six original
local FP32 SSM elements outside the unchanged budget, while saved remote CPU
and CUDA cases fit it. This additional calculation does not replace the
original native-oracle failures or complete CUDA qualification.

[Native/Script queued serving](mamba-native-serving.md) passes all24 CPU rows,
B1/B2/B4 and C up to128, with1022 request executions/32704 output tokens.
This completes CPU qualification for these fixed cohorts only; GPU, HTTP,
fresh-process performance and all remaining model families are still required.
