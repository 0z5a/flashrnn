# TiRex 35M full-forecast E2E: first numerical gate failure

The first H100 BF16 run used the pinned 35M checkpoint, the official TiRex
forecast API, Torch 2.4.0a0+gite3b9b71, Triton 3.0.0, and the B2-only
FlashRNN2 dispatch. It queued 32 deterministic 128-point series in groups of
two and requested 64 forecast points. The candidate failed the **first
untimed output comparison**, before warmup or any paired timing block. The
scientific process, wrapper, and GPU step each exited with code 1.

| Case | Candidate path | Numerical gate | Completed paired blocks | Forecast/s | Speedup and 95% CI |
|---|---|---|---:|---:|---:|
| B2/C32 | FlashRNN2 | Fail: 194/1,152 elements in one quantile tensor | 0/20 | Not measured | Not measured |
| B2/C64 | FlashRNN2 | Not run | 0/20 | Not measured | Not measured |
| B2/C128 | FlashRNN2 | Not run | 0/20 | Not measured | Not measured |
| B1/C32 | Original Torch fallback | Not run | 0/20 | Not measured | Not measured |
| B4/C32 | Original Torch fallback | Not run | 0/20 | Not measured | Not measured |

The comparison kept `atol=rtol=1e-4`. The traceback reports greatest absolute
difference 0.00205660 and greatest relative difference 0.0271646. The runner
did not record which of the 16 queued groups failed or save the new reference
and candidate arrays. Therefore this failure cannot be independently replayed
numerically from the r1 evidence. The complete private failure archive and all
180 payload hashes were checked; its SHA256 is
`13b9bffc8fd251ca2f4b77e0fbc6cfa3e1b17bddb219c6676ce9e9bfb375122c`.
No throughput or confidence interval is inferred from this failed run.

The earlier [r5 qualification](tirex-gpu-r5-results.md) remains a valid
381/381 result for its fixed inputs. Its B2 forecast used only the first two
series, so it did not establish correctness for every series in this C32
queue. The revised diagnostic first repeats the unhooked
baseline-all-groups then candidate-all-groups order of the failed E2E run
and saves every final output. It then captures all 16 groups' cell outputs
and four states across 12 blocks and two patches with forward hooks.
Separate comparisons measure whether the hooks change either arm's final
output. The earlier hook-only diagnostic was not run. The [distinct r2 GPU
diagnostic](tirex-e2e-r2-diagnostic-results.md) saved and independently
audited all 2,048 tensor pairs: eight original unhooked forecast pairs fail,
and no hook-effect pair fails. It has no timing result.
