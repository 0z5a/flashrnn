# TiRex 35M full-forecast diagnostic r2: numerical failure

The H100 BF16 diagnostic used the pinned 35M checkpoint, B2/C32 queued as
16 groups, 128 context points, 64 forecast points, and the B2 FlashRNN2
recurrence. It saved the original unhooked Torch forecast for all groups,
then the original unhooked candidate forecast for all groups. A separate
pass captured 12 blocks × 2 patches × cell output and four recurrent states
with hooks on both arms; both-arm controls checked whether hooks changed the
final forecast. The original numerical budget remained
`1e-4 × (1 + |reference|)`.

The scientific child, wrapper, and GPU step naturally exited 1. The complete
62,042,100-byte private archive (SHA256
`d1b473abdb5d638c395eee6a7b5db11c7772b467971b6968b8127db191e2af96`)
contains the 214,756,410-byte raw tensor PT (SHA256
`e10f8e641ca180547456c63ebe1fff9d69b2b4cf35377552d5de0ea6a82b79a3`);
all 185 payload hashes were checked. The [producer report](evidence/tirex-e2e-diagnostic-r2/producer.json)
and [independent NumPy audit](evidence/tirex-e2e-diagnostic-r2/independent-audit.json)
cover all 2,048 expected tensor pairs and 26,820,608 elements. The audit
found 215 failing pairs, no missing or unexpected pairs, and no disagreement
with the producer's pass/fail flags. The [scope summary](evidence/tirex-e2e-diagnostic-r2/scope-summary.json)
records the category counts.

| Scope | Compared pairs | Failed pairs |
|---|---:|---:|
| Original unhooked forecast | 32 | 8 |
| Hooked forecast | 32 | 8 |
| Hook effect on both arms | 64 | 0 |
| Hooked cell output and four states | 1,920 | 199 |
| **Total** | **2,048** | **215** |

The table below shows all 16 groups. "First state mismatch" is the earliest
layer/patch with any failing cell output or state comparison. A matching final
forecast does not imply that all intermediate states matched.

| Group | Original unhooked forecast failures | Hooked forecast failures | Cell/state failures | First state mismatch |
|---|---|---:|---:|---|
| G0 | Pass | 0/2 | 0/120 | — |
| G1 | median, quantiles | 2/2 | 42/120 | L3/P1 |
| G2 | Pass | 0/2 | 0/120 | — |
| G3 | Pass | 0/2 | 0/120 | — |
| G4 | median, quantiles | 2/2 | 40/120 | L4/P1 |
| G5 | Pass | 0/2 | 0/120 | — |
| G6 | quantiles | 1/2 | 10/120 | L10/P1 |
| G7 | quantiles | 1/2 | 15/120 | L9/P1 |
| G8 | Pass | 0/2 | 15/120 | L9/P0 |
| G9 | Pass | 0/2 | 0/120 | — |
| G10 | Pass | 0/2 | 0/120 | — |
| G11 | median, quantiles | 2/2 | 77/120 | L3/P0 |
| G12 | Pass | 0/2 | 0/120 | — |
| G13 | Pass | 0/2 | 0/120 | — |
| G14 | Pass | 0/2 | 0/120 | — |
| G15 | Pass | 0/2 | 0/120 | — |

G1 reproduces the first E2E r1 quantile failure: 194 of 1,152 quantile
elements exceed the same budget, with maximum absolute error 0.00205660.
G8 has 15 intermediate failures while its final forecast passes. The largest
normalized error among all comparisons is 162.602 times the allowed budget,
in a G11 normalizer tensor. The earliest observed failing comparisons are
at G1 L3/P1 and G11 L3/P0; this identifies where divergence becomes visible,
not which arithmetic operation causes it. Since both-arm hook-effect controls
pass and the original unhooked path fails, the captured hooks do not explain
this run's output failure. This does not establish a general cause for other
inputs or implementations.

The prior [r5 gate](tirex-gpu-r5-results.md) remains 381/381 passing on its
fixed first-two-series input. It does not qualify the new groups in this C32
workload. The distinct [first E2E attempt](tirex-e2e-r1-failure.md) stopped
before any timing block. This r2 diagnostic contains **no timing**, so
forecast/s, speedup, and 95% confidence intervals remain **unmeasured**.
The other B2/C64, B2/C128, B1/C32, and B4/C32 cases are still unrun. Keep
the original tolerance and preserve these failed results while isolating the
first arithmetic divergence before another qualification attempt.
