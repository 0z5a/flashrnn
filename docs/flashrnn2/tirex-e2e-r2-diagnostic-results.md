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
all 185 payload hashes were checked. The producer report
and independent NumPy audit
cover all 2,048 expected tensor pairs and 26,820,608 elements. The audit
found 215 failing pairs, no missing or unexpected pairs, and no disagreement
with the producer's pass/fail flags. The scope summary
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

A separate read of the saved raw output tensors locates the first **bitwise**
difference in each affected group. Every earlier captured cell output in that
group is bitwise identical. These are sequence steps within the indicated
64-step patch; some first differences are still within the numerical budget.
The bitwise summary
binds this read to the raw PT SHA256 above.

| Group | First different layer/patch | First different step | Different output elements there |
|---|---|---:|---:|
| G1 | L3/P1 | 61 | 4 |
| G4 | L3/P1 | 59 | 1 |
| G6 | L10/P1 | 61 | 21 |
| G7 | L9/P1 | 61 | 21 |
| G8 | L9/P0 | 61 | 10 |
| G11 | L3/P0 | 61 | 3 |

Reading the same archived raw tensor at each first differing step gives one
changed output element per target. Each difference is one BF16 representable
increment; the table uses the original `1e-4 × (1 + |reference|)` budget.

| Group | First differing output coordinate (batch, head, width) | Reference | Candidate | Difference / budget |
|---|---|---:|---:|---:|
| G1 | (1, 0, 32) | -0.0049133301 | -0.0049438477 | 0.304× |
| G4 | (1, 1, 50) | 0.0128173828 | 0.0127563477 | 0.603× |
| G6 | (1, 1, 26) | -0.1738281250 | -0.1748046875 | 8.319× |
| G7 | (1, 2, 32) | 0.0512695313 | 0.0515136719 | 2.322× |
| G8 | (0, 1, 89) | -0.0172119141 | -0.0173339844 | 1.200× |
| G11 | (1, 0, 86) | -0.0397949219 | -0.0395507813 | 2.348× |

The single-element first difference narrows the next gate/state calculation to
inspect. It still does not identify which operation crossed the BF16 rounding
boundary, and later failures cannot be attributed to one operation from this
output snapshot alone.

The late, input-dependent first differences make the corresponding recurrent
matmul and pointwise operations the next diagnostic target. They do not yet
identify which operation introduces the first difference.

The prior [r5 gate](tirex-gpu-r5-results.md) remains 381/381 passing on its
fixed first-two-series input. It does not qualify the new groups in this C32
workload. The distinct first E2E attempt stopped
before any timing block. This r2 diagnostic contains **no timing**, so
forecast/s, speedup, and 95% confidence intervals remain **unmeasured**.
The other B2/C64, B2/C128, B1/C32, and B4/C32 cases are still unrun. Keep
the original tolerance and preserve these failed results while isolating the
first arithmetic divergence before another qualification attempt.
