# Prepared single-GPU inference and controlled H100 results

FlashRNN2 now has an opt-in fixed-buffer execution path for generic LSTM/sLSTM
FP16/BF16 inference. It separates FP32 computation carry from output snapshots,
keeps chunk addressing inside the original buffers, and lets a caller request
full-state history, hidden history or the final hidden state. PreparedSequence
can capture CUDA Graphs. ExecutionPolicy is an explicit offline exact-shape
lookup whose table must match its recorded identity; an unknown key retains
MMA16 / four warps / weight operand B. It is not an online autotuner.

The original Triton-fused FlashRNN already uses a fused time loop, weight reuse,
FP32 internal state and local feedback. Those mechanisms are inherited. The new
implementation concerns continuation/output contracts, reusable buffers and
explicit execution choices. The [architecture figures](../architecture/flashrnn2/README.md)
make this distinction with source hashes and visible Implemented / Documented /
Proposed / Unknown labels.

## Frozen optimization cohort

The five core implementation files are byte-identical to the measured source
manifest `dc3b85003a41743daf99d2cdf362e8fb35a824660d60db9678fbf7a1e3f4ad69`.
[Per-file hashes](../../evidence/flashrnn2-prepared-h100-20261013/qualified-source.json)
separate that qualification from publication edits to tests, documentation and CI.

The cohort contains 40 generic cell configurations, two fixture splits, 40
selected holdout configurations, and eight fixed-weight four-layer classifiers.
Cell output checks use FP32 snapshots/carry; complete classifier checks include
layer observables, all carries and logits for three independently changing CPU
request payloads. All 888 saved GPU outputs pass the independent reader under
the predeclared cell/model budgets. SIMT has a different reduction order and is
numerically qualified within that budget; it is not assumed bitwise equivalent.

Selection uses split 0 only; the table is frozen before split 1 qualification and
timing. It chooses 16 SIMT configurations (eight each at four/eight warps), 21
MMA configurations at eight warps, and three unchanged MMA four-warp controls.
A representative PTX sample contains warp `mma.sync`, `ldmatrix` and `bar.sync`;
it does not establish WGMMA, TMA, asynchronous prefetch or effective overlap.

| Controlled comparison | Independent cases | Paired speedup range |
| --- | ---: | ---: |
| Selected configuration vs MMA four warps, changed choices only | 37 | 1.0503–2.0276× |
| Prepared chunks vs previous chunk slicing/packing/concatenation | 12 | 1.1291–2.4444× |
| Complete optimized classifier vs previous FlashRNN2 execution | 8 | 1.2133–2.8846× |
| Optimized vs previous execution, CUDA Graphs in both arms | 8 | 1.1395–2.0508× |
| Hidden-output reduction alone, after matched dispatch | 8 | 1.0131–1.0938× |

These comparisons are within-FlashRNN2 ablations. Complete calls include CPU ID
transfer, embedding, all four projections/recurrences, readout and CPU output
delivery. They are fixed-weight classifiers, not pretrained-model accuracy or
GPU backward results. The public default and the historical TiRex arithmetic
are not changed by an opt-in selected plan.

| Classifier | Previous ms | Optimized graph ms | Paired ratio [95% CI] | Matched graph ratio |
| --- | ---: | ---: | --- | ---: |
| model-lstm-fp16-b2-t32-d32-h4 | 4.8844 | 1.8088 | 2.6957 [2.6880, 2.7027] | 1.8739 |
| model-lstm-fp16-b16-t32-d64-h2 | 0.7974 | 0.5256 | 1.5194 [1.5159, 1.5232] | 1.2817 |
| model-lstm-fp16-b16-t128-d64-h2 | 2.2930 | 1.8004 | 1.2755 [1.2722, 1.2790] | 1.1947 |
| model-lstm-fp16-b32-t128-d64-h2 | 1.1314 | 0.9070 | 1.2477 [1.2455, 1.2501] | 1.1769 |
| model-slstm-fp16-b2-t32-d32-h4 | 5.5911 | 1.9427 | 2.8846 [2.8751, 2.8952] | 2.0508 |
| model-slstm-fp16-b16-t32-d64-h2 | 0.8175 | 0.5817 | 1.4071 [1.4020, 1.4139] | 1.1581 |
| model-slstm-fp16-b16-t128-d64-h2 | 2.3295 | 1.8954 | 1.2352 [1.2293, 1.2410] | 1.1443 |
| model-slstm-fp16-b32-t128-d64-h2 | 1.1474 | 0.9453 | 1.2133 [1.2119, 1.2149] | 1.1395 |

Each timing row has five process starts and 20 ABBA/BAAB blocks per start. The
10,000-sample hierarchical bootstrap resamples starts and blocks; starts share
one physical H100 and allocation, so the interval is not device/job variance.
The 392 rows contain 39,200 paired blocks. The frozen analyzer originally labels
38,900 genuine blocks and 300 replicas. Independent mechanism inspection finds
six more identical hidden/small model paths, giving **38,300 genuine comparison
blocks and 900 same-configuration replicas**. Both classifications are preserved
in [summary.json](../../evidence/flashrnn2-prepared-h100-20261013/summary.json).

Negative results are retained: SIMT loses eight of 24 direct comparisons for
each warp choice; MMA32 loses all four tests; two dispatch-only classifier cases
regress; SIMT in the two eligible complete classifiers is slower; and one graph
chunk case is slower. Full group counts and intervals are in the
[result table](../../evidence/flashrnn2-prepared-h100-20261013/RESULTS.md).
The [statistics audit](../../evidence/flashrnn2-prepared-h100-20261013/statistics-audit.json)
recomputes all intervals from 240 raw timing files. Complete arrays and raw
archives are retained separately by their original hashes and are not embedded
in this source PR.

## State precision, D/heads and chunk boundaries

An earlier generic domain cohort spans D=16,17,32,48,64,65,96,128, varied heads,
ragged B/T, FP16/BF16 and full state outputs. Its 56 FP32-history and carry checks
pass. The original fixed snapshot budget admits 42/56 low-precision full-state
histories; the other 14 BF16 sLSTM cases fail and remain failures. Widening an
already rounded snapshot does not recover computation carry. Request FP32
snapshots when that qualified observable is required; do not relax the budget.

All 72 whole/chunk comparisons are bitwise equal with full computation carry.
All 24 shard controls pass. Twenty-four rounded-carry and per-step-rounding
controls detect changed results; six mixed-initialization controls detect an
incorrect local all-zero reduction; 24 low-precision continuation inputs are
rejected. These are semantic controls, not a distributed implementation. The
[qualification summary](../../evidence/flashrnn2-prepared-h100-20261013/state-qualification-summary.json)
retains the source report hash. Per-configuration timings and compiler resources
are provided separately; full physical placement and spill attribution for every
state/weight/input object remain unknown.

## Separate external baseline supplement

[external-baselines.csv](../../evidence/flashrnn2-prepared-h100-20261013/external-baselines.csv)
retains a different source-bound cohort. Original public FlashRNN is pinned at
`0b84fc30120ecccbde97976bc5b086f3215572d2`. Eighteen FP16 classifier controls cover
LSTM/sLSTM and three B/concurrency/T configurations, with five starts and 100
paired blocks each (90 starts, 1,800 blocks):

| Original FlashRNN backend | Cases | Ratio range |
| --- | ---: | ---: |
| CUDA | 6 | 3.5758–4.4143× |
| CUDA fused | 6 | 1.8445–2.6751× |
| Triton fused | 6 | 1.1649–2.2708× |

The supplement also retains eager/compiled FP32-carry controls and a TiRex
repeated-start result: 20.1381× versus its official Torch path, interval
[20.0368,20.2379]. This TiRex row uses B2, concurrency 32 and context 64 with its
recorded forecast protocol. It is distinct from the historical 20.0234×
within-process cohort already reported in the full-model table. Neither row
qualifies changed kernel sources or fills unmeasured author CUDA/ONNX, C128,
other-batch, distributed or language-model comparisons.

A separate [negative original pinned TiRex CUDA-cell fixture qualification](../../evidence/flashrnn2-prepared-h100-20261013/external-qualification-negatives.json)
records 364 failed pairs out of 381 under the unchanged `1e-4` budget across
B1/B2/B4 model fixtures. It admits no B2/concurrency-32 timing and establishes no
task-quality result. It is distinct from both the original FlashRNN CUDA
classifier baseline and the qualified FlashRNN2-versus-Torch TiRex row above.

## Reproducibility and publication checks

The [harness snapshot](../../evidence/flashrnn2-prepared-h100-20261013/harness-snapshot/)
retains the exact finite shape plan, NumPy oracle, baseline implementation,
qualification/selection/timing worker and analyzer. These files are documentary
snapshots of the frozen run; they expect their original manifest/fixture/ready
receipts and are not an automatic resource-allocation service. The independent
replica correction above is deliberately not rewritten into the frozen analyzer.

Publication checks run 32 CPU regression tests and validate the source mapping,
report hashes, editable architecture graphics and whitespace. No GPU run was
performed for publication. Gluon, per-model changed arithmetic, GPU backward,
resource-aware fallback, async prefetch, cluster execution and FP8 require their
own implementation or qualification.
