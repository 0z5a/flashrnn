# Exact-shape table fill: FP16 and BF16 grid (D 192-768 x B 1/8/32/128 x T 64/512)

The author's grid: D in {192, 256, 384, 512, 768} x B in {1, 8, 32, 128} x T in {64, 512} x {LSTM, sLSTM}, H = 1,
FP16 and BF16, one batch per dtype on one node (run table fill FP16 FP16 and table fill BF16 BF16, 30.1 / 31.0 min,
`ALL_STARTS_COMPLETED`, archives verified, 2284 files each). Evidence: `evidence/flashrnn2-rtiled/` and
`evidence/flashrnn2-rtiled/`.

## Protocol

* Candidates: the M0-qualified Triton R-tiled configurations of each (cell, D) and every TileLang configuration
  qualified under the full contract in both dtypes (M6, M7, HC1 v2, HC3 v2: 76 TileLang candidates, all with a
  full-contract PASS in FP16 and BF16), 944 candidates per dtype.
* Split 0: every candidate re-checked against the FP64 oracle under fixture_math's CELL_BUDGET at the exact shape,
  the fastest qualified one selected; comparison and policy tables frozen (hashed) before any split-1 run.
* Split 1: selected configuration vs `default_config(D)` served through `ExecutionPolicy.select`, 5 process starts x
  20 ABBA/BAAB blocks, 10000-draw hierarchical bootstrap.
* Phase 3: every frozen entry through the public one-call `dispatch` with the frozen table as an explicit policy
  (served configuration; bitwise equal to the backend called directly, to `PreparedSequence` and to chunked
  continuation; fixture_math E1): 80/80 entries PASS in each dtype.
* Inclusion rule, fixed before the batches ran (`make_table`): 5/5 starts PASS, 95% interval above 1, selected !=
  default, dispatch contract PASS.

## Result

`flashrnn/flashrnn2/tables/h100-rtiled-grid-20261015.json`: 150 of 160 shapes enter the table (same identity as M6
and M7, no overlapping keys; `dispatch` now serves M6 + M7 + this file = 200 shapes). The 10 excluded shapes are
sLSTM at D = 384/512 where the selection was Triton R-tiled and no faster than the default (interval covering 1 or
below 1); they keep `default_config`.

* Selected families: 57 TileLang cluster (BM 16), 24 TileLang grid (BM 16), 8 TileLang grid two warpgroups (BM 64,
  D = 768, B = 128), 61 Triton R-tiled.
* Speedup of the selected configuration over the default, all 160 shapes: geometric mean 1.34x, range
  0.997x-2.28x; per D (FP16 / BF16 geometric means): 192 1.50 / 1.49, 256 1.32 / 1.32, 384 1.19 / 1.20,
  512 1.15 / 1.18, 768 1.59 / 1.60.

Speedup over `default_config` per shape (split 1; * = excluded by the rule):

| dtype | cell | D | B1 T64 | B1 T512 | B8 T64 | B8 T512 | B32 T64 | B32 T512 | B128 T64 | B128 T512 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bfloat16 | lstm | 192 | 1.07 | 1.10 | 1.09 | 1.06 | 1.41 | 1.73 | 1.38 | 1.76 |
| bfloat16 | lstm | 256 | 1.09 | 1.11 | 1.09 | 1.10 | 1.39 | 1.66 | 1.32 | 1.62 |
| bfloat16 | lstm | 384 | 1.28 | 1.30 | 1.28 | 1.30 | 1.28 | 1.51 | 1.19 | 1.20 |
| bfloat16 | lstm | 512 | 1.21 | 1.22 | 1.21 | 1.20 | 1.21 | 1.36 | 1.14 | 1.14 |
| bfloat16 | lstm | 768 | 1.37 | 1.40 | 1.37 | 1.27 | 1.73 | 1.91 | 1.78 | 1.94 |
| bfloat16 | slstm | 192 | 1.44 | 1.46 | 1.44 | 1.45 | 1.88 | 2.28 | 1.77 | 2.18 |
| bfloat16 | slstm | 256 | 1.18 | 1.20 | 1.10 | 1.12 | 1.55 | 1.82 | 1.45 | 1.72 |
| bfloat16 | slstm | 384 | 1.00* | 1.01 | 1.01* | 1.01 | 1.30 | 1.62 | 1.05 | 1.07 |
| bfloat16 | slstm | 512 | 1.14 | 1.15 | 1.12 | 1.08 | 1.28 | 1.41 | 1.02 | 1.02 |
| bfloat16 | slstm | 768 | 1.39 | 1.42 | 1.39 | 1.29 | 1.78 | 2.04 | 1.82 | 2.00 |
| float16 | lstm | 192 | 1.10 | 1.12 | 1.08 | 1.11 | 1.44 | 1.71 | 1.39 | 1.75 |
| float16 | lstm | 256 | 1.09 | 1.12 | 1.09 | 1.10 | 1.40 | 1.65 | 1.33 | 1.60 |
| float16 | lstm | 384 | 1.28 | 1.29 | 1.27 | 1.29 | 1.28 | 1.50 | 1.18 | 1.19 |
| float16 | lstm | 512 | 1.21 | 1.22 | 1.19 | 1.12 | 1.19 | 1.37 | 1.04 | 1.05 |
| float16 | lstm | 768 | 1.36 | 1.39 | 1.37 | 1.26 | 1.73 | 1.90 | 1.79 | 1.92 |
| float16 | slstm | 192 | 1.44 | 1.46 | 1.43 | 1.45 | 1.90 | 2.28 | 1.80 | 2.21 |
| float16 | slstm | 256 | 1.17 | 1.19 | 1.10 | 1.12 | 1.55 | 1.80 | 1.45 | 1.73 |
| float16 | slstm | 384 | 1.00* | 1.00* | 1.00* | 1.00* | 1.31 | 1.62 | 1.00* | 1.00* |
| float16 | slstm | 512 | 1.13 | 1.13 | 1.10 | 1.06 | 1.27 | 1.45 | 1.00* | 1.00* |
| float16 | slstm | 768 | 1.38 | 1.41 | 1.39 | 1.29 | 1.78 | 2.02 | 1.81 | 1.98 |
