# First four optimization results

Ratios are paired reference/candidate time; values below one are slower.

| Comparison | Cases | Ratio range | Faster CI | Slower CI | Includes one |
| --- | ---: | ---: | ---: | ---: | ---: |
| cell/mma4/selected | 37 | 1.0503--2.0276 | 37 | 0 | 0 |
| cell/mma4/hidden | 40 | 0.8879--1.3545 | 34 | 2 | 4 |
| cell/mma4/last | 40 | 0.9933--1.4861 | 37 | 1 | 2 |
| cell/mma4/simt4 | 24 | 0.3276--2.0022 | 16 | 8 | 0 |
| cell/mma4/simt8 | 24 | 0.0841--1.9174 | 16 | 8 | 0 |
| cell/mma4/mma32 | 4 | 0.6401--0.6902 | 0 | 4 | 0 |
| cell/legacy_chunks/prepared_chunks | 12 | 1.1291--2.4444 | 12 | 0 | 0 |
| cell/prepared_chunks/graph_chunks | 12 | 0.9960--1.4355 | 11 | 1 | 0 |
| model/legacy/dispatch | 8 | 0.9737--1.1663 | 6 | 2 | 0 |
| model/dispatch/hidden | 8 | 1.0131--1.0938 | 8 | 0 | 0 |
| model/hidden/small | 2 | 0.9020--0.9645 | 0 | 2 | 0 |
| model/small/prepared | 8 | 1.0361--2.0314 | 8 | 0 | 0 |
| model/prepared/optimized_graph | 8 | 1.0318--1.5651 | 8 | 0 | 0 |
| model/legacy/legacy_graph | 8 | 1.0604--1.4489 | 8 | 0 | 0 |
| model/legacy_graph/optimized_graph | 8 | 1.1395--2.0508 | 8 | 0 | 0 |
| model/legacy/optimized_graph | 8 | 1.2133--2.8846 | 8 | 0 | 0 |

| Model | Legacy ms | Optimized graph ms | Ratio and 95% CI | Matched graph ratio |
| --- | ---: | ---: | --- | ---: |
| model-lstm-fp16-b2-t32-d32-h4 | 4.8844 | 1.8088 | 2.6957 [2.6880, 2.7027] | 1.8739 |
| model-lstm-fp16-b16-t32-d64-h2 | 0.7974 | 0.5256 | 1.5194 [1.5159, 1.5232] | 1.2817 |
| model-lstm-fp16-b16-t128-d64-h2 | 2.2930 | 1.8004 | 1.2755 [1.2722, 1.2790] | 1.1947 |
| model-lstm-fp16-b32-t128-d64-h2 | 1.1314 | 0.9070 | 1.2477 [1.2455, 1.2501] | 1.1769 |
| model-slstm-fp16-b2-t32-d32-h4 | 5.5911 | 1.9427 | 2.8846 [2.8751, 2.8952] | 2.0508 |
| model-slstm-fp16-b16-t32-d64-h2 | 0.8175 | 0.5817 | 1.4071 [1.4020, 1.4139] | 1.1581 |
| model-slstm-fp16-b16-t128-d64-h2 | 2.3295 | 1.8954 | 1.2352 [1.2293, 1.2410] | 1.1443 |
| model-slstm-fp16-b32-t128-d64-h2 | 1.1474 | 0.9453 | 1.2133 [1.2119, 1.2149] | 1.1395 |

These are four-layer synthetic classifiers with total width 128 and three changing CPU request payloads. CPU input transfer and output delivery are timed.
Five starts share one GPU and allocation. The shape policy is trained on split 0; timing uses split 1. Default production behavior and TiRex are unchanged.
The frozen analyzer counts 38,900 genuine blocks and 300 replicas. Source inspection identifies six additional hidden/small model comparisons with the same selected MMA configuration and execution path. The independent mechanism audit labels these 600 blocks as replicas: 38,300 genuine blocks and 900 replicas, 39,200 total. Original measurements and analyzer outputs are preserved.
Result SHA256: 937e432e79b98424488ea1955c9126d03160d5f6fbf39ab4ab1f10c0c9e2e29c
