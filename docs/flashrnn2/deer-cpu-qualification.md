# DEER GRU CPU numerical qualification

The pinned [DEER implementation](https://github.com/machine-discovery/deer/tree/409140ff98d2426e49090e3552a63da10d9cd6dc) ran its original sequential and DEER solvers on identical FP32 weights and inputs. The two-layer B2/T16/D8 gate passed 21 output/input/weight-gradient pairs and four independent NumPy forward replays.

A separate four-layer T128/D32 recurrence screen used a squared final-layer-history loss and compared every layer history, all 36 weight gradients and the input gradient. NumPy independently replayed all four layers for both solvers. The fixed budget was `1e-4 × (1 + abs(reference))`.

| Batch | Paired tensors | Independent forward replays | Converged layers | Failed elements |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 41/41 | 8/8 | 4/4 in both solvers | 0 |
| 4 | 41/41 | 8/8 | 4/4 in both solvers | 0 |
| 16 | 41/41 | 8/8 | 4/4 in both solvers | 0 |
| 32 | 41/41 | 8/8 | 4/4 in both solvers | 0 |
| 64 | 41/41 | 8/8 | 4/4 in both solvers | 0 |

The isolated CPU runtime used JAX/JAXLIB 0.4.34, NumPy 1.26.4, Optimistix 0.0.7 and Equinox 0.11.10. These are numerical preflights, not H100 GPU qualification, a complete task-head/optimizer training step, high-concurrency E2E or a measured speedup. The [screen source](../../tools/flashrnn2/deer_four_layer_cpu_screen.py) and the independent small-gate auditor remain available for the separate GPU admission.
