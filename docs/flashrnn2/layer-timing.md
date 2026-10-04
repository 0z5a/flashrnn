# LSTM layer pilot

One RTX 5090 / SM120 session, PyTorch 2.12.1+cu130, CUDA 13.0,
cuDNN 9.20. The workload includes the input projection, recurrence, hidden
history and final states. B=16, T=128, heads=1, D=64; inputs/weights and public
outputs are BF16. Candidate local states are FP32. cuDNN's internal arithmetic
is opaque. Both paths pass the same independently computed reference gate.

| Workload | Baseline | Candidate | Baseline median | Candidate median | Paired speed ratio | Within-process 95% bootstrap CI |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| L1 LSTM B16/T128/H1/D64 | torch.nn LSTM / cuDNN | Inferred-layout single-CTA Triton | 1.61775 ms | 0.26742 ms | **6.367×** | 6.088–6.741× |

The ratio is the geometric mean of the 20 paired ratios, not the ratio of
the two medians. Ten blocks run baseline first and ten run candidate first.
Baseline-first medians are 1.61615/0.27886 ms; candidate-first medians are
1.61775/0.24757 ms. The shared node held both GPU locks and the heavy-I/O lock
through the finite run; its controller exited 0 naturally after 3.79 s.

Each measurement uses host wall time and a CUDA synchronization on each side.
Dispatch, allocation and per-call layout conversion are included. The
one-time torch.nn parameter mapping takes 14.20 ms and is excluded from steady
state. First calls take 177.57 ms for the baseline and 769.30 ms for the
candidate; these use the existing task compiler cache and are not clean-build
measurements. Five warmups precede the paired blocks.

The profiler observes `aten::_cudnn_rnn`, establishing actual cuDNN dispatch.
Maximum hidden-history/final-state errors are 0.0004883/0.0009766 for cuDNN
and 0.0002441/0.0000305 for the candidate, within the frozen gate.

This is one layer shape and one process. Independent startup validation,
the original FlashRNN CUDA/Triton baselines, other shapes, concurrency,
backward and complete model timings remain required. No full-model speedup
is inferred from this table.

Evidence: [20 pairs](evidence/l1-r1.jsonl),
[metadata and source hashes](evidence/l1-r1.meta.json),
[controller](evidence/l1-r1-controller.json),
[analysis](evidence/l1-r1-analysis.json).

```sh
PYTHONPATH=. python tools/flashrnn2/bench.py --output l1.jsonl --candidate triton_persistent
python tools/flashrnn2/analyze.py l1.jsonl --output l1-analysis.json
```
