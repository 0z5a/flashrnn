# Recurrence contracts — 2026-10-05

This draft adds experimental reference and measurement utilities. It does not change the public dispatcher or production kernels. GPU acceleration, training acceleration and model E2E have not been established.

Baseline: `NX-AI/flashrnn@0b84fc30120ecccbde97976bc5b086f3215572d2`. Python 3.12.3, PyTorch 2.12.1+cu130. Tests used CPU tensors with CUDA devices hidden and one CPU thread. The reference uses the public `[B,T,G,N,D]` input and `[S,B,T,N,D]` output layouts. GRU retains its distinct three recurrent/input gates and four bias gates. sLSTM retains the upstream vanilla global zero-normalizer initialization condition.

## Results

| Check | Cases | Actual result |
| --- | --- | --- |
| Public vanilla output and all four input gradients | LSTM, sLSTM, GRU, Elman × zero/nonzero initial state × final-only/history loss | Passed 16 combinations |
| Directional finite difference | Four cells, two timesteps, double precision | Passed |
| Chunk continuation and unchanged inputs | Four cells | Passed |
| Recurrent packing and coordinate ownership | D=3,64,192,256,384,512,768 | Passed |
| Paired statistics | A/A identity, 2× speed vs 50% latency reduction, invalid inputs | Passed |

Seven unittest methods passed. [Raw output](evidence/cpu-validation.log). These checks establish CPU reference behavior, not fused-backend numerical equivalence. The optional `mma_dtype` explicitly rounds recurrent hidden values; low precision kernel budgets still require baseline calibration.

## Speed comparison

| Workload | Strong matched baseline | Candidate | Speedup | Status |
| --- | --- | --- | --- | --- |
| Recurrence public API | Not timed | Not timed | N/A | Pending GPU baseline and numerical contracts |
| Full forward/backward training | Not timed | Not timed | N/A | Pending full GPU gradients and training measurement |
| Full model, multiple batch sizes and concurrency | Not run | Not run | N/A | Pending model/runtime qualification |

There is no performance claim. The analyzer rejects failed blocks, mixed workloads/devices/contracts/source revisions, duplicate blocks and fewer than 20 paired blocks. Its interval is conditional on one session and does not establish cross-startup significance.

## Reproduce

Run from the checkout with the existing PyTorch environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 python -m unittest discover -s tests/flashrnn2 -v
python tools/flashrnn2/probe.py --environment-only --output artifacts/flashrnn2/environment.json
```

No dependency installation is performed. `cases.yaml` uses JSON syntax, which is valid YAML, and preserves the plan's smoke, target, regression, holdout and long-sequence points. GPU properties and instruction/function-attribute probes are distinct: the current probe records instruction tests and function attributes as not run.
