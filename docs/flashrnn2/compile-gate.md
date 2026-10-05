# CPU Inductor state and gradient qualification

Default CPU Inductor passes the mathematical policy for all four cells but
fails every tested BF16-operand case. Enabling
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` restores all BF16 full-state,
all-input-gradient and chunk checks at the unchanged error budget. The public
recurrence implementation is unchanged. Original default failures remain in
the evidence; the option is verified inside each controlled child process.

The installed Torch configuration documents that default fusion may omit
intermediate low-precision rounding. Its precision-preservation option retains
those casts and also affects other numerical optimizations. These controls
qualify that option on the tested build; they do not attribute every difference
to one individual compiler pass.

| Policy and compiler setting | Cases passing | Independent saved-tensor comparisons |
|---|---:|---:|
| Mathematical FP32, default Inductor | 4/4 | Part of 96-pair default audit |
| BF16 operands, default Inductor | 0/4 | Original failures retained in same audit |
| BF16 operands, preserve precision casts | 4/4 | 48/48 comparisons recomputed |

Every case uses B=2, T=4, H=1, D=8, FP32 public tensors, full history plus final
carry, all four input gradients, and two chunks of T=2. The same eager public
API is the oracle, with `atol=rtol=1e-5`. All passing cases capture two shapes
with `fullgraph=True`, `dynamic=False`, `mode="default"`; none recompiles during
the steady probe. This does not qualify unseen shapes, long Python loops,
scan, masks, compiled readonly/alias behavior or additional devices.

## Local latency probe

Apple M5 CPU, Torch 2.13.0, Python 3.12.14, Apple clang 21.0.0. Torch/OMP/MKL
and Inductor compile threads are one. Every cell/policy runs in its own fresh
process with a separate cache. Ten alternating ABBA/BAAB blocks provide 20
samples per method. Times below are medians of **forward + loss + backward**;
they exclude input projection, optimizer steps, compilation and model serving.
Only correctness-passing cases are timed.

| Cell | Policy / Inductor setting | Eager ms | Compiled ms | Observed speedup |
|---|---|---:|---:|---:|
| LSTM | Mathematical / default | 1.302583 | 0.807125 | 1.614× |
| sLSTM | Mathematical / default | 2.531396 | 1.461021 | 1.733× |
| GRU | Mathematical / default | 0.560667 | 0.190354 | 2.945× |
| Elman | Mathematical / default | 0.639938 | 0.319792 | 2.001× |
| LSTM | BF16 operands / preserve casts | 0.844500 | 0.427146 | 1.977× |
| sLSTM | BF16 operands / preserve casts | 1.792146 | 0.521251 | 3.438× |
| GRU | BF16 operands / preserve casts | 1.057145 | 0.323771 | 3.265× |
| Elman | BF16 operands / preserve casts | 0.554125 | 0.229250 | 2.417× |

These are small-tensor local probes from one process per case, without repeated
fresh-process confidence intervals or a controlled quiet machine. Compare the
two methods within each row; cross-policy timing differences are not a matched
experiment. They are not promoted as full-model or GPU speedups.

First forward, including compilation, costs 14.15–27.18 seconds for passing
cases; first backward costs 6.27–24.13 seconds and first chunk execution with
compilation costs 5.48–15.23 seconds. Wrapper creation, raw phase times,
process/finished-child peak RSS, graph counters and all timing samples are
recorded separately. CPU compilation uses only the existing runtime/toolchain
and task-owned caches; no environment installation or update is performed.

## Reproduction and evidence

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
TORCHINDUCTOR_CACHE_DIR="$RESULTS/inductor-cache" \
python tools/flashrnn2/compile_precision_gate.py \
  --cell slstm --numerics fp32_state_bf16_mma --output "$RESULTS/slstm"
```

Use `compile_gate.py` without the precision-cast environment flag to reproduce
the default experiment. Output and cache directories must be fresh. The
precision option is version-specific; other Torch builds need their own gate.

[Evidence](../../evidence/flashrnn2/compile-gate/) contains process receipts,
original default failures, the controlled rerun, independent audits, generated
code hashes and installed compiler-source hashes. `*-results.jsonl` stores the
complete per-case records; writing each `record` with `json.dumps(indent=2)`
plus a newline exactly recovers its `original_json_sha256`. Raw `.pt` files and
generated C++/Python/shared-library files remain in the execution workspace.
The auditors recompute all 144 saved comparison pairs across both runs and
verify the timing sample counts, order and medians. Their successful exit does
not turn the four original numerical failures into passes.

Full-model/high-concurrency E2E, GPU compilation and dedicated recurrent kernels
remain separate, unfinished qualifications.
