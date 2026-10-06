# Portable Torch state and gradient qualification

Splitting a BF16-operand recurrence at a chunk boundary changed its recurrent
weight gradient. In contract v1, the full call returned `BF16(sum(dR_t))`, while
separate calls returned the FP32 sum of separately rounded chunk cotangents.
All 14 failing cases are reproduced exactly by this cast-placement difference.
Sharing a single cast node across the chunks also removes all 14 failures.

Contract v2 instead casts R inside each time step. Each step's cotangent passes
through the ordinary Torch BF16 cast before FP32 accumulation. The same cast
nodes now exist for every partition of the sequence. This intentionally changes
R gradients; it does not introduce a straight-through custom backward or relax
the error budget. All 152 saved whole/chunk forward tensor pairs and all 114
other full-call input-gradient pairs remain bitwise equal to v1.

The default mathematical policy and existing direct reference calls keep their
behavior. The optional reference argument `recurrent_mma_dtype` controls this
new weight cast; its default is `None`. This is an eager Torch arithmetic
contract, not a Tensor Core or native CUDA backward emulation.

## Results

Local runtime: Python 3.12.14, Torch 2.13.0 CPU, arm64, one Torch/OMP/MKL thread.
No runtime installation or environment update was performed. Base is Draft #30,
commit `4d5b12b18397bd0a718b509eef6d8f0eae472afa`.

| Numerical policy | Cases | v1 passes | v2 passes | Forward budget | Gradient budget |
|---|---:|---:|---:|---|---|
| Mathematical FP64 | 19 | 19 | 19 | atol=rtol=1e-12 | atol=rtol=1e-11 |
| FP32 state / BF16 operands | 19 | 5 | 19 | atol=rtol=1e-5 | atol=rtol=1e-5 |
| Total | 38 | 24 | 38 | Unchanged | Unchanged |

| Evidence | Result |
|---|---|
| Original r1 natural process exit | 1; all 38 cases recorded, 14 R-gradient failures |
| Corrected r2 natural process exit | 0; all 38 cases pass |
| Independent audit, each revision | 456 tensor pairs and 176 per-state comparisons recomputed |
| Cast-placement diagnosis | All 14 failed R gradients exactly predicted |
| Original worst chunk R-gradient difference | 0.05078125, saturated sLSTM |
| Local unit regression | 15 methods pass; includes three different chunk sizes for all four cells |
| Input mutation / repeat | Passed runner checks; r2 also saves inputs after calls and repeated outputs for independent audit |
| Device / dtype | Passed runner checks; snapshots are moved to CPU for storage |

The matrix covers LSTM/sLSTM/GRU/Elman, B/T/H/D of 1/1/1/3, 3/3/2/7,
2/17/1/16 and 1/33/2/5, both policies, plus sLSTM zero, mixed normalizer and
saturated-gate cases. Every case compares full history, final carry, all four
input gradients, a split continuation, readonly inputs and repeatability.
Mathematical outputs use the upstream vanilla reference. The BF16 oracle uses
separate single-step calls with explicit casts, independently of the new
weight-cast argument. The loss uses every history element and the final carry.

## Reproduction and evidence

From an environment with this package installed:

```sh
python tools/flashrnn2/portable_gate.py --device cpu \
  --config tools/flashrnn2/portable-small.json --output results/portable.jsonl
```

The output path must be fresh. JSONL rows point to all-input, full-state and
full-gradient tensor snapshots with SHA256 hashes. Metadata records source and
configuration hashes. The CPU-wheel workflow runs this matrix against the
installed package outside the checkout and uploads both rows and tensor files.
The independent Linux CPU-wheel run passed on implementation commit
`bd72edbeb8b3ccd07b8658d04190d74e6e958b5f`: [run 37348607245](https://github.com/0z5a/flashrnn/actions/runs/37348607245).
It used Python 3.12.14, Torch 2.14.1+cpu and einops 0.8.2 on x86_64.
All 38 cases and the 15-method test step passed. Downloaded tensors were
independently recomputed: 456 pairs and 176 per-state checks, plus readonly
and repeat comparisons. Installed source hashes match that commit; package
origins and wheel hashes are retained. The full log fetch timed out; the
GitHub job/step receipts and all tensor artifacts were retrieved successfully.

[Raw evidence](../../evidence/flashrnn2/portable-gate/) includes original and
corrected JSONL, process receipts, independent audits, the original source
gate snapshot and exact parent-commit source references, executed audit/diagnosis scripts and a file-hash manifest. Local raw
`.pt` files are retained under `evidence/portable-cpu-r{1,2}.tensors/` in the
execution workspace; their hashes are in the published rows. They are not
included in the Git diff. Reproduction regenerates the small fixtures.

| Measurement level | Baseline | Candidate | Speedup |
|---|---|---|---|
| CPU correctness matrix | Contract v1 | Contract v2 | Not measured |
| [H20 CUDA correctness matrix](h20-portable-cuda.md) | Public Torch oracle | Portable Torch CUDA, 38/38 pass | Not measured |
| Full-model high-concurrency GPU E2E | Unmeasured | Unmeasured | Unmeasured |

The H20 run exercised CUDA and the two-stream output checks with independent
tensor-level audits. ROCm/XPU, compile compatibility, sanitizer checks, GPU
training and the complete model/high-concurrency campaign remain unqualified.
Repeated R casts can affect eager execution cost; no throughput improvement is
claimed.
