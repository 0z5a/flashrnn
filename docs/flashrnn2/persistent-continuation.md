# Persistent continuation and qualification

The retained state/qualification iteration starts from
`9e86b5d622097b18eeeddf678d12ea7ea24a28aa`. Subsequent single-GPU experiments
add PreparedSequence, explicit output policy and offline MMA/SIMT selection.
The [H100 results](prepared-h100-results.md) qualify the frozen generic source
files and their particular workloads; historical TiRex evidence retains its
separate executed identity. No distributed transport, scheduler, tiled partial
GEMM or GPU backward is implemented here.

## Separate output snapshots from compute carry

`triton_persistent.recurrence` keeps its default `(history, last_snapshot)` API.
The last snapshot has the output dtype and is **not** a lossless generic
LSTM/sLSTM continuation when the output is FP16/BF16. Use the explicit opt-in:

```python
from flashrnn.flashrnn2.state_contract import begin_sequence
from flashrnn.flashrnn2.triton_persistent import recurrence

# Compute once over the original logical call, before batch/head sharding.
context = begin_sequence(initial, cell="slstm", input_dtype=wx.dtype)
carry = initial
offset = 0
histories = []
for length in (1, 3, 13):
    history, carry = recurrence(
        wx[:, offset:offset + length], recurrent, bias, carry, cell="slstm",
        return_compute_carry=True, sequence_context=context,
    )
    histories.append(history)
    context = context.advance(length)
    offset += length
```

The kernel writes final carry directly from its computation state into a
separate buffer; it does not reconstruct it by widening the last output.
Only the opt-in path allocates/writes this extra buffer. Subsequent generic
chunks reject a low-precision initial snapshot. Output snapshots may remain
low precision or explicitly use FP32 independently of the carry buffer.

| Arithmetic | Compute/transport carry | Default output | Initialization |
| --- | --- | --- | --- |
| Generic LSTM | FP32 h/c | Input FP16/BF16 | None |
| Generic sLSTM | FP32 h/c/n/m | Input FP16/BF16 | Full-call all-zero n, first logical step only |
| TiRex sLSTM | BF16 h/c/n/m, retaining per-step rounding | BF16 | Full-call all-zero n, first logical step only |
| xLSTM sLSTM | FP32 h/c/n/m | FP32 | Elementwise n==0 on each step |

Generic hidden operands still round to the input dtype before recurrent GEMM.
The TiRex dot-result/per-step rounding and rounded division remain unchanged;
xLSTM keeps its own IEEE FP32 dot and elementwise normalizer behavior.
All accelerated paths remain forward-only with the existing requires-grad guard.

`SequenceContext` carries the immutable full-call predicate and a logical step
offset. Batch/head shards share the predicate; they must not recompute it from
their local n. A distributed caller must copy/broadcast it to each rank's device
and retain the same logical-call identity outside this local kernel API. The
caller owns synchronization, weight-version identity and context advancement;
the context is not a distributed protocol or a generation/credit mechanism.

The CPU reference accepts this context for **generic** arithmetic, with FP32
inputs/states and an explicit FP16/BF16 recurrent operand policy. It checks
continuation/reduction semantics, not CUDA math instruction equivalence.

## Gluon launcher

Both kernels share the required public launch signature, including the xLSTM
selector, compute-carry output and logical-start flag. Triton additionally
accepts optional TOTAL_T, STEP_BASE and OUTPUT_MODE for prepared execution. Gluon explicitly rejects TiRex and
xLSTM semantics; adding the parameter does not enable these modes. Its generic
BF16 LSTM/sLSTM path still needs a separate real GPU gate; the prepared H100
cohort uses Triton MMA and SIMT, and does not qualify Gluon.

## Source-bound TiRex qualification

New GPU gate reports contain an execution manifest covering the Triton and
Gluon kernels, state contract, adapter, layout, reference, gate, auditors,
diagnostic and E2E entry points. The manifest also binds reference-source pins,
runtime/device capability, precision settings and compile options. Compiler
environment values are hashed. Source identity is based on file contents rather
than commit labels. Gate raw archives now retain cell inputs/initial states and
forecast inputs alongside outputs, under the existing raw SHA binding.

Both the diagnostic and E2E runner reject absent or changed manifests. For an
accelerated cohort, E2E additionally requires `--diagnostic` and
`--diagnostic-audit`: matching B/C, fixture, gate identity, all comparisons,
independent audit and raw archive hashes are checked before model timing.
The implemented diagnostic remains B2/C32, with 2048 comparisons. A different
cohort such as C128 needs its own diagnostic implementation and qualification;
the C32 report cannot admit it. The original 381-pair model gate and `1e-4`
budget are unchanged. Fallback batches remain explicitly classified as fallback.

Old evidence can still be audited as historical evidence. It cannot authorize
timing of these changed sources. No historical reports are rewritten.

## Validation and remaining work

CPU regression cases cover B3/T17/H2/D48 and chunks `[1,3,13]`, mixed batch/head
initialization, all-zero/nonzero n, negative controls for local re-reduction and
rounded carry, distinct model contracts, kernel launch ABI, changed source and
compiler identities, diagnostic cohort mismatch and raw-file tampering.
They run in the existing Torch CPU workflow.

Publication validation on Torch 2.14.1 covers 32 CPU regression tests, including
reference, Torch backend/layer, TiRex layout/qualification, state context, kernel
launch ABI and exact-shape policy. Whole/chunk CPU history and carry are bitwise
equal. Batch/head shard controls retain the separate CPU-only absolute budget
of `5e-7`; this does not relax generic or TiRex GPU budgets. The ABI check treats
prepared Triton flags as optional and retains the common public Gluon call.

The recorded generic H100 cohorts are separate from these CPU checks: FP32
history/carry passes all 56 domain cases, while low-precision full-state
snapshots fail 14 of those cases under the original fixed budget. All 72
whole/chunk comparisons pass bitwise. The later prepared/output/selection cohort
independently audits 888 saved GPU outputs. See the result document for exact
boundaries and source hashes.

TiRex, xLSTM-specific arithmetic and Gluon must receive their own source-bound
model/backend gates before timing a changed source. Multi-node carry transport,
rank/device identity, global completion and distributed protocols remain outside
this implementation. The withdrawn distributed implementation is not restored.
