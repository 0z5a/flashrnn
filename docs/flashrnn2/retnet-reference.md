# RetNet full-checkpoint continuation reference

The complete RetNet 1.3B CPU run finishes all 36 cached-versus-full-prefix
steps and 84 greedy token comparisons. Logits and tokens pass throughout.
BS=1 retains nine recurrent-state failures; BS=2/4 logits and all 24 layer
states are bitwise identical. The original process naturally exits 1.

| Batch | Logits pass | All-layer states pass | Matching tokens | Maximum logit error | Maximum state error |
|---:|---:|---:|---:|---:|---:|
| 1 | 12/12 | 3/12 | 12/12 | 0.0000314713 | 0.203125 |
| 2 | 12/12 | 12/12 | 24/24 | 0 | 0 |
| 4 | 12/12 | 12/12 | 48/48 | 0 | 0 |

Each batch uses three fixed prompt groups, five input tokens and four generated
tokens. BS=1 fails state comparisons at decode steps 1–3 of every group; its
worst element is 188.8065 times the allowed error. Frozen budgets are
`atol=rtol=1e-3` for logits and `atol=rtol=1e-5` for states. Thresholds were not
relaxed. The batch-dependent pattern alone does not establish its cause.

## Checkpoint and implementation scope

`fla-hub/retnet-1.3B-100B` is pinned to
`7fddefc4d5e196a8d1f076bb7612d54321b3effe`. All eight files were hash-verified;
the 2,703,483,944-byte weight file has SHA256
`de46d5e8ce1d524ac790776abfc8c3293e1c42df86a461a719a101197e8af4bd`.
All 267 BF16 tensors are promoted to FP32 and strictly loaded into all
1,351,727,104 parameters: 24 layers, hidden width 2048, eight heads, K=256,
V=512 and an untied 32,000-token vocabulary head. No layers are omitted.

The adapter executes unchanged pinned FLA attention/block/rotary class and
function bodies at `17dd5662554d46b6bcb1d1ff728cebb461c9aef9`. Existing pinned
Torch RMS/MLP helpers and an explicit FP32 retention recurrence replace
accelerated operators. Per head, `S = decay*S + outer(k,v)`, where
`decay = 1 - 2**(-5-head)`; scaled q reads the updated S. RoPE slices use the
incoming per-layer sequence length. Each layer's position counter is checked
at every model call. Variable lengths, masks and `cu_seqlens` are excluded.
CUDA/Triton/FLA fused kernels have not executed in this qualification.

| Preflight control | Result |
|---|---|
| Five retention shapes, zero/nonzero initial states | 10/10 pass; chunk outputs/states bitwise |
| Retention against pinned parallel formula / independent FP64 final state | Max errors 9.26063e-8 / 4.37047e-7 |
| RoPE width 8/256, chunk positions and wrong-offset negative control | 2/2 pass |
| Separate per-layer position lengths | Pass |
| Pinned small complete block, BS=1/2/4 | 3/3 pass; maximum output error 4.69387e-7 |

Local runtime is Python 3.12.14, Torch 2.13.0 CPU, Transformers 4.54.1,
safetensors 0.8.0, one Torch/OMP/MKL thread. Both Tokenizers and the Transformers
fast wrapper produce identical tokens for all 12 prompts. The run experienced
substantial shared paging; elapsed time is not a performance result.

## Independent evidence and reproduction

The [audit](evidence/retnet-reference-r1-audit.json) recomputes all 36 pairs of
saved full-vocabulary logits, 84 token choices and 72 complete final layer-state
pairs: all 24 layers for group 0 at each batch. All 48 BS=2/4 state pairs pass;
all 24 BS=1 final pairs fail. Other intermediate state checks are runner
evidence. Source, snapshot and tokenizer hashes, row coverage and layer-length
counters are verified. The audit naturally exits 0 while retaining the model's
failed status. Both sides of every audited comparison are saved locally.

```sh
python tools/flashrnn2/retnet_reference_gate.py \
  --model "$RETNET_CHECKPOINT" --source "$PINNED_RETNET_SOURCES" \
  --common "$PINNED_GLA_SOURCES" --output "$RESULTS/retnet-reference.jsonl"
```

[Evidence inventory](evidence/retnet-reference-evidence-manifest.json) includes
raw JSONL, the process receipt, metadata, independent audit and source/model
manifests. Binary snapshots remain in the execution workspace with their
published SHA256 hashes. The initial download timeout and successful resumed
download remain recorded. Model weights are retained for unfinished native GPU
and full-baseline consumers.

| Required comparison | Baseline tok/s | Candidate tok/s | Speedup |
|---|---:|---:|---|
| CPU cached/full-prefix correctness | — | — | Not measured |
| Native FLA GPU generation | — | — | Not measured |
| High-concurrency, multi-batch full GPU E2E | — | — | Not measured |

The next numerical diagnostic must preserve these original failures. Native
GPU, longer contexts, training and high-concurrency performance remain unfinished.
