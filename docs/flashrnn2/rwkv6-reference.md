# RWKV6 full-checkpoint continuation reference

The complete RWKV6 1.6B checkpoint finishes all 36 cached-versus-full-prefix
CPU comparisons. Every logit comparison and all 84 greedy token choices pass.
B2/B4 logits and all three cache categories are bitwise identical. B1 retains
nine recurrent-state failures under the original `1e-5 + 1e-5 * abs(reference)`
budget; neither the threshold nor those failed results has been replaced.

| Batch | Logits pass | Recurrent states pass | Attention/FFN shift states pass | Matching tokens | Maximum logit / recurrent error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12/12 | 3/12 | 12/12, 12/12 | 12/12 | 0.000167847 / 0.000457764 |
| 2 | 12/12 | 12/12 | 12/12, 12/12 | 24/24 | 0 / 0 |
| 4 | 12/12 | 12/12 | 12/12, 12/12 | 48/48 | 0 / 0 |

Each batch uses three prompt groups with five input tokens and four generated
tokens. In B1, steps 1–3 of every group fail recurrent-state comparison; the
worst element is 13.661272 times its allowed error. Attention and FFN shift
states remain within budget. The complete gate naturally exits 1.

The hash-verified `fla-hub/rwkv6-1.6B-finch` checkpoint is pinned to
`94302fd437462b5110b06a9e83b32b8a1684e8d4`. All 582 BF16 tensors are promoted
to FP32 and strictly assigned: 1,599,873,024 parameters, 24 layers, hidden size
2048 and 32 heads of size 64. The embedding and vocabulary head remain untied.
No model layers or checkpoint tensors are omitted.

The adapter executes unchanged class/function bodies extracted from FLA
`17dd5662554d46b6bcb1d1ff728cebb461c9aef9`, using its own naive recurrent,
LayerNorm, GroupNorm and token-shift references. CUDA/Triton operators are not
executed. Its two per-layer cache updates retain recurrent, attention-shift and
FFN-shift state; FLA's offset argument is accepted but not accumulated, so
position counters and the full native Cache API remain unqualified. Upstream
warns that this RWKV implementation needs comparison with BlinkDL's original;
that independent full-model crosscheck is still outstanding.

The [independent audit](evidence/rwkv6-reference-r1-audit.json) recomputes all
36 saved cached/full-prefix logit pairs, all 84 token choices and 216 final
cache tensor pairs (all 24 layers and three categories, group 0 at each batch).
Other intermediate cache comparisons are runner evidence. The auditor also
checks source hashes, exact row coverage, the binary snapshot hash and an
independent longest-byte-match reconstruction of all 12 fixed ASCII inputs.
The tokenizer has 65,530 entries; the model vocabulary is padded to 65,536.
An earlier precheck's incorrect equal-size assumption is explicitly retained.
The auditor naturally exits 0 while preserving the model's failed status.

Before loading the checkpoint, two random blocks passed 27 continuation cases
and 189 tensor comparisons, with maximum error 4.76837e-7. These primitive
controls do not establish full-checkpoint accuracy. The original source-fetch
attempt's incorrect `naive.py` path and its corrected pinned
`recurrent_naive.py` fetch are retained in the source manifest.

| Required comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Native FLA/BlinkDL GPU generation | — | — | Not measured |
| Multiple batches and high concurrency E2E | — | — | Not measured |

Reproduce with the pinned checkpoint and FLA source assets on an existing runtime:

```bash
python tools/flashrnn2/rwkv6_reference_gate.py \
  --model "$RWKV6" --source "$PINNED_FLA" \
  --output "$RESULTS/rwkv6-reference.jsonl"
```

[Evidence inventory](evidence/rwkv6-reference-evidence-manifest.json). Long
sequences, native accelerated kernels, training and high-concurrency GPU E2E
remain unfinished. Checkpoint files are retained for these consumers.
