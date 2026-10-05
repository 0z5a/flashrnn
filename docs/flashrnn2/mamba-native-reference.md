# Full native Mamba Torch reference

The numerical diagnosis previously compared the same scripted model on two
platforms. This adds an independent native Torch fallback on each platform:
selected unchanged Transformers methods execute the complete Mamba-130M
network. The serialized artifact supplies only audited weights; its computation
graph is never called for native execution.

The local CPU qualification passes all short and long-generation cases exactly.
Same-platform remote CPU/CUDA comparisons are prepared but have not run. The
previous remote failures against the original native CPU oracle remain failures.

| CPU execution, complete 24-layer / 129,135,360-parameter model | Batch | Compared steps | Token choices | Largest tensor error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Native P5, prefill plus one decode; original oracle and script compared separately | 1 | 3 input cases | All match | 0 | PASS |
| Native P128/G32, serial and interleaved | 1 | 192 | 192 | 0 | PASS |
| Native P128/G32, serial and interleaved | 2 | 192 | 384 | 0 | PASS |
| Native P128/G32, serial and interleaved | 4 | 192 | 768 | 0 | PASS |
| Existing script backend regression, P128/G32 | 1 | 192 | 192 | 0 | PASS |

The three long native runs cover **576 batch-step comparisons / 1,344 token
choices** against previously generated native goldens. Every step compares the
full-vocabulary next-token logits and greedy token; prefill and final steps also
compare the convolution and SSM caches. Three independent prompt groups per
batch run serially and in round-robin order, with separate request-owned caches.
This is cache-isolation qualification on one CPU execution stream, not concurrent
GPU serving. The unchanged script default also passes its full B1 regression.

The source is pinned to
[Transformers cf97f6cfd1c9cff6f3f188e91b32dd3b2a1c5fae](https://github.com/huggingface/transformers/blob/cf97f6cfd1c9cff6f3f188e91b32dd3b2a1c5fae/src/transformers/models/mamba/modeling_mamba.py),
SHA256 `ad28b5f1a464a64d1eabf96b42394f41ef322dbe5f4276e85d4f27368b61a796`.
It is byte-identical to the tested local installation. The adapter extracts
MambaCache and MambaRMSNorm plus the unchanged mixer/block/backbone/LM forward
bodies. It supplies module construction, binds the original methods and uses
the original tuple-return path. Hashes are checked before source execution.
It omits package imports, generation helpers and optional extension discovery;
it does not emulate `mamba_ssm` or claim its CUDA acceleration.

All 242 tensors from the original full safetensors checkpoint were compared
bitwise with the transported state. The tied embedding/output-head storage is
checked and the parameter count is asserted. The artifact has 243 state entries
because the tied head appears twice; no layer or tensor is pruned. The native
runner accepts the true input batch/sequence dimensions independently of the
artifact's B1/P5 tracing metadata. The B2/B4 and P128 tests exercise this behavior.

The short gate records two distinct comparisons: script versus same-device
native, and same-device native versus the original oracle. A successful first
comparison never overwrites a failed second comparison. All original budgets
remain logits `atol=rtol=1e-3`, caches `atol=rtol=1e-5`. The local tests are exact;
they do not resolve the remote Torch2.12.1 versus local Torch2.13 discrepancies.

| Required GPU E2E comparison | Native baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Same-platform Mamba native/script, multiple batch sizes | Not measured | Not measured | N/A |
| High-concurrency serving with accelerated backends | Not measured | Not measured | N/A |

The short controller19672/child19678 and the long controller24301 with all four
children naturally exited0. The runs use the existing CPU Torch2.13 installation
with one thread and offline inputs. No package or environment changes were made.
The official source pin, all input hashes, raw rows, commands and source hashes
are retained. Checkpoints remain required for the unfinished GPU campaign.

Reproduction with the pinned source and existing verified artifacts:

```bash
python tools/flashrnn2/mamba_native_gate.py \
  --model "$MODEL/prefill-decode-b1-t5-cuda.pt" \
  --source "$PINNED_SOURCE/modeling_mamba.py" --config "$MODEL/config.json" \
  --goldens "$MODEL/script-goldens-b1-t5.pt" \
  --checkpoint "$MODEL/model.safetensors" --device cpu \
  --output "$RESULTS/native-short.jsonl"

for batch in 1 2 4; do
  python tools/flashrnn2/mamba_generation_gate.py \
    --backend native --model "$MODEL/prefill-decode-b1-t5-cuda.pt" \
    --native-source "$PINNED_SOURCE/modeling_mamba.py" \
    --native-config "$MODEL/config.json" \
    --goldens "$MODEL/generation-b${batch}-p128-g32.pt" \
    --device cpu --output "$RESULTS/native-b${batch}.jsonl"
done
```

Evidence: [source provenance](evidence/mamba-native-source-manifest.json),
[short comparison rows](evidence/mamba-native-local-r1.jsonl),
[short audit](evidence/mamba-native-local-r1-audit.json),
[long-generation audit](evidence/mamba-native-generation-r1-audit.json),
[actual long-run commands and exits](evidence/mamba-native-generation-r1-controller.json).
Per-batch JSONL, metadata and logs are retained beside these records. The short
snapshot tensor SHA is in the metadata; its binary is retained locally.
