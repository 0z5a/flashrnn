# KDA and signed Complex KDA: paired full-checkpoint CPU numerical screen

The public [signed Complex KDA](https://huggingface.co/openeurollm/complex-kda-1.3B-100B) and [sigmoid KDA](https://huggingface.co/openeurollm/kda-sigmoid-1.3B-100B) checkpoints were pinned to revisions `5ca8cba4c02b4a8fbcb3684532348ef5d7675c63` and `88510f58bd0bba11104869f9d83d4006e083b037`. Each verified 2,725,315,744-byte BF16 weight strictly loads all 507 tensors into its author's 24-layer, 1,362,630,016-parameter `ComplexKDAForCausalLM`. The pinned model code and tokenizer are byte-identical across the two repositories; the configs select `signed_sigmoid2`/negative eigenvalues or `sigmoid`/nonnegative eigenvalues. The weights differ, so this comparison does not isolate a gate change.

The author's portable Torch implementation ran with `COMPLEX_KDA_BACKEND=torch` in the existing local Python environment. No package or environment was changed. For the same fixed token sequence, the screen compared one full-prefix forward pass with one-token-at-a-time cached forward passes, including the final 32,000 logits and all 24 recurrent matrices plus three short-convolution histories per layer. The five-token full path uses `recurrent_kda_torch`; the 16-token full path uses `chunk_kda_torch`; each cached one-token step uses the recurrent path. This is a complete-model numerical screen at batch size 1, not a generation-throughput or training qualification.

The BF16 gate was frozen at `atol=rtol=0.02` and the FP32-upcast diagnostic at `atol=rtol=1e-4`, with an element passing when `abs(actual-reference) <= atol + rtol*abs(reference)`. No gate was relaxed after observing failures. Each run's independently saved raw-tensor audit passed, including exact snapshot SHA-256 verification, all logits, 96 state pairs and the greedy token.

| Checkpoint | Input | BF16 failed logits / 32,000 | BF16 max abs | FP32 failed logits / 32,000 | FP32 max abs | Final state pairs | Greedy token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Signed Complex KDA | B1/P5 | 159 | 0.125 | 0 | 0.0000276566 | 96/96 in both dtypes | Same |
| Signed Complex KDA | B1/P16 | 248 | 0.09375 | 0 | 0.0000205040 | 96/96 in both dtypes | Same |
| Sigmoid KDA | B1/P5 | 255 | 0.09375 | 0 | 0.0000267029 | 96/96 in both dtypes | Same |
| Sigmoid KDA | B1/P16 | 216 | 0.09375 | 0 | 0.0000295639 | 96/96 in both dtypes | Same |

The BF16 numerical status is **failed** for all four inputs, despite matching greedy choices and passing final-state tensors. The FP32-upcast control passes all four inputs. The P16 failures include a chunk-versus-recurrent comparison, while the P5 failures show that the BF16 discrepancy is not confined to chunk mode. This screen does not identify the first divergent layer or establish whether any native CUDA path has the same behavior.

The [manifest](evidence/kda-numerics/manifest.json), eight [run records](evidence/kda-numerics/complex-kda-bf16-probe-r1.json) and adjacent independent `.audit.json` files retain the pinned hashes, budgets, failed-element counts, natural numerical exits and raw snapshot digests. The 50–54 MB raw tensor snapshots remain in the local task workspace and are not committed. Both weights were evicted only after the CPU screens, audits, exact SHA-256 rechecks and no-reader checks; the two eviction receipts are in the same evidence directory. Later native GPU tests require pinned re-download.

To reproduce a row, download the manifest's exact model revision into a local directory, then run `python tools/flashrnn2/complex_kda_numerics_probe.py --model-source MODEL_DIR --checkpoint MODEL_DIR/model.safetensors --checkpoint-sha256 MANIFEST_SHA256 --dtype bf16 --batch-size 1 --prompt-tokens 16 --output RESULT.json`. Run `python tools/flashrnn2/audit_complex_kda_probe.py --result RESULT.json --snapshot RESULT.pt --output RESULT.audit.json` to recheck the saved tensors. The probe sets the portable Torch backend itself.

| Required full-model GPU E2E comparison | Workload | Baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Signed KDA author fused-recurrent/chunk kernels versus FlashRNN candidate | BS1/4/16/32/64; concurrency 1/8/32/64/128 | — | — | Unmeasured |
| Sigmoid KDA author fused-recurrent/chunk kernels versus FlashRNN candidate | Same paired request traces | — | — | Unmeasured |
| Hopper SM90 and Blackwell SM100 dedicated paths | Prefill, cached decode and backward | — | — | Unmeasured |

The native author kernels, backward, high-concurrency full-model E2E, Hopper and B200 paths remain unqualified. The original BF16 failures are retained rather than counted as native baseline passes.
