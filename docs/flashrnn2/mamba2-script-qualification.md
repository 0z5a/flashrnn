# Complete Mamba2 TorchScript export

The existing GPU runtime has Torch and no Transformers. This static export
preserves the complete Mamba2 checkpoint and exposes prefill/decode methods
with request-owned convolution and SSM caches. It uses the existing
[strict checkpoint adapter](mamba2-qualification.md), retaining all 24 layers,
128,989,632 parameters and the unchanged Transformers `torch_forward` path.

The checkpoint pin is `state-spaces/mamba2-130m` at
`3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49`. Both original files are rechecked
against the recorded immutable size/hash manifest before loading. The original
FP16 weights load into the FP32 model through the same strict 219-key mapping.

| CPU check | Inputs | Prefill maximum errors: logits / conv / SSM | Decode maximum errors | Result |
| --- | --- | --- | --- | --- |
| Native vs traced, case 0 | B1/P5 | 0 / 0 / 0 | 0 / 0 / 0 | PASS |
| Native vs traced, case 1 | B1/P5 | 0 / 0 / 0 | 0 / 0 / 0 | PASS |
| Native vs traced, case 2 | B1/P5 | 0 / 0 / 0 | 0 / 0 / 0 | PASS |

All three independent inputs use a different seed from tracing. Decode runs
one greedy step after prefill, with separate cloned native/traced caches.
The controller naturally exits 0. The saved 516,690,916-byte artifact has SHA256
`ff7d5d2eca96916fb7259d64c9153539272e341b83453583745571ee09e70492`, independently
verified after execution. Weights and the binary artifact are not published.

Tracing fixes B1, prompt length 5 and the prefill/decode branches. The retained
TracerWarnings correspond to these branch/shape decisions. Dynamic shapes,
other batches, device-literal portability, cross-version loading, CUDA
execution and longer generation are not qualified by this run. It does not
implement the native `mamba_ssm` CUDA path or revise the earlier strict
cached/full-prefix logit failures.

| Full-model CUDA multi-BS/high-concurrency comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Mamba2-130M | Not measured | Not measured | N/A |

Reproduction in the existing Torch 2.13 / Transformers 4.54.1 environment:

```sh
python tools/flashrnn2/export_mamba2.py --model MAMBA2_DIR \
  --batch 1 --prompt-length 5 --output prefill-decode-b1-t5.pt
```

See [source and artifact metadata](evidence/mamba2-script-cpu-export.meta.json),
[log](evidence/mamba2-export-r1.log), and
[natural exit](evidence/mamba2-export-r1-controller.json).
