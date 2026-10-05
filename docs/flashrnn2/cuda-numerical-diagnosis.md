# Native CUDA and full Mamba numerical diagnosis

The finite RTX 5090 window completed and returned its resources. The original
alternating CUDA library failed at the first forward call. The complete
Mamba-130M artifact ran all three cases on both the remote CPU and CUDA:
one CPU case and two CUDA cases fail the original native-oracle cache gate.
These are retained failures; no GPU E2E speedup is established.

| Task | Requested / completed comparisons | Actual result | Natural exit |
| --- | --- | --- | ---: |
| Alternating CUDA LSTM BF16, B16/H1/D64, T1/17/128, two losses | 6 / 0 | First forward raises runtime error | 1 |
| Full Mamba-130M, remote Torch 2.12.1 CPU, B1/P5 | 3 / 3 | 2 PASS, 1 prefill SSM failure | 1 |
| Same full artifact, remote Torch 2.12.1 CUDA | 3 / 3 | 1 PASS, 2 prefill SSM failures | 1 |
| cuobjdump ELF/PTX listing and PTX dump | 3 / 3 commands | Six SM80 PTX records; no ELF cubin found | 0 each |

The native library is the independently built seven-source artifact from
[the build report](native-baseline-qualification.md), SHA256
`b2d55255b1a2079f91bf0ce824319d0b3ef0aab30a00060fc7fac9e1e49e05b3`.
The frozen task reuses it without rebuilding. Sources, headers, build manifest
and model inputs were hashed before execution. Its first T1/final call raises
`RuntimeError: Errors during CUDA kernel calls forward.` No comparison row or
gradient result was produced. The raw metadata still says RUNNING because
the exception preceded its final write; the process controller proves the
terminal exit 1. The remaining five cases are not executed, not numerical
failures. The raw diagnostic does not expose the underlying CUDA error code. A [subsequent single-forward probe](native-cuda-ptx-failure.md) reports `cudaErrorInvalidPtx` (218); the original records below remain unchanged.

The binary has six PTX records targeting `sm_80`, PTX version 9.0, and no
ELF device image in the cuobjdump listing. Setting the environment architecture
to 12.0 did not replace the upstream explicit compute-80 PTX flag. This result
does not demonstrate SM120 native machine code, nor does it establish the
cause of the forward failure. The captured build already contains
`-static-global-template-stub=false`; simply adding that flag again would
not test a new hypothesis. NVIDIA documents the relevant linkage default
change in its [compiler update](https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/).

The Mamba experiment uses one unchanged serialized full model, 24 layers /
129,135,360 parameters, SHA256
`e329c77a06bf7d19fb236059c224be484ee33be1b9f2005b637ddac405aebfc0`.
Native goldens are the original independently generated Torch 2.13 CPU
prefill/one-decode outputs, SHA256
`f3ebb63d7b53370b44ba31ee513e85f8e3cbedd27fde7e2301086ea12bb40f41`.
The diagnostic retargets 50 embedded device constants for CPU execution and
zero for CUDA. The same retargeted artifact on the original local Torch 2.13
CPU produces bitwise-equal outputs in all three cases. Remote CPU and CUDA
both use the existing Torch 2.12.1+cu130 installation with one CPU thread and
TF32 disabled. No environment was changed.

| Reference: original native CPU oracle | Case 0 prefill SSM failures | Case 1 prefill SSM failures | Case 2 prefill SSM failures | Other comparisons |
| --- | ---: | ---: | ---: | --- |
| Local Torch 2.13 CPU artifact | 0 | 0 | 0 | All bitwise equal |
| Remote Torch 2.12.1 CPU artifact | 0 | 6 | 0 | All pass |
| Remote Torch 2.12.1 CUDA artifact | 1 | 7 | 0 | All pass |

All greedy tokens, full-vocabulary logits, convolution caches and decode SSM
caches pass the original contract. CPU case 1 failures are in layer 23.
CUDA case 0 has one failure in layer 23; case 1 has one in layer 22 and six
in layer 23. The worst CPU coordinate is `[23,0,1384,15]` at 2.0753 times
the allowed error; CUDA at that coordinate is 2.6444 times its budget.
Budgets remain logits `atol=rtol=1e-3`, states `atol=rtol=1e-5`.

A subsequent local analysis compares the already-saved remote CUDA outputs
directly with the already-saved remote CPU outputs. It reruns no model and
keeps the same budgets. All six tensors per case fit the budgets in this
comparison; the largest normalized error is 0.7980.

| Remote CUDA versus remote CPU | Largest logit error, prefill/decode | Largest SSM error, prefill/decode | Elements outside original budgets |
| --- | ---: | ---: | ---: |
| Case 0 | 2.28882e-4 | 3.05176e-5 | 0 |
| Case 1 | 1.37329e-4 | 3.05176e-5 | 0 |
| Case 2 | 1.29700e-4 | 1.16825e-5 | 0 |

This shows that the original reference discrepancy is not exclusive to CUDA.
It does not isolate PyTorch version, CPU architecture, math libraries or
their interaction. Two executions of the same scripted artifact are not an
independent native baseline, so this comparison does not replace the failed
native-oracle qualification or justify changing tolerances.

| GPU performance comparison | Baseline tok/s | Candidate tok/s | Speedup |
| --- | --- | --- | --- |
| Full Mamba multi-BS/high-concurrency | Not measured | Not measured | N/A |
| Native alternating CUDA LSTM | No completed forward qualification | Not measured | N/A |

The controller waited for all three children despite nonzero results.
Controller 23333, children 23340/23350/23354 and local SSH 12967 all naturally
exited 1. The 39,430,307-byte results archive has SHA256
`44ba19eb8cdb4f9084c01a50e153c3c895e8c9f510dfaf9474dc89ce06e5090a`;
all 19 members were independently verified after transfer. Final checks found
no task processes, no GPU1 compute process and no mappings/open descriptors
for this task's model files. The original GPU1 and IO lock inodes were taken
and released. Completion marker SHA256:
`a5ae231acef33df268d8cfd571cccf83d3df95ac9abefdb79fcc3192d26666d9`.

Reproduction commands are preserved in the
[frozen manifest](evidence/numerical-window-r2-manifest.json).
For saved-snapshot comparison, use:

```bash
python tools/flashrnn2/compare_mamba_diagnostics.py \
  --actual "$RESULTS/mamba-script-diagnostic-cuda-remote-r1.pt" \
  --reference "$RESULTS/mamba-script-diagnostic-cpu-remote-r1.pt" \
  --output "$RESULTS/mamba-cuda-vs-remote-cpu-r1.json"
```

Evidence: [controller](evidence/numerical-window-r2-controller.json),
[native CUDA traceback](evidence/upstream-cuda-r1.log),
[remote CPU](evidence/mamba-script-diagnostic-cpu-remote-r1.jsonl),
[remote CUDA](evidence/mamba-script-diagnostic-cuda-remote-r1.jsonl),
[saved-snapshot comparison](evidence/mamba-cuda-vs-remote-cpu-r1.json),
[completion marker](evidence/numerical-window-r2-complete.json).
Matching metadata, logs, binary-ISA listings and archive manifest are retained
beside these records; snapshot tensors remain in the verified local archive.
