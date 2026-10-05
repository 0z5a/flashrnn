# Native CUDA first-forward PTX failure

The previously retained alternating CUDA LSTM forward failure reports
`cudaErrorInvalidPtx` (218): `a PTX JIT compilation failed`. This narrows the
failure to PTX JIT compilation. It does not identify the rejected PTX statement
or establish numerical correctness; driver JIT logs remain the next diagnostic.

| One frozen B16/H1/D64/T1 BF16 forward | Observed result |
| --- | --- |
| CUDA runtime error before forward | 0 |
| Forward call | Raises `Errors during CUDA kernel calls forward.` |
| CUDA runtime error after forward | 218, `cudaErrorInvalidPtx` |
| Completed numerical comparisons / gradients | 0 / 0 |
| Diagnostic child / controller / SSH exit | 0 / 0 / 0 |

The diagnostic process exits successfully because it records the expected
forward exception and runtime error. That exit is **not a passing forward**.
The six-case numerical campaign in [the preceding report](cuda-numerical-diagnosis.md)
remains unqualified. This follow-up repeats only its first T1 input shape,
using the same seed and homogeneous BF16 contract.

The probe imports the existing seven-source library only after its name,
compiler flags, source hashes and artifact SHA match the frozen build manifest.
The controller also checks all retained headers. There is no build fallback.
Library SHA256 is
`b2d55255b1a2079f91bf0ce824319d0b3ef0aab30a00060fc7fac9e1e49e05b3`.
It resolves the already-loaded CUDA runtime from process mappings and uses
`RTLD_NOLOAD`; `cudaPeekAtLastError` does not clear the observed runtime status.

The library's previously archived ISA listing contains six SM80 PTX9.0
records and no ELF cubin. Its existing build already includes
`-static-global-template-stub=false`. Neither this inventory nor error218
alone establishes whether a source defect, unresolved symbol, unsupported
instruction or another JIT issue caused the failure. No driver or environment
change was made.

| Performance scope | Baseline latency | Candidate latency | Speedup |
| --- | --- | --- | --- |
| Alternating CUDA LSTM | Forward qualification failed | Not measured | N/A |
| Full-model multi-batch/high-concurrency E2E | Not measured | Not measured | N/A |

Controller24929, child24934 and local SSH18900 naturally exited0. The archived
seven members total 3,659 compressed bytes, SHA256
`c90017abcbd91e42eacd2bc284642caee295b588dda28cd14bad3637ee7c0a31`.
Each file hash was independently checked after transfer. Before resource return,
the task PIDs were absent, GPU1 had no compute process and this task's model
files had no mapping/open-descriptor readers. The original GPU1/IO lock inodes
12895501695/12895501666 were taken and released. The completion marker SHA256 is
`511a76c54a54fb5deea862c9a0d7c5772e36da233b406eeca8ce1bf00e71c99c`.
Model inputs remain required by the unfinished campaign.

Reproduce with the same existing build and verified source tree:

```bash
PYTHONPATH="$SOURCE" python "$SOURCE/tools/flashrnn2/native_cuda_error_probe.py" \
  --build-manifest "$BUILD/build-manifest.json" \
  --output "$RESULTS/native-cuda-error.json"
```

Evidence: [runtime result](evidence/native-cuda-error-r1.json),
[raw log](evidence/native-cuda-error-r1.log),
[frozen manifest](evidence/native-error-window-r1-manifest.json),
[controller](evidence/native-error-window-r1-controller.json),
[archive manifest](evidence/native-error-window-r1-archive-manifest.json),
[completion marker](evidence/native-error-window-r1-complete.json).
