# CUDA 13 baseline build qualification

The pinned GPU-info extension does not compile with CUDA 13: eight deprecated
`cudaDeviceProp` members were removed. The source patch uses the replacement
attribute APIs and `asyncEngineCount` listed in NVIDIA's
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).
It omits `maxTexture1DLinear` and `cooperativeMultiDeviceLaunch` on CUDA 13;
the former has no descriptor-independent replacement and the latter was
removed without replacement. Older CUDA retains both fields.

The recurrence selector reads `maxThreadsPerBlock`, `regsPerMultiprocessor`,
`multiProcessorCount`, `sharedMemPerBlockOptin` (with the first two reused
for forward/backward). Those query expressions are unchanged. Recurrence
source and compilation flags are unchanged. This is a compatibility-patched
baseline, not an unmodified upstream build.

The existing environment lacks Ninja. `build_without_ninja.py` supplies a
task-local `BuildExtension(use_ninja=False)` transport using the installed
compiler and setuptools. It stages byte-identical sources under unique stems
to avoid the `.cc`/`.cu` object-name collision and records source/header hashes,
flags and the resulting shared-library hash. No packages are installed.

| Attempt | Source / transport | Actual result |
| --- | --- | --- |
| r1 | Original GPU-info; same-stem distutils objects | Failed: duplicate symbols at link |
| r2 | Original GPU-info; unique source stems | Failed: removed CUDA 13 struct members |
| r3 | Explicit GPU-info compatibility patch; unique source stems | Build and import PASS; controller naturally exited 0 in 28.17 s |
| r4 | Original seven-unit alternating LSTM build invocation | Failed at project import, before compilation; missing task PYTHONPATH |
| r5 | Same seven-unit sources/flags, corrected process PYTHONPATH | Fresh homogeneous-BF16 SM120 build/import PASS; natural exit 0 in 38.26 s |

Evidence: [r1 log](evidence/baseline-build-r1.log),
[r2 log](evidence/baseline-build-r2.log),
[r3 log](evidence/baseline-build-r3.log),
[r3 controller](evidence/baseline-build-r3-controller.json), and
[r3 build manifest](evidence/baseline-build-r3-manifest.json).
The run hid CUDA devices and did not call the compiled GPU-info query or recurrence kernel.
The later [alternating recurrence build](native-baseline-qualification.md)
now compiles and imports successfully. Device-query runtime qualification,
CUDA 12 compilation, numerical calibration and timing remain untested by
these compile-only checks.

The separate CPU-only [dtype probe](evidence/resolved-dtype-contracts.json)
confirms the default and explicitly selected configuration fields. It does
not establish numerical equivalence; see [contracts](numerical-contracts.md).
