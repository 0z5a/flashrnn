# Thor SM110 TMEM qualification

Thor is a separate experimental target for FlashRNN2's Blackwell TMEM path.
NVIDIA lists Jetson AGX Thor as compute capability 11.0 and the PTX ISA lists
`tcgen05.alloc`, `st`, `ld`, `wait`, and `dealloc` for `sm_110a`/`sm_110f`.
The `sm_110a` target must be tested with the existing Thor toolchain; generic
`sm_110` is not evidence for architecture-specific instructions. SM110 data
does not establish the original plan's B200 SM100 result.

The first finite gate is [thor_tmem_roundtrip.cu](../../tools/flashrnn2/thor_tmem_roundtrip.cu):
one CTA, one warp, 32 TMEM columns. Each lane writes a unique 32-bit value to
TMEM, waits for the store, reads it back, waits for the load, checks all 32
values on the host, then deallocates and relinquishes the permit. It does not
claim BF16 MMA, recurrent correctness, or speedup.

Before execution, the Thor resource owner must finish or hand back the current
Qwen/A3 GPU window and grant a separate FlashRNN slot on the original shared
lock. The executor records the current boot ID, GPU UUID, lock inode and owner,
and existing `nvcc`/driver versions. No package or environment change is part
of this gate. Compile only after the existing compiler is identified, using
`nvcc -std=c++17 -O2 -arch=sm_110a -Xptxas=-v` and a task-local output path.
Inspect the generated image for `tcgen05` and run exactly one CTA under the
granted lock. Preserve compile output, program output, exit status, compiler
resource report, and source/image SHA256. A compile or runtime failure is a
result to record, not a reason to change the shared environment or repeat
unknown GPU work.

| Thor gate | Result | Scope |
| --- | --- | --- |
| Existing compiler accepts `sm_110a` and PTX | UNRUN | Compile only |
| 32-lane alloc/store/load/dealloc roundtrip | UNRUN | Correctness only |
| BF16 `tcgen05.mma` with FP32 accumulator | UNRUN | Separate future gate |
| R-SMEM versus R-TMEM, both operand orientations | UNRUN | Complete recurrent loop required |
| High-batch/high-concurrency full-model E2E speedup | UNMEASURED | Same-workload baseline required |

Sources: [Jetson Thor CUDA setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/0.1.0/setup_cuda.html),
[CUDA compute-capability targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html),
[PTX tensor memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory),
and [CUTLASS tcgen05 BF16 operation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html).
