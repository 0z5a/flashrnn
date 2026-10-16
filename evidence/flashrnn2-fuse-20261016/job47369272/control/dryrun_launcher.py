"""Login-node check of the GPU-free part of Triton's launch path: cuda_utils and the _rtiled launcher.

Builds both C modules with the job's $CC, exactly as the first kernel launch would.
"""
import json
import os
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import CudaLauncher, CudaUtils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "tools"))
from rtiled_compile_sweep import _rtiled, k_parts, SPIN_LIMIT  # noqa: E402


def main():
    rec = dict(cc=os.environ.get("CC"), cxx=os.environ.get("CXX"), triton=triton.__version__)
    CudaUtils()
    rec["cuda_utils"] = "BUILT"
    k1, k2 = k_parts(768)
    sig = dict(WX="*fp16", R="*fp16", BIAS="*fp16", INITIAL="*fp32", OUTPUT="*fp32", FINAL="*fp32",
               XBUF="*fp16", SYNC="*i32", ERR="*i32", ZERO_NORMALIZER="*i1", B="i32", T="i32", TOTAL_T="i32",
               STEP_BASE="i32", SEQUENCE_START="i32", GROUPS="i32", SLOTS="i32", PERSISTENT="i32",
               H="constexpr", D="constexpr", SLSTM="constexpr", BM="constexpr", S="constexpr", K1="constexpr",
               K2="constexpr", OUTPUT_MODE="constexpr", SPIN="constexpr", VERIFY="constexpr")
    const = dict(H=1, D=768, SLSTM=True, BM=16, S=32, K1=k1, K2=k2, OUTPUT_MODE=0, SPIN=SPIN_LIMIT, VERIFY=0)
    src = triton.compiler.ASTSource(fn=_rtiled, signature=sig, constexprs=const)
    k = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=4, enable_fp_fusion=False, launch_cooperative_grid=True))
    CudaLauncher(src, k.metadata)
    rec["rtiled_launcher"] = "BUILT"
    rec["launch_cooperative_grid"] = bool(k.metadata.launch_cooperative_grid)
    print("DRYRUN_LAUNCHER " + json.dumps(rec))


if __name__ == "__main__":
    main()
