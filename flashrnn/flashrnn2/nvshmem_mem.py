"""NVSHMEM 3.9 host helper (nvshmem_mem.cu) for the one-process-per-GPU multi-GPU runs (tools/scale_nvshmem.py).

Build: nvcc with ``FRNN_NVSHMEM_HOME``/include, linked against its libnvshmem_host (rpath). Measurement path for
docs/flashrnn2/scale-out-plan.md section 11; not served by dispatch.
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("nvshmem_mem.cu")


def build(cache_dir=None):
    home = os.environ["FRNN_NVSHMEM_HOME"]
    # nvshmemx_init_attr (inline in the headers) calls into libnvshmem_device.a: relocatable device code, linked with
    # the toolkit that built that library (CUDA 13.2; 12.8's nvlink rejects its ABI).
    cuda_home = os.environ.get("FRNN_NVSHMEM_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    flags = ["-std=c++17", "-O2", "-arch=sm_90", "-rdc=true", "-shared", "-Xcompiler=-fPIC", "-cudart=static",
             f"-I{home}/include"]
    link = [f"-L{home}/lib", "-lnvshmem_host", "-lnvshmem_device", "-lcuda", "-Xlinker", "-rpath", "-Xlinker",
            f"{home}/lib"]
    header = (Path(home) / "include" / "nvshmem.h").read_bytes()
    digest = hashlib.sha256(SOURCE.read_bytes() + header + " ".join(flags + link).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"nvshmem_mem_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), *flags, str(SOURCE), "-o", str(tmp), *link],
                       check=True, capture_output=True)
        tmp.replace(lib)
    return lib


def load():
    lib = ctypes.CDLL(str(build()))
    lib.frnn_shm_malloc.restype = ctypes.c_longlong
    lib.frnn_shm_malloc.argtypes = [ctypes.c_size_t]
    lib.frnn_shm_ptr.restype = ctypes.c_longlong
    lib.frnn_shm_ptr.argtypes = [ctypes.c_longlong, ctypes.c_int]
    lib.frnn_shm_zero.argtypes = [ctypes.c_longlong, ctypes.c_size_t]
    lib.frnn_shm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return lib
