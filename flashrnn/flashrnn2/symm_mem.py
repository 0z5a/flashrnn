"""Symmetric exchange buffers (NCCL 2.32 ncclMemAlloc + symmetric windows) for the multi-GPU recurrence.

One session per process over a fixed device list (one NCCL communicator per GPU, ncclCommInitAll). A call on P GPUs
uses the first P devices of the session; the peer-address tables come from each window's LSA mapping (symm_mem.cu).
NCCL 2.32 is opened by path (``FRNN_NCCL232_LIB``) next to PyTorch's own NCCL; headers from ``FRNN_NCCL232_INCLUDE``.
Measurement path for docs/flashrnn2/scale-out-plan.md section 9; not served by dispatch.
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name("symm_mem.cu")
_SESSION = None


def build(cache_dir=None):
    cuda_home = os.environ.get("FRNN_HYBRID_CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    include = os.environ["FRNN_NCCL232_INCLUDE"]
    flags = ["-std=c++17", "-O2", "-arch=sm_90", "-shared", "-Xcompiler=-fPIC", "-cudart=static", f"-I{include}"]
    header = (Path(include) / "nccl.h").read_bytes()
    digest = hashlib.sha256(SOURCE.read_bytes() + header + " ".join(flags).encode()).hexdigest()[:16]
    cache = Path(cache_dir or os.environ.get("FRNN_HYBRID_CACHE",
                                             os.environ.get("TMPDIR", str(Path.home() / ".cache")) + "/frnn_hybrid_bw"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"symm_mem_{digest}.so"
    if not lib.exists():
        tmp = lib.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), *flags, str(SOURCE), "-o", str(tmp), "-ldl"],
                       check=True, capture_output=True)
        tmp.replace(lib)
    return lib


class Session:
    def __init__(self, devices):
        self.devices = list(devices)
        self.lib = lib = ctypes.CDLL(str(build()))
        lib.frnn_symm_last_error.restype = ctypes.c_char_p
        lib.frnn_symm_alloc.restype = ctypes.c_int
        lib.frnn_symm_alloc.argtypes = [ctypes.c_size_t]
        lib.frnn_symm_local.restype = ctypes.c_longlong
        lib.frnn_symm_peer_table.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        lib.frnn_symm_zero.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
        lib.frnn_symm_read.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
        devs = (ctypes.c_int * len(self.devices))(*self.devices)
        version = ctypes.c_int(0)
        self._check(lib.frnn_symm_init(os.environ["FRNN_NCCL232_LIB"].encode(), len(self.devices), devs,
                                       ctypes.byref(version)), "init")
        self.nccl_version = version.value
        self.pools = {}  # name -> (buffer id, bytes)
        self.probe = self._probe()

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"symmetric memory {what} failed ({rc}): {self.lib.frnn_symm_last_error().decode()}")

    def buffer(self, name, nbytes):
        """Buffer id of pool ``name`` with at least ``nbytes`` (grown by allocating a larger one)."""
        cur = self.pools.get(name)
        if cur is None or cur[1] < nbytes:
            size = max(nbytes, 2 * cur[1] if cur else nbytes)
            bid = self.lib.frnn_symm_alloc(size)
            if bid < 0:
                self._check(bid, f"alloc {size} B")
            self.pools[name] = (bid, size)
        return self.pools[name][0]

    def table(self, bid, rank):
        """int64 tensor on rank's device: the LSA address of every session rank's copy of the buffer."""
        dev = self.devices[rank]
        t = torch.zeros(len(self.devices), device=f"cuda:{dev}", dtype=torch.int64)
        torch.cuda.synchronize(dev)
        self._check(self.lib.frnn_symm_peer_table(bid, rank, 0, ctypes.c_void_p(t.data_ptr())), "peer table")
        return t

    def zero(self, bid, rank, nbytes):
        dev = self.devices[rank]
        stream = torch.cuda.current_stream(dev).cuda_stream
        self._check(self.lib.frnn_symm_zero(bid, rank, 0, nbytes, ctypes.c_void_p(stream)), "zero")

    def read_int32(self, bid, rank, count):
        host = (ctypes.c_int32 * count)()
        self._check(self.lib.frnn_symm_read(bid, rank, 0, 4 * count, ctypes.cast(host, ctypes.c_void_p)), "read")
        return list(host)

    def local(self, bid, rank):
        return self.lib.frnn_symm_local(bid, rank)

    def _probe(self):
        """Every rank stores rank + 1 into slot rank of every rank's buffer through its table; every buffer must then
        hold [1, 2, ..., P]: table entry j addresses rank j's memory, from every rank."""
        from .multi_gpu import _probe_store

        n = len(self.devices)
        bid = self.buffer("probe", 4 * 64)
        for r in range(n):
            self.zero(bid, r, 4 * 64)
        for d in self.devices:
            torch.cuda.synchronize(d)
        for r, d in enumerate(self.devices):
            torch.cuda.set_device(d)
            _probe_store[(n,)](self.table(bid, r), r + 1, r)
        for d in self.devices:
            torch.cuda.synchronize(d)
        seen = {r: self.read_int32(bid, r, n) for r in range(n)}
        expected = [i + 1 for i in range(n)]
        if any(v != expected for v in seen.values()):
            raise RuntimeError(f"symmetric peer-table probe failed: {seen}")
        return dict(ok=True, seen=seen,
                    local_addresses=[self.local(bid, r) for r in range(n)],
                    table_rank0=self.table(bid, 0).tolist())


def session(devices):
    """The process's session; ``devices`` must be a prefix of the first session's device list."""
    global _SESSION
    if _SESSION is None:
        _SESSION = Session(devices)
    if list(devices) != _SESSION.devices[:len(devices)]:
        raise ValueError(f"devices {devices} are not a prefix of the session's {_SESSION.devices}")
    return _SESSION
