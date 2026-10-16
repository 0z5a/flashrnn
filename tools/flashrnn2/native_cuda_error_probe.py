"""Expose the CUDA runtime error behind the retained first-forward failure."""

import argparse
import ctypes
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import torch
from build_without_ninja import import_artifact

from flashrnn.flashrnn import cuda_init
from flashrnn.flashrnn.flashrnn import FlashRNNConfig, flashrnn


class ExistingBuild:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest = json.loads(manifest_path.read_text())

    def load(
        self,
        name: str,
        sources: Sequence[str],
        *,
        extra_cflags: Sequence[str],
        extra_cuda_cflags: Sequence[str],
        extra_ldflags: Sequence[str],
        verbose: bool,
        with_cuda: bool,
    ) -> ModuleType:
        manifest = self.manifest
        assert with_cuda and name == manifest["name"]
        assert list(extra_cflags) == manifest["cflags"]
        assert list(extra_cuda_cflags) == manifest["cuda_cflags"]
        assert list(extra_ldflags) == manifest["ldflags"]
        assert {
            str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in sources
        } == manifest["sources"]
        path = Path(manifest["artifact"])
        return import_artifact(path.stem, path, manifest["artifact_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32 = False
    cuda_init._load = ExistingBuild(args.build_manifest).load
    os.environ["CUDA_LIB"] = str(Path(cuda_init.CUDA_HOME) / "lib64")
    config = FlashRNNConfig(
        function="lstm",
        backend="cuda",
        dtype="bfloat16",
        batch_size=16,
        hidden_dim=64,
        num_heads=1,
    )
    shapes = ((16, 1, 4, 1, 64), (4, 1, 64, 64), (4, 1, 64), (2, 16, 1, 1, 64))
    inputs = [(torch.randn(shape) * 0.05).bfloat16() for shape in shapes]
    inputs[1].div_(8)
    inputs = [value.cuda().requires_grad_() for value in inputs]
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libcudart.so." in line
    }
    assert len(paths) == 1, paths
    library_path = paths.pop()
    runtime = ctypes.CDLL(library_path, mode=os.RTLD_NOLOAD)
    runtime.cudaPeekAtLastError.argtypes = []
    runtime.cudaPeekAtLastError.restype = ctypes.c_int
    runtime.cudaGetErrorName.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorName.restype = ctypes.c_char_p
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    result = {
        "scope": "ONE_FIRST_FORWARD_ERROR_CODE_PROBE_NO_TIMING",
        "pid": os.getpid(),
        "started": time.time(),
        "runtime_library": library_path,
        "before_cuda_error": runtime.cudaPeekAtLastError(),
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        flashrnn(inputs[0], inputs[1], inputs[2], inputs[3], config=config)
    except RuntimeError as error:
        result["forward_exception"] = str(error)
    else:
        result["forward_exception"] = None
    code = runtime.cudaPeekAtLastError()
    result.update(
        after_cuda_error=code,
        cuda_error_name=runtime.cudaGetErrorName(code).decode(),
        cuda_error_string=runtime.cudaGetErrorString(code).decode(),
        finished=time.time(),
        numerical_qualification=False,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
