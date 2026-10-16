"""Task-local compiler transport with source and flag evidence.

Uses PyTorch's existing distutils backend without installing Ninja or packages.
The build manifest is part of baseline evidence; this is not the stock JIT loader.
"""

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from setuptools import Distribution
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def import_artifact(name: str, artifact: Path, checksum: str) -> ModuleType:
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != checksum:
        raise ValueError("compiled baseline artifact hash mismatch")
    spec = importlib.util.spec_from_file_location(name, artifact)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TaskBuilder:
    def __init__(self, root: Path, source_root: Path) -> None:
        self.root = root.resolve()
        self.source_root = source_root.resolve()

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
        if not with_cuda:
            raise ValueError("this adapter is for CUDA baseline builds")
        manifest = {
            "name": name,
            "sources": {
                str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in sources
            },
            "headers": {
                str(path.relative_to(self.source_root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(self.source_root.rglob("*"))
                if path.suffix in {".h", ".cuh"}
            },
            "cflags": list(extra_cflags),
            "cuda_cflags": list(extra_cuda_cflags),
            "ldflags": list(extra_ldflags),
            "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "automatic"),
            "build_transport": "BuildExtension(use_ninja=False); unique source stems v2",
        }
        digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        module_name = name + "_" + digest[:12]
        if module_name in sys.modules:
            return sys.modules[module_name]
        directory = self.root / digest
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "build-manifest.json"
        if manifest_path.is_file():
            cached = json.loads(manifest_path.read_text())
            if "artifact" in cached:
                return import_artifact(
                    module_name, Path(cached["artifact"]), cached["artifact_sha256"]
                )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        # Distutils otherwise maps gpu_info.cc and gpu_info.cu to the same .o.
        staged_sources = []
        for index, source in enumerate(sources):
            staged = directory / f"{index}_{Path(source).name}"
            staged.write_bytes(Path(source).read_bytes())
            staged_sources.append(str(staged))
        extension = CUDAExtension(
            module_name,
            staged_sources,
            include_dirs=sorted(
                {str(Path(source).resolve().parent) for source in sources}
            ),
            extra_compile_args={
                "cxx": list(extra_cflags),
                "nvcc": list(extra_cuda_cflags),
            },
            extra_link_args=list(extra_ldflags),
        )
        distribution = Distribution({"name": module_name, "ext_modules": [extension]})
        command = BuildExtension(
            distribution, use_ninja=False, no_python_abi_suffix=True
        )
        command.build_temp = str(directory / "objects")
        command.build_lib = str(directory)
        command.verbose = verbose
        command.finalize_options()
        command.run()
        artifact = Path(command.get_ext_fullpath(module_name))
        manifest.update(
            staged_sources=staged_sources,
            artifact=str(artifact),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return import_artifact(module_name, artifact, manifest["artifact_sha256"])
