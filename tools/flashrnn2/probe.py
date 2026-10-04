"""Record the existing runtime without installing or modifying dependencies."""

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    installed = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in metadata.distributions()
    }
    packages = (
        "torch",
        "triton",
        "einops",
        "ninja",
        "tilelang",
        "pytest",
        "nvidia-cutlass-dsl",
    )
    tools = {}
    for name in ("nvcc", "ptxas", "ninja", "nsys", "compute-sanitizer"):
        path = shutil.which(name)
        cuda_path = Path("/usr/local/cuda/bin") / name
        if path is None and cuda_path.is_file():
            path = str(cuda_path)
        if path is None:
            tools[name] = {"status": "MISSING"}
        else:
            result = subprocess.run(
                [path, "--version"], text=True, capture_output=True, check=False
            )
            tools[name] = {
                "path": path,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
    record = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {name: installed.get(name) for name in packages},
        "tools": tools,
        "mode": "environment_only" if args.environment_only else "device_properties",
        "instruction_probes": "NOT_RUN",
        "kernel_function_attributes": "NOT_RUN",
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        record["torch_cuda_runtime"] = torch.version.cuda
        if not args.environment_only:
            devices = []
            for index in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": prop.name,
                        "cc": [prop.major, prop.minor],
                        "sm_count": prop.multi_processor_count,
                        "memory_bytes": prop.total_memory,
                    }
                )
            record["devices"] = devices
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
