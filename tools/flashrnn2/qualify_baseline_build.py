"""Qualify the CUDA-13-compatible GPU-info source using existing compilers."""

import argparse
import json
import os
from pathlib import Path

from build_without_ninja import TaskBuilder

from flashrnn.flashrnn import cuda_init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(cuda_init.__file__).parent
    cuda_init._load = TaskBuilder(args.build_root, source_root).load
    source = source_root / "gpu_info"
    # CUDA is hidden for this compile-only qualification; no device query runs.
    os.environ["CUDA_LIB"] = str(Path(cuda_init.CUDA_HOME) / "lib64")
    module = cuda_init.load(
        name="gpu_info2",
        sources=[str(source / "gpu_info.cc"), str(source / "gpu_info.cu")],
    )
    print(
        json.dumps(
            {
                "status": "BUILT_AND_IMPORTED_NOT_EXECUTED",
                "artifact": module.__file__,
                "device_query_executed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
