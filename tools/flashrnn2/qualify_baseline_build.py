"""Build/import GPU-info or an upstream alternating CUDA recurrence."""

import argparse
import json
import os
from pathlib import Path

from build_without_ninja import TaskBuilder

from flashrnn.flashrnn import cuda_init
from flashrnn.flashrnn.flashrnn import FlashRNNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--target", choices=("gpu-info", "cuda"), default="gpu-info")
    parser.add_argument(
        "--cell", choices=("lstm", "slstm", "gru", "elman"), default="lstm"
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    source_root = Path(cuda_init.__file__).parent
    cuda_init._load = TaskBuilder(args.build_root, source_root).load
    # CUDA is hidden for this compile-only qualification; no device query runs.
    os.environ["CUDA_LIB"] = str(Path(cuda_init.CUDA_HOME) / "lib64")
    if args.target == "gpu-info":
        source = source_root / "gpu_info"
        module = cuda_init.load(
            name="gpu_info2",
            sources=[str(source / "gpu_info.cc"), str(source / "gpu_info.cu")],
        )
        resolved = None
    else:
        config = FlashRNNConfig(
            function=args.cell,
            backend="cuda",
            dtype=args.dtype,
            batch_size=16,
            hidden_dim=64,
            num_heads=1,
        )
        sources = [
            "alternating/flashrnn.cc",
            "alternating/flashrnn_forward.cu",
            "alternating/flashrnn_backward.cu",
            "alternating/flashrnn_backward_cut.cu",
            f"alternating/{args.cell}_pointwise.cu",
            "util/blas.cu",
            "util/cuda_error.cu",
        ]
        module = cuda_init.load(
            name=config.function,
            sources=[str(source_root / name) for name in sources],
            extra_cflags=[f"-D{k}={v}" for k, v in config.constants.items()]
            + config.defines,
        )
        resolved = {
            "dtype_a": config.dtype_a,
            "dtype_s": config.dtype_s,
            "defines": config.defines,
        }
    print(
        json.dumps(
            {
                "status": "BUILT_AND_IMPORTED_NOT_EXECUTED",
                "artifact": module.__file__,
                "target": args.target,
                "cell": args.cell,
                "resolved_config": resolved,
                "device_query_executed": False,
                "recurrence_executed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
