"""Build the pinned Haste LSTM/GRU CUDA bindings for an assigned H100."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

SOURCE_SHA256 = {
    "frameworks/pytorch/lstm.cc": "ff5e9bb160876f0f7ba3f018a800e32b42945b8112eebfdf73cc3267bf9784c6",
    "frameworks/pytorch/gru.cc": "a8aaeb9df1ced4b747675382213d357c4d5043f7944935a6b68d44dbc4fc09f7",
    "frameworks/pytorch/support.h": "7ecadb5a72e899d65d521d9ae0f478d125c523eb8acf43a682a245607353caeb",
    "lib/lstm_forward_gpu.cu.cc": "724b7bcfaa566571c4d23128577e19ea55cd301c09de96028823830e93ad2e66",
    "lib/lstm_backward_gpu.cu.cc": "ae27797b69c61981d4669153243b4e6ad97b129448a9c5dda0de1ea57844fc4e",
    "lib/gru_forward_gpu.cu.cc": "b57f425e591eee8d8cb935bed2eab9e19f23807908e650efcddec8f0b80b5ef1",
    "lib/gru_backward_gpu.cu.cc": "cb1b936ce55291ee3065ee8b1f0850b29bc8344832e9d7bb4717612336585d54",
    "lib/blas.h": "7affb68391dcd743fd942c262c381e65740d20112fbde0297e974e99781f77db",
    "lib/device_assert.h": "5b316a6a1147e26ae5832ae7cf64edfb53103fc9cf1b6bf9c85bad4410028fa0",
    "lib/haste.h": "e019932b8c5fb042339f443f4517f6177e176a797ad82ece03bb3c3746f1e9df",
    "lib/haste/gru.h": "380379082d17d1a8672cb1bb076b705dabcd2e7b8250ddf6c36e952d4fbaa98b",
    "lib/haste/indrnn.h": "e4bda87e7da8895598c39ddc38dd31f523f7e3f1249e40b78d8bbc2a6da1f470",
    "lib/haste/layer_norm.h": "16ad4a9328e3a0e6dead383445d86582ba4c4fcebc563f84d7046488ff61ff98",
    "lib/haste/layer_norm_gru.h": "82157bfe0ca384efb842bf879ba7660c37e969d6e83136ff8e613af9570fa0a6",
    "lib/haste/layer_norm_indrnn.h": "16aafe777b8b380d08f9b8a74b287fe84c31ab751972d2cd69c77aa5c604d512",
    "lib/haste/layer_norm_lstm.h": "b18b7fd686180d966aa8dabc81ae33c78ff22c79ce334d4b343b72e6f1675fd5",
    "lib/haste/lstm.h": "936c10ddcd75dd531f08d30b4d1abb158ffd7577bac59f6ac0b5b0642e32b760",
    "lib/inline_ops.h": "45ecd55f7b20e1e0b3ad1bda60bf0e20788a0fed242fbe5d52219d033e0f58d3",
}

REGISTRATION = """#include <torch/extension.h>
void lstm_init(py::module&);
void gru_init(py::module&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  lstm_init(module);
  gru_init(module);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for name, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((args.source_root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Haste source mismatch: {name}")
    if args.check_only:
        print(json.dumps({"status": "SOURCE_VERIFIED", "files": len(SOURCE_SHA256)}))
        return
    if args.build_root is None:
        parser.error("--build-root is required for a build")
    args.build_root.mkdir(parents=True, exist_ok=False)
    sources = [
        args.source_root / "frameworks/pytorch/lstm.cc",
        args.source_root / "frameworks/pytorch/gru.cc",
    ]
    for cell in ("lstm", "gru"):
        for direction in ("forward", "backward"):
            name = f"{cell}_{direction}_gpu"
            staged = args.build_root / f"{name}.cu"
            shutil.copyfile(args.source_root / "lib" / f"{name}.cu.cc", staged)
            sources.append(staged)
    registration = args.build_root / "haste_lstm_gru_module.cc"
    registration.write_text(REGISTRATION)
    sources.append(registration)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    from torch.utils.cpp_extension import load

    module = load(
        name="haste_lstm_gru_native",
        sources=[str(path) for path in sources],
        extra_include_paths=[
            str(args.source_root / "lib"),
            str(args.source_root / "frameworks/pytorch"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcublas"],
        build_directory=str(args.build_root),
        with_cuda=True,
    )
    print(
        json.dumps(
            {
                "status": "BUILT_AND_IMPORTED_NOT_EXECUTED",
                "source_files_verified": len(SOURCE_SHA256),
                "cuda_arch": "sm_90",
                "functions": sorted(
                    name
                    for name in dir(module)
                    if name.endswith(("_forward", "_backward"))
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
