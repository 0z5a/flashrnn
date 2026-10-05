"""Get driver JIT logs for the six PTX records already captured from the library."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    raw = args.dump.read_bytes()
    blocks = raw.decode().split("Fatbin ptx code:\n================\n")[1:]
    assert len(blocks) == 6 and all(".version 9.0" in block for block in blocks)
    torch.empty(1, device="cuda")
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libcuda.so." in line
    }
    assert len(paths) == 1, paths
    library = paths.pop()
    driver = ctypes.CDLL(library, mode=os.RTLD_NOLOAD)
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    driver.cuModuleLoadDataEx.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    driver.cuModuleLoadDataEx.restype = ctypes.c_int
    driver.cuModuleUnload.argtypes = [ctypes.c_void_p]
    driver.cuModuleUnload.restype = ctypes.c_int
    driver.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorName.restype = ctypes.c_int
    version = ctypes.c_int()
    assert driver.cuDriverGetVersion(ctypes.byref(version)) == 0
    rows = []
    for index, block in enumerate(blocks):
        ptx = (".version" + block.split(".version", 1)[1]).encode()
        image = ctypes.create_string_buffer(ptx)
        error = ctypes.create_string_buffer(8192)
        # CU_JIT_ERROR_LOG_BUFFER / CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES.
        options = (ctypes.c_int * 2)(5, 6)
        values = (ctypes.c_void_p * 2)(ctypes.addressof(error), len(error))
        module = ctypes.c_void_p()
        result = driver.cuModuleLoadDataEx(
            ctypes.byref(module),
            ctypes.cast(image, ctypes.c_void_p),
            2,
            options,
            values,
        )
        name = ctypes.c_char_p()
        assert driver.cuGetErrorName(result, ctypes.byref(name)) == 0
        row = {
            "record": index + 1,
            "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
            "returncode": result,
            "error_name": name.value.decode(),
            "jit_error_log": error.value.decode(),
            "kernel_launched": False,
        }
        if result == 0:
            row["unload_returncode"] = driver.cuModuleUnload(module)
            assert row["unload_returncode"] == 0
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "scope": "SIX_CAPTURED_PTX_RECORDS_DRIVER_JIT_LOGS_NO_KERNEL_LAUNCH",
        "driver_library": library,
        "driver_api_version": version.value,
        "torch": torch.__version__,
        "dump_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "performance_claim": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
