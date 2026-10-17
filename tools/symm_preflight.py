"""Preflight of the symmetric-buffer batch (docs/flashrnn2/scale-out-plan.md section 9), report only.

Records the NCCL versions in this process (PyTorch's own and the opened 2.32), whether every GPU reports multicast
(NVLS) support, and the symmetric-memory session over all GPUs with its peer-table probe (every rank's store through
its LSA table lands in the right rank's buffer). A failure is written to the record, never raised past it.
Usage: symm_preflight.py --out FILE [--devices 0,1,2,3]
"""
import argparse
import ctypes
import json
import os
import time
import traceback
from pathlib import Path

import torch

CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    a = p.parse_args()
    devices = [int(x) for x in a.devices.split(",")]
    rec = dict(started=time.time(), devices=devices, torch_nccl=list(torch.cuda.nccl.version()))
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit(0)
        rec["multicast_supported"] = {}
        for d in devices:
            dev, v = ctypes.c_int(), ctypes.c_int(-1)
            cuda.cuDeviceGet(ctypes.byref(dev), d)
            rc = cuda.cuDeviceGetAttribute(ctypes.byref(v), CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev)
            rec["multicast_supported"][str(d)] = v.value if rc == 0 else f"error {rc}"
    except Exception:  # noqa: BLE001
        rec["multicast_error"] = traceback.format_exc()[-800:]
    try:
        from flashrnn.flashrnn2 import multi_gpu, symm_mem
        rec["peer_probe"] = multi_gpu.enable_peer_access(devices)
        sess = symm_mem.session(devices)
        rec.update(symm_ok=True, nccl232_version=sess.nccl_version, probe=sess.probe)
    except Exception:  # noqa: BLE001
        rec.update(symm_ok=False, error=traceback.format_exc()[-1500:])
    rec["finished"] = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    print("SYMM_PREFLIGHT " + json.dumps({k: rec.get(k) for k in ("torch_nccl", "nccl232_version", "symm_ok",
                                                                  "multicast_supported")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
