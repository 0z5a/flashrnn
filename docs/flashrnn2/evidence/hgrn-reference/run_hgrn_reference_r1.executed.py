"""Freeze inputs and run the full public HGRN checkpoint in one CPU child."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / "source/tools/flashrnn2"
names = (
    "hgrn_reference_gate.py",
    "hgrn_torch_reference.py",
    "gla_torch_reference.py",
    "gla_reference_gate.py",
)
files = [source / name for name in names] + [
    root / "run_hgrn_gate_cached.py",
    root / "uv_cache_runtime.py",
]
preflight = json.loads((root / "evidence/hgrn-checkpoint-r1.json").read_text())
primitive = json.loads((root / "evidence/hgrn-primitives-r2.json").read_text())
assert preflight["status"] == primitive["status"] == "PASS"
assert preflight["checkpoint_tensors"] == 244
assert primitive["layers_in_scope"] == 2
assert shutil.disk_usage(root).free > 2 * 1024**3
stem = "hgrn-reference-r1"
output = root / "evidence" / f"{stem}.jsonl"
receipt = root / "evidence" / f"{stem}-controller.json"
assert not output.exists() and not receipt.exists()
frozen = root / "evidence" / f"{stem}.sources"
frozen.mkdir()
manifest = {}
for path in files:
    data = path.read_bytes()
    (frozen / path.name).write_bytes(data)
    manifest[str(path)] = {
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
(frozen / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
command = [
    sys.executable,
    str(root / "run_hgrn_gate_cached.py"),
    "--model",
    str(root / "models/hgrn-1.3b"),
    "--source",
    str(root / "evidence/models/hgrn-sources"),
    "--common",
    str(root / "evidence/models/gla-sources"),
    "--output",
    str(output),
]
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "command": command,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    },
    "scope": "FULL_HGRN_1_3B_CPU_B1_B2_B4_P5_G4",
    "initial_free_bytes": shutil.disk_usage(root).free,
}
env = dict(
    os.environ,
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    CUDA_VISIBLE_DEVICES="",
)
with (root / "evidence" / f"{stem}.log").open("x") as log:
    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    record["child_pid"] = child.pid
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    record["returncode"] = child.wait()
record["finished"] = time.time()
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(record["returncode"])
