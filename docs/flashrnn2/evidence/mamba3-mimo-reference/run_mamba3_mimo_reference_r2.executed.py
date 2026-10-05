"""Freeze inputs and run the full public Mamba-3 MIMO CPU checkpoint."""

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
files = [
    source / name
    for name in (
        "mamba3_reference_gate.py",
        "mamba3_torch_reference.py",
        "mamba3_mimo_torch_reference.py",
        "gla_reference_gate.py",
    )
] + [root / "run_mamba3_gate_cached.py", root / "uv_cache_runtime.py"]
download = json.loads(
    (root / "evidence/mamba3-mimo-download-r1-controller.json").read_text()
)
manifest = json.loads(
    (root / "models/mamba3-mimo-187m/verified-manifest.json").read_text()
)
assert download["returncode"] == 0
assert manifest["revision"] == "8fd6e9eb7b795f2e15d7f6353251d0137980c43e"
assert shutil.disk_usage(root).free > 2 * 1024**3
stem = "mamba3-mimo-reference-r2"
output = root / "evidence" / f"{stem}.jsonl"
receipt = root / "evidence" / f"{stem}-controller.json"
assert not output.exists() and not receipt.exists()
frozen = root / "evidence" / f"{stem}.sources"
frozen.mkdir()
source_manifest = {}
for path in files:
    data = path.read_bytes()
    (frozen / path.name).write_bytes(data)
    source_manifest[str(path)] = {
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
(frozen / "manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
command = [
    sys.executable,
    str(root / "run_mamba3_gate_cached.py"),
    "--model",
    str(root / "models/mamba3-mimo-187m"),
    "--source",
    str(root / "evidence/models/mamba3-sources"),
    "--output",
    str(output),
    "--variant",
    "mimo",
    "--ssm-budget",
    "0.0001",
    "--key-budget",
    "0.00005",
]
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "command": command,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    },
    "scope": "FULL_MAMBA3_MIMO_187M_CPU_B1_B2_B4_P5_G4_CALIBRATED_STATE",
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
