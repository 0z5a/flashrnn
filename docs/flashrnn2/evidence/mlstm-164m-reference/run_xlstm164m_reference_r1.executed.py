"""Freeze inputs and run the full public xLSTM mLSTM 164M CPU checkpoint."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
files = [
    root / "source/tools/flashrnn2/mlstm_scaling_reference_gate.py",
    root / "run_xlstm_gate_cached.py",
    root / "uv_cache_runtime.py",
]
download = json.loads(
    (root / "evidence/xlstm-mlstm-164m-download-r1-controller.json").read_text()
)
manifest = json.loads(
    (root / "models/xlstm-mlstm-164m/verified-manifest.json").read_text()
)
assert download["returncode"] == 0
assert manifest["revision"] == "2ce74c14add515517ae6a32d3ae80cc766f62c03"
assert shutil.disk_usage(root).free > 2 * 1024**3
stem = "xlstm-mlstm-164m-reference-r1"
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
    str(root / "run_xlstm_gate_cached.py"),
    "--model",
    str(root / "models/xlstm-mlstm-164m"),
    "--output",
    str(output),
    "--state-budget",
    "0.0001",
]
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "command": command,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    },
    "scope": "FULL_XLSTM_MLSTM_164M_CPU_B1_B2_B4_P5_G4",
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
