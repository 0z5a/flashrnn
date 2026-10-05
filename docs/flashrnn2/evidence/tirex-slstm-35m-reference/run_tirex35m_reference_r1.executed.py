"""Freeze and execute the complete TiRex sLSTM CPU forecast gate."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
files = [
    root / "source/tools/flashrnn2/tirex_slstm_reference_gate.py",
    root / "run_tirex_gate_cached.py",
    root / "uv_cache_runtime.py",
]
download = json.loads((root / "evidence/tirex-slstm-35m-download-r1.json").read_text())
assert download["verified"] and download["curl_exit"] == 0
stem = "tirex-slstm-35m-reference-r1"
output = root / "evidence" / f"{stem}.jsonl"
receipt = root / "evidence" / f"{stem}-controller.json"
assert not output.exists() and not receipt.exists()
frozen = root / "evidence" / f"{stem}.sources"
frozen.mkdir()
for path in files:
    (frozen / path.name).write_bytes(path.read_bytes())
source_sha256 = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
}
(frozen / "manifest.json").write_text(json.dumps(source_sha256, indent=2) + "\n")
command = [
    sys.executable,
    str(root / "run_tirex_gate_cached.py"),
    "--model",
    str(root / "models/tirex-slstm-35m"),
    "--source",
    str(root / "evidence/models/tirex-source"),
    "--output",
    str(output),
]
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "command": command,
    "source_sha256": source_sha256,
    "scope": "FULL_TIREX_SLSTM_35M_CPU_B1_B2_B4_CONTEXT128_H64",
}
env = dict(
    os.environ,
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    HF_HUB_OFFLINE="1",
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
