"""Run the full public Gated DeltaNet checkpoint in one CPU child."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
assert len(sys.argv) == 2 and sys.argv[1] in ("r2", "r3", "r4")
label = sys.argv[1]
source = root / "source/tools/flashrnn2"
names = (
    "gdn_reference_gate.py",
    "gdn_torch_reference.py",
    "deltanet_torch_reference.py",
    "gla_torch_reference.py",
    "gla_reference_gate.py",
)
files = [source / name for name in names] + [
    root / "run_gdn_gate_cached.py",
    root / "uv_cache_runtime.py",
]
checkpoint = json.loads((root / "evidence/gdn-checkpoint-r1.json").read_text())
primitive = json.loads((root / "evidence/gdn-primitives-r9.json").read_text())
preflight = json.loads(
    (root / "evidence/gdn-historical-architecture-preflight-r1.json").read_text()
)
assert checkpoint["status"] == primitive["status"] == "PASS"
assert preflight["status"] == "PASS"
assert checkpoint["layers"] == 24 and primitive["layers_in_scope"] == 1
assert (
    primitive["source_sha256"]["gdn_torch_reference.py"]
    == hashlib.sha256((source / "gdn_torch_reference.py").read_bytes()).hexdigest()
)
assert shutil.disk_usage(root).free > 3 * 1024**3
stem = f"gdn-reference-{label}"
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
    str(root / "run_gdn_gate_cached.py"),
    "--model",
    str(root / "models/gdn-340m"),
    "--source",
    str(root / "evidence/models/gdn-sources-2024-12-22"),
    "--delta-source",
    str(root / "evidence/models/deltanet-sources"),
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
    "scope": "FULL_GATED_DELTANET_340M_CPU_B1_B2_B4_P5_G4",
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
