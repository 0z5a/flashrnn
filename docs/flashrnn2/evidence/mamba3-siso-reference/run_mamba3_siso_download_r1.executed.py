"""Download the pinned public Mamba-3 SISO checkpoint asynchronously."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


root = Path(__file__).resolve().parent
script = root / "download_hgrn2_r1.py"
metadata = root / "evidence/models/mamba3_siso187m-hf-immutable-api.json"
assert shutil.disk_usage(root).free > 1024**3
assert json.loads(metadata.read_text())["sha"] == "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"
receipt = root / "evidence/mamba3-siso-download-r1-controller.json"
assert not receipt.exists()
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    "scope": "MAMBA3_SISO_187M_FULL_PINNED_PUBLIC_DOWNLOAD_ONLY",
}
with (root / "evidence/mamba3-siso-download-r1.log").open("x") as log:
    child = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(metadata),
            str(root / "models/mamba3-siso-187m"),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    record["child_pid"] = child.pid
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    record["returncode"] = child.wait()
record["finished"] = time.time()
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(record["returncode"])
