"""Download one pinned HGRN checkpoint while reference development continues."""

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
metadata = root / "evidence/models/hgrn-1.3b-hf-immutable-api.json"
assert shutil.disk_usage(root).free > 5 * 1024**3
assert json.loads(metadata.read_text())["sha"] == "1adad50103ad6b9c5f79df6b3ce6c9fa2299a572"
receipt = root / "evidence/hgrn-download-r1-controller.json"
assert not receipt.exists()
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    "scope": "HGRN_FULL_PINNED_CHECKPOINT_AND_TOKENIZER_DOWNLOAD_ONLY",
}
with (root / "evidence/hgrn-download-r1.log").open("x") as log:
    child = subprocess.Popen(
        [sys.executable, str(script), str(metadata), str(root / "models/hgrn-1.3b")],
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
