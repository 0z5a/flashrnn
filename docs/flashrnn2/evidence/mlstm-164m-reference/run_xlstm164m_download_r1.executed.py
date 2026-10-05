"""Download and hash-check the pinned public 164M mLSTM checkpoint."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
metadata_path = root / "evidence/models/xlstm-scaling-laws-hf-immutable-api.json"
metadata = json.loads(metadata_path.read_text())
assert metadata["id"] == "NX-AI/xlstm_scaling_laws"
assert metadata["sha"] == "2ce74c14add515517ae6a32d3ae80cc766f62c03"
assert len(metadata["siblings"]) == 4
assert shutil.disk_usage(root).free > 2 * 1024**3
target = root / "models/xlstm-mlstm-164m"
target.mkdir(exist_ok=True)
receipt = root / "evidence/xlstm-mlstm-164m-download-r1-controller.json"
assert not receipt.exists()
record = {
    "controller_pid": os.getpid(),
    "started": time.time(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    "scope": "PINNED_OFFICIAL_XLSTM_MLSTM_164M_COMPLETE_CHECKPOINT_DOWNLOAD",
    "files": [],
}
receipt.write_text(json.dumps(record, indent=2) + "\n")
for entry in metadata["siblings"]:
    name = entry["rfilename"]
    assert name.startswith(metadata["selected_directory"] + "/")
    filename = Path(name).name
    assert filename in (
        "config.json",
        "metadata.json",
        "model.safetensors.index.json",
        "model_0.safetensors",
    )
    path = target / filename
    url = (
        "https://hf-mirror.com/"
        + metadata["id"]
        + "/resolve/"
        + metadata["sha"]
        + "/"
        + name
        + "?download=true"
    )
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        subprocess.run(
            [
                "curl",
                "-fLsS",
                "--connect-timeout",
                "15",
                "--max-time",
                "1800",
                "-C",
                "-",
                url,
                "-o",
                str(partial),
            ],
            check=True,
        )
        partial.rename(path)
    assert path.stat().st_size == entry["size"]
    if "lfs" in entry:
        digest = hashlib.sha256()
        expected = entry["lfs"]["sha256"]
    else:
        digest = hashlib.sha1(("blob " + str(entry["size"]) + "\0").encode())
        expected = entry["blobId"]
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == expected
    item = {"file": filename, "size": entry["size"], "checksum": expected}
    record["files"].append(item)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(item), flush=True)
manifest = {
    "model": metadata["id"],
    "revision": metadata["sha"],
    "directory": metadata["selected_directory"],
    "metadata_origin": "Hugging Face immutable repository API; public blobs via hf-mirror.com",
    "files": record["files"],
}
(target / "verified-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
record["returncode"] = 0
record["finished"] = time.time()
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
