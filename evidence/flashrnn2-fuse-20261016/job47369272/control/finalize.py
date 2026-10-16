"""Cluster-side finalizer: archive complete raw outputs and verify the copy independently."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
BASE = CONTROL.parent
ARCHIVES = BASE / "archives"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    record = dict(job=job, start_exit=os.environ.get("MN5_START_EXIT_CODE"), started=time.time())
    # Owned children must already be joined by the driver; record any survivor.
    survivors = subprocess.run(["pgrep", "-af", str(BASE / "science")],
                               capture_output=True, text=True).stdout
    record["surviving_owned_processes"] = survivors.strip().splitlines()
    # Remove only this job's own node-local compile caches.
    removed = []
    for d in Path("/scratch/tmp").glob(f"tueb043681-fuse2-{job}-*"):
        subprocess.run(["rm", "-rf", "--one-file-system", str(d)], check=False)
        removed.append(dict(path=str(d), gone=not d.exists()))
    record["removed_local_caches"] = removed
    ARCHIVES.mkdir(exist_ok=True)
    members = [p for p in ("science", "control") if (BASE / p).exists()]
    files, blobs = {}, {}
    for m in members:
        for f in sorted((BASE / m).rglob("*")):
            if f.is_file() and not f.name.endswith((".partial", ".tmp")) and not f.name.startswith("offline-finalize"):
                name = str(f.relative_to(BASE))
                data = f.read_bytes()  # read once: the archived bytes are the hashed bytes
                blobs[name] = data
                files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest = ARCHIVES / f"fuse2-{job}-manifest.json"
    manifest.write_text(json.dumps(files, indent=2))
    archive = ARCHIVES / f"fuse2-{job}.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for name, data in blobs.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o600
            tar.addfile(info, io.BytesIO(data))
        tar.add(manifest, arcname=manifest.name)
    # Independent reader: reopen the archive and re-hash every member against the manifest.
    mismatches = []
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        for name, meta in files.items():
            if name not in names:
                mismatches.append(dict(name=name, error="missing"))
                continue
            data = tar.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != meta["sha256"] or len(data) != meta["bytes"]:
                mismatches.append(dict(name=name, error="hash"))
    record.update(archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                  manifest=str(manifest), files=len(files), mismatches=mismatches,
                  status="ARCHIVE_VERIFIED" if not mismatches else "ARCHIVE_MISMATCH",
                  finished=time.time(), scientific_pass=False)
    (CONTROL / "finalize-result.json").write_text(json.dumps(record, indent=2))
    return 0 if not mismatches and not record["surviving_owned_processes"] else 1


if __name__ == "__main__":
    sys.exit(main())
