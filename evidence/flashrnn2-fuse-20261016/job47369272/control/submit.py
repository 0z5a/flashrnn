"""Login-side submission: fresh name-collision check, sbatch, and submission receipt."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/gpfs/home/tueb/tueb043681/flashrnn2-fuse-20261016-v2")
CONTROL = BASE / "control"
SCRIPT = CONTROL / "start.sbatch"
NAME = re.search(r"^#SBATCH --job-name=(\d+)$", SCRIPT.read_text(), re.M).group(1)  # sealed with the script


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    active = subprocess.run(["squeue", "-u", "tueb043681", "-h", "-o", "%j"], capture_output=True,
                            text=True, check=True).stdout.split()
    history = subprocess.run(["sacct", "-u", "tueb043681", "-S", "2026-10-01", "-n", "-X", "-o", "JobName%20"],
                             capture_output=True, text=True, check=True).stdout.split()
    if NAME in active or NAME in history:
        print(json.dumps(dict(status="NAME_COLLISION", name=NAME)))
        return 2
    prepare = json.loads((CONTROL / "prepare.json").read_text())
    manifest = json.loads((CONTROL / "start-command.json").read_text())
    assert prepare["status"] == "READY", prepare
    assert manifest["preparation_receipt"]["sha256"] == sha(CONTROL / "prepare.json")
    assert not (CONTROL / "submission.json").exists(), "already submitted"
    script = SCRIPT
    assert prepare["job_name"] == NAME, (prepare["job_name"], NAME)
    out = subprocess.run(["sbatch", "--parsable", f"--job-name={NAME}", str(script)], capture_output=True, text=True,
                         check=True)
    job = out.stdout.strip().split(";")[0]
    receipt = dict(
        status="SUBMITTED_PREPARED_FINITE_BATCH", job_id=job, numeric_name=NAME, nodes=1, time="03:00:00",
        account="ehpc390", partition="acc", qos="acc_ehpc", allocated_gpu_hour_ceiling=12,
        node_reason=json.loads((CONTROL / "plan.json").read_text())["node_reason"],
        real_start_argv=["env", "-u", "PYTHONHOME", "-u", "PYTHONPATH", "-u", "LD_PRELOAD",
                         "LD_LIBRARY_PATH=/apps/ACC/PYTHON/3.12.1/GCC/lib:/usr/lib64",
                         "/apps/ACC/PYTHON/3.12.1/GCC/bin/python3", "-I", "-B", "-S",
                         "/gpfs/scratch/ehpc390/tueb043681/short-test-start-command-20261013-v1/offline_batch.py",
                         "--manifest", str(CONTROL / "start-command.json")],
        script=str(script), script_sha256=sha(script),
        start_command_sha256=sha(CONTROL / "start-command.json"),
        prepare_sha256=sha(CONTROL / "prepare.json"),
        source_manifest_sha256=prepare["source_manifest_sha256"],
        frozen_configs=prepare.get("frozen_configs"), case_plan=str(CONTROL / "plan.json"),
        output=str(BASE / "science"), archive=str(BASE / "archives"),
        desktop_connection_required=False, scientific_pass=False, submitted=time.time(),
        sbatch_stdout=out.stdout, sbatch_stderr=out.stderr,
    )
    (CONTROL / "submission.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(status=receipt["status"], job_id=job, name=NAME)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
