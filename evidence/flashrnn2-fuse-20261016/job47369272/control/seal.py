"""Login-side preparation for the fused backward batch: verify inputs, seal source/control files, write start-command.json."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/gpfs/home/tueb/tueb043681/flashrnn2-fuse-20261016-v2")
CONTROL = BASE / "control"
WRAPPER = Path("/gpfs/scratch/ehpc390/tueb043681/short-test-start-command-20261013-v1/offline_batch.py")
PY = "/apps/ACC/PYTHON/3.12.1/GCC/bin/python3"
# Names already used by this account (fresh squeue/sacct check before sealing; submit.py re-checks).
USED_NAMES = set(Path(CONTROL / "used-names.txt").read_text().split()) if (CONTROL / "used-names.txt").exists() else None
V1_BASE = "/gpfs/scratch/ehpc390/tueb043681/paper-baselines-20261013-v1"


def quota_headroom(text, filesystem):
    """Hard limit minus usage (bytes) of one bsc_quota row, or None."""
    units = dict(KB=1e3, MB=1e6, GB=1e9, TB=1e12)
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        cols = line.split()
        if cols and cols[0] == filesystem:
            values = [float(cols[i]) * units[cols[i + 1]] for i in range(2, 8, 2)]  # usage, quota, limit
            return values[2] - values[0]
    return None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks = {}
    plan = json.loads((CONTROL / "plan.json").read_text())
    checks["plan_counts"] = tuple(len(plan[k]) for k in ("prebuild", "qualify", "race", "sanitizer", "profile",
                                                          "select_cases", "comparisons")) == (20, 186, 32, 16, 8, 30, 60)
    names = [x["name"] for k in ("prebuild", "qualify", "race", "sanitizer", "profile", "select_cases", "comparisons")
             for x in plan[k]]
    checks["task_names_unique"] = len(names) == len(set(names))
    fuse = json.loads((CONTROL / "dryrun-fuse.json").read_text())
    checks["dryrun_fuse_kernels_compile"] = fuse.get("status") == "PASS"
    dry = json.loads((CONTROL / "dryrun-launcher.json").read_text())
    checks["dryrun_launcher_built"] = dry.get("cuda_utils") == "BUILT" and dry.get("rtiled_launcher") == "BUILT" \
        and dry.get("cc") == "/usr/bin/gcc"
    pre = json.loads((CONTROL / "dryrun-prefix.json").read_text())
    checks["dryrun_prefix_applied_v1_control_reproduced"] = pre.get("status") == "PASS"
    name = re.search(r"^#SBATCH --job-name=(\S+)$", (CONTROL / "start.sbatch").read_text(), re.M).group(1)
    checks["job_name_numeric_unused"] = USED_NAMES is not None and name.isdigit() and name not in USED_NAMES
    checks["no_prior_start"] = not (CONTROL / "offline-start.once").exists()
    checks["science_absent"] = not (BASE / "science").exists()
    free = shutil.disk_usage(BASE).free
    # v1 (job 47365377) failed with EDQUOT: the group scratch quota was at its hard limit while df still showed free
    # bytes. The batch now lives in home and the gate reads the quota itself: hard limit minus usage of gpfs_home must
    # exceed 500 MB (the batch writes about 10-50 MB), and a 50 MB probe write must succeed.
    quota = subprocess.run(["bsc_quota"], capture_output=True, text=True).stdout
    headroom = quota_headroom(quota, "gpfs_home")
    checks["home_quota_headroom_gt_500MB"] = headroom is not None and headroom > 5e8
    probe = CONTROL / "quota-probe.bin"
    try:
        with open(probe, "wb") as f:
            f.write(b"\0" * (50 << 20))
            f.flush()
            os.fsync(f.fileno())
        checks["probe_write_50MB"] = probe.stat().st_size == 50 << 20
    except OSError:
        checks["probe_write_50MB"] = False
    finally:
        probe.unlink(missing_ok=True)
    for p in ("/gpfs/home/tueb/tueb043681/flashns-math376-job47032336/activate_flashns.sh",
              "/gpfs/scratch/ehpc390/tueb043681/flashrnn2-distributed-20261011-v2/deps36/triton/__init__.py",
              "/apps/ACC/CUDA/12.8/bin/nvcc", PY, str(WRAPPER)):
        checks["exists:" + p] = Path(p).exists()
    sealed_paths = sorted(str(p) for p in (BASE / "source").rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts)
    sealed_paths += [str(CONTROL / n) for n in ("run_fuse.py", "make_plan.py", "finalize.py", "runtime.sh", "plan.json",
                                                "start.sbatch", "seal.py", "dryrun_launcher.py", "dryrun-launcher.json",
                                                "dryrun_fuse.py", "dryrun-fuse.json", "dryrun_prefix.py",
                                                "dryrun-prefix.json")]
    sealed_paths += [str(WRAPPER)]
    sealed = [dict(path=p, sha256=sha(p)) for p in sealed_paths]
    source_manifest = {r["path"]: r["sha256"] for r in sealed if "/source/" in r["path"]}
    (CONTROL / "source.manifest.json").write_text(json.dumps(source_manifest, indent=2))
    status = "READY" if all(checks.values()) else "NOT_READY"
    prepare = dict(status=status, checks=checks, prepared=time.time(), gpfs_free_bytes=free, bsc_quota=quota,
                   home_quota_headroom_bytes=headroom,
                   source_manifest_sha256=sha(CONTROL / "source.manifest.json"), sealed_count=len(sealed),
                   job_name=name)
    (CONTROL / "prepare.json").write_text(json.dumps(prepare, indent=2))
    manifest = dict(
        version=1, control_dir=str(CONTROL),
        start_argv=[PY, "-I", "-B", "-S", str(CONTROL / "run_fuse.py")],
        finalize_argv=[PY, "-I", "-B", "-S", str(CONTROL / "finalize.py")],
        sealed_files=sealed + [dict(path=str(CONTROL / "source.manifest.json"),
                                    sha256=sha(CONTROL / "source.manifest.json"))],
        preparation_receipt=dict(path=str(CONTROL / "prepare.json"), sha256=sha(CONTROL / "prepare.json"),
                                 status_key="status", status_value="READY"),
    )
    (CONTROL / "start-command.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(status=status, checks=checks, sealed=len(sealed),
                          start_command_sha256=sha(CONTROL / "start-command.json")), indent=2))
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
