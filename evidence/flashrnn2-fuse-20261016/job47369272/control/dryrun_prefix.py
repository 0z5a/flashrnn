"""Login-node check that runtime.sh applies FRNN_PREFIX (the sanitizer wrapper) to the Python process.

Negative control: the v1 runtime.sh, which ignored the prefix, must not print the marker.
"""
import json
import os
import subprocess
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
V1_RUNTIME = Path("/gpfs/scratch/ehpc390/tueb043681/flashrnn2-rtiled-m6-20261014-v1/control/runtime.sh")
MARKER = "PREFIX_APPLIED"


def run(runtime):
    env = dict(os.environ, FRNN_PREFIX=f"echo {MARKER}", FRNN_CACHE_TAG="dryprefix", FRNN_TILELANG="0")
    p = subprocess.run(["bash", str(runtime), "-c", "print('NO_PREFIX')"], cwd=CONTROL.parent / "source", env=env,
                       capture_output=True, text=True, timeout=600)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    return dict(runtime=str(runtime), exit=p.returncode, last_line=lines[-1] if lines else None)


def main():
    rec = dict(v2=run(CONTROL / "runtime.sh"), v1_negative_control=run(V1_RUNTIME))
    rec["status"] = "PASS" if (rec["v2"]["last_line"] == f"{MARKER} python3 -B -c print('NO_PREFIX')"
                               and rec["v1_negative_control"]["last_line"] == "NO_PREFIX") else "FAIL"
    (CONTROL / "dryrun-prefix.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
