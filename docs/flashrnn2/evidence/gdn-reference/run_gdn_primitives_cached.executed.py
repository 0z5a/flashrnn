"""Use verified local cached dependencies for one finite GDN CPU control."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

from uv_cache_runtime import activate

root = Path(__file__).resolve().parent
assert len(sys.argv) == 2
label = sys.argv[1]
assert label in ("r6", "r7", "r8", "r9")
paths = activate()
probe = root / "check_gdn_primitives_r1.py"
runtime = {
    "python": sys.executable,
    "archives": paths,
    "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    "reference_sha256": hashlib.sha256(
        (root / "source/tools/flashrnn2/gdn_torch_reference.py").read_bytes()
    ).hexdigest(),
    "scope": "READ_ONLY_EXISTING_LOCAL_CACHE_NO_INSTALL_OR_UPGRADE",
}
(root / f"evidence/gdn-primitives-{label}-runtime.json").write_text(
    json.dumps(runtime, indent=2) + "\n"
)
sys.argv = [str(probe), str(root / f"evidence/gdn-primitives-{label}.json")]
runpy.run_path(str(probe), run_name="__main__")
