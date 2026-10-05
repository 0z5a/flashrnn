"""Download and verify the pinned public TiRex sLSTM checkpoint."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/tirex-slstm-35m"
EVIDENCE = ROOT / "evidence/tirex-slstm-35m-download-r1.json"
REVISION = "63c740922493f5fbe60b277609ec62babfba2762"
SIZE = 141230262
SHA256 = "b8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    MODEL.mkdir(parents=True, exist_ok=True)
    partial = MODEL / "model.ckpt.partial"
    final = MODEL / "model.ckpt"
    assert not partial.exists() and not final.exists()
    url = f"https://hf-mirror.com/NX-AI/TiRex/resolve/{REVISION}/model.ckpt"
    started = time.time()
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            "600",
            "--output",
            str(partial),
            url,
        ],
        check=False,
    )
    receipt = {
        "model": "NX-AI/TiRex",
        "revision": REVISION,
        "source": url,
        "started": started,
        "finished": time.time(),
        "curl_exit": result.returncode,
        "expected_bytes": SIZE,
        "expected_sha256": SHA256,
        "observed_bytes": partial.stat().st_size if partial.exists() else 0,
        "observed_sha256": sha256(partial) if partial.exists() else None,
    }
    receipt["verified"] = (
        result.returncode == 0
        and receipt["observed_bytes"] == SIZE
        and receipt["observed_sha256"] == SHA256
    )
    if receipt["verified"]:
        partial.replace(final)
    EVIDENCE.write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["verified"]


if __name__ == "__main__":
    main()
