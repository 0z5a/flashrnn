"""Independently audit saved TiRex forecast outputs and all sLSTM states."""

import hashlib
import json
from pathlib import Path

from uv_cache_runtime import activate

activate()
import torch

root = Path(__file__).resolve().parent
stem = "tirex-slstm-35m-reference-r1"
evidence = root / "evidence"
model = root / "models/tirex-slstm-35m/model.ckpt"
source = evidence / "models/tirex-source"
meta = json.loads((evidence / f"{stem}.meta.json").read_text())
controller = json.loads((evidence / f"{stem}-controller.json").read_text())
rows = [
    json.loads(line) for line in (evidence / f"{stem}.jsonl").read_text().splitlines()
]
assert controller["returncode"] == 0 and meta["status"] == "PASS"
assert len(rows) == len(meta["snapshots"]) == 9
assert model.stat().st_size == meta["weight_bytes"]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


assert sha256(model) == meta["weight_sha256"]
frozen = evidence / f"{stem}.sources"
source_manifest = json.loads((frozen / "manifest.json").read_text())
for path, checksum in source_manifest.items():
    assert sha256(frozen / Path(path).name) == checksum
    assert sha256(Path(path)) == checksum
assert (
    meta["gate_sha256"]
    == source_manifest[
        str(root / "source/tools/flashrnn2/tirex_slstm_reference_gate.py")
    ]
)
pins = json.loads((source / "source-pins.json").read_text())
assert pins == meta["source_pins"]
for entry in pins["files"]:
    path = source / entry["path"]
    data = path.read_bytes()
    assert len(data) == entry["bytes"]
    assert (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["sha"]
    )


def compare(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    assert actual.shape == reference.shape
    delta = torch.abs(actual - reference)
    allowed = meta["budget"] * (1 + torch.abs(reference))
    bad = torch.logical_or(
        torch.logical_or(~torch.isfinite(actual), ~torch.isfinite(reference)),
        delta > allowed,
    )
    return {
        "pass": not torch.any(bad).item(),
        "bitwise": torch.equal(actual, reference),
        "max_abs": torch.max(delta).item(),
        "worst_normalized": torch.max(delta / allowed).item(),
        "failed_elements": torch.count_nonzero(bad).item(),
    }


summary = []
seen = set()
for row, entry in zip(rows, meta["snapshots"], strict=True):
    batch, case = row["batch"], row["case"]
    assert (batch, case) == (entry["batch"], entry["case"])
    assert batch in (1, 2, 4) and case in (0, 1, 2)
    assert (batch, case) not in seen
    seen.add((batch, case))
    path = evidence / entry["file"]
    assert path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    context = snapshot["context"]
    assert context.shape == (batch, 128) and torch.isfinite(context).all()
    checks = {
        "quantiles_batch_vs_serial": compare(
            snapshot["quantiles"], snapshot["serial_quantiles"]
        ),
        "median_batch_vs_serial": compare(
            snapshot["median"], snapshot["serial_median"]
        ),
        "quantiles_two_patch_composition": compare(
            snapshot["quantiles"], snapshot["composed_quantiles"]
        ),
        "median_two_patch_composition": compare(
            snapshot["median"], snapshot["composed_median"]
        ),
        "median_matches_q50": compare(
            snapshot["median"], snapshot["quantiles"][:, :, 4]
        ),
    }
    assert checks == row["checks"]
    assert snapshot["quantiles"].shape == (batch, 64, 9)
    assert snapshot["median"].shape == (batch, 64)
    states = []
    batched = snapshot["batched_states"]
    serial = snapshot["serial_states"]
    assert len(batched) == len(serial) == 12
    for layer in range(12):
        assert len(batched[layer]) == 2 and len(serial[layer]) == 2 * batch
        for phase in range(2):
            merged = torch.cat(
                [serial[layer][2 * index + phase] for index in range(batch)], dim=1
            )
            assert merged.shape == batched[layer][phase].shape == (4, batch, 512)
            for index, name in enumerate(
                ("hidden", "cell", "normalizer", "stabilizer")
            ):
                states.append(
                    {
                        "layer": layer,
                        "phase": phase,
                        "state": name,
                        **compare(batched[layer][phase][index], merged[index]),
                    }
                )
    assert states == row["state_checks"] and len(states) == row["state_pairs"]
    assert row["pass"] == all(x["pass"] for x in checks.values()) and all(
        x["pass"] for x in states
    )
    assert row["bitwise"] == all(x["bitwise"] for x in checks.values()) and all(
        x["bitwise"] for x in states
    )
    summary.append(
        {
            "batch": batch,
            "case": case,
            "pass": row["pass"],
            "bitwise": row["bitwise"],
            "state_pairs": len(states),
        }
    )

assert seen == {(batch, case) for batch in (1, 2, 4) for case in range(3)}
assert all(row["pass"] and row["bitwise"] for row in summary)
audit = {
    "audit": "PASS",
    "model_result": meta["status"],
    "natural_exit": controller["returncode"],
    "forecast_cases_recomputed": len(rows),
    "forecast_series_recomputed": sum(row["batch"] for row in rows),
    "quantile_values_per_path_recomputed": sum(row["batch"] * 64 * 9 for row in rows),
    "state_pairs_recomputed": sum(row["state_pairs"] for row in summary),
    "snapshots_verified": len(meta["snapshots"]),
    "all_bitwise": True,
    "rows": summary,
    "gpu_qualification": False,
    "performance_claim": False,
}
out = evidence / f"{stem}-audit.json"
if out.exists():
    assert json.loads(out.read_text()) == audit
else:
    out.write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
