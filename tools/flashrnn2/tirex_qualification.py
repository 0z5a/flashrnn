"""Content-bound TiRex qualification. Historical reports remain readable."""

import hashlib
import json
from pathlib import Path

SOURCE_FILES = (
    "flashrnn/flashrnn2/triton_persistent.py",
    "flashrnn/flashrnn2/gluon_persistent.py",
    "flashrnn/flashrnn2/state_contract.py",
    "flashrnn/flashrnn2/tirex_adapter.py",
    "flashrnn/flashrnn2/tirex_layout.py",
    "flashrnn/flashrnn2/reference.py",
    "tools/flashrnn2/tirex_slstm_reference_gate.py",
    "tools/flashrnn2/tirex_gpu_gate.py",
    "tools/flashrnn2/audit_tirex_gpu_gate.py",
    "tools/flashrnn2/tirex_e2e.py",
    "tools/flashrnn2/tirex_e2e_diagnostic.py",
    "tools/flashrnn2/audit_tirex_e2e_diagnostic.py",
    "tools/flashrnn2/tirex_qualification.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_manifest(root, runtime, source_pins_sha256):
    """Hash files, not Git HEAD: dirty or uncommitted kernels must invalidate gates."""
    return {
        "schema_version": 1,
        "sources": {name: sha256(Path(root) / name) for name in SOURCE_FILES},
        "runtime": runtime,
        "reference_source_pins_sha256": source_pins_sha256,
        "compile_options": {
            "num_warps": 4,
            "enable_fp_fusion": False,
            "weight_operand": "b",
            "register_layout": False,
            "slstm_semantics": "tirex",
        },
    }


def current_manifest(source):
    """Called by GPU runners; CPU manifest tests need no Torch/Triton import."""
    import inspect
    import os

    import torch
    import triton

    from flashrnn.flashrnn2 import (
        state_contract,
        tirex_adapter,
        tirex_layout,
        triton_persistent,
    )

    root = Path(__file__).resolve().parents[2]
    for module in (state_contract, tirex_adapter, tirex_layout, triton_persistent):
        expected = (
            root / "flashrnn/flashrnn2" / (module.__name__.split(".")[-1] + ".py")
        )
        if Path(inspect.getfile(module)).resolve() != expected:
            raise ValueError(
                "imported candidate source is outside the manifest checkout"
            )
    for key in ("TRITON_INTERPRET", "TRITON_KERNEL_OVERRIDE"):
        if os.environ.get(key, "0") not in ("", "0"):
            raise ValueError(
                "qualification requires compiled kernels without external overrides"
            )
    runtime = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "compiler_environment": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in sorted(os.environ.items())
            if key.startswith("TRITON_")
            and key not in ("TRITON_CACHE_DIR", "TRITON_DUMP_DIR")
            or key in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")
        },
    }
    return execution_manifest(root, runtime, sha256(source / "source-pins.json"))


def require_manifest(report, current):
    if report.get("execution_manifest") != current:
        raise ValueError(
            "qualification is missing or has changed source/runtime/compile identity; rerun numerical gates"
        )


def require_diagnostic(
    report, audit, *, current, gate_sha256, fixture_sha256, batch, concurrency
):
    """Validate the exact accelerated cohort before entering the timed path."""
    require_manifest(report, current)
    count = (concurrency // batch) * 128
    if (
        report.get("status") != "PASS"
        or report.get("budget") != 1e-4
        or report.get("batch") != batch
        or report.get("queued_series") != concurrency
        or report.get("fixture_sha256") != fixture_sha256
        or report.get("qualification_gate_sha256") != gate_sha256
        or len(report.get("comparisons", [])) != count
        or not all(row.get("pass") is True for row in report["comparisons"])
        or audit.get("status") != "PASS"
        or audit.get("observed_pairs") != count
        or audit.get("required_pairs") != count
    ):
        raise ValueError(
            "complete matching TiRex diagnostic and independent audit required"
        )


def load_diagnostic(path, audit_path, **kwargs):
    report = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    require_diagnostic(report, audit, **kwargs)
    if (
        audit.get("diagnostic_sha256") != sha256(path)
        or audit.get("raw_sha256") != report.get("raw_sha256")
        or sha256(path.with_suffix(".pt")) != report.get("raw_sha256")
    ):
        raise ValueError("diagnostic audit/raw identity changed")
