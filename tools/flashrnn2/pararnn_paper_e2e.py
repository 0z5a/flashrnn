"""Paired full-model ParaRNN paper control after native and shape gates."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

SOURCE_REVISION = "513d75dade40843bb27ac4a521969997188b5459"
BUDGET = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PaperModel:
    def __init__(self, cell_class, config_class, mode_class, width: int):
        torch.manual_seed(20261009)
        self.mode_class = mode_class
        self.embedding = torch.randn(512, width, device="cuda") * 0.1
        self.readout = torch.randn(width, 512, device="cuda") * 0.1
        self.layers = [
            cell_class(
                config_class(
                    state_dim=width,
                    input_dim=width,
                    num_heads=2,
                    device="cuda",
                    dtype=torch.float32,
                    mode="sequential",
                )
            ).eval()
            for _ in range(4)
        ]

    def set_mode(self, mode: str) -> None:
        for layer in self.layers:
            layer.mode = self.mode_class(mode)

    def forward(self, ids: torch.Tensor, trace: bool = False):
        x = self.embedding[ids.to("cuda")]
        history = []
        for layer in self.layers:
            x = layer(x)
            if trace:
                history.append(x.detach().cpu())
        return x[:, -1] @ self.readout, history


def qualify(model: PaperModel, fixtures: list[torch.Tensor], mode: str):
    records = []
    parameters = [
        parameter for layer in model.layers for parameter in layer.parameters()
    ]
    weights = [parameter.detach().clone() for parameter in parameters]
    for ids in fixtures:
        model.set_mode("sequential")
        baseline, baseline_layers = model.forward(ids, True)
        model.set_mode(mode)
        candidate, candidate_layers = model.forward(ids, True)
        records.append(
            {
                "ids": ids.clone(),
                "baseline": {"logits": baseline.cpu(), "layers": baseline_layers},
                "candidate": {"logits": candidate.cpu(), "layers": candidate_layers},
            }
        )
    if not all(
        torch.equal(parameter, weight) for parameter, weight in zip(parameters, weights)
    ):
        raise ValueError("ParaRNN weights changed between paired modes")
    return records


def burst(model: PaperModel, fixtures: list[torch.Tensor], mode: str):
    model.set_mode(mode)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    outputs = []
    finished_ms = []
    for ids in fixtures:
        logits, _ = model.forward(ids)
        outputs.append(logits.cpu())
        finished_ms.append((time.perf_counter_ns() - started) / 1e6)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return outputs, {
        "elapsed_ms": elapsed_ms,
        "responses_per_second": sum(ids.shape[0] for ids in fixtures)
        * 1000
        / elapsed_ms,
        "last_response_ms": finished_ms[-1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-extension-sha", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--native-gate-raw", type=Path, required=True)
    parser.add_argument("--native-gate-audit", type=Path, required=True)
    parser.add_argument("--qualification-auditor", type=Path, required=True)
    parser.add_argument("--cell", choices=("gru", "lstm"), required=True)
    parser.add_argument(
        "--mode", choices=("parallel_CUDA", "parallel_FUSED"), required=True
    )
    parser.add_argument("--batch", type=int, choices=(1, 4, 16, 32, 64), required=True)
    parser.add_argument(
        "--concurrency", type=int, choices=(1, 8, 32, 64, 128), required=True
    )
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.concurrency < args.batch
        or args.concurrency % args.batch
        or args.blocks < 20
    ):
        parser.error("requires C >= B, C divisible by B and at least 20 paired blocks")
    qualification = args.output.with_suffix(".qualification.pt")
    audit_path = args.output.with_suffix(".qualification.audit.json")
    meta_path = args.output.with_suffix(".meta.json")
    if any(
        path.exists() for path in (args.output, qualification, audit_path, meta_path)
    ):
        parser.error("output evidence already exists")

    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != SOURCE_REVISION or manifest["file_count"] != 31:
        raise ValueError("ParaRNN source revision differs")
    for entry in manifest["files"]:
        if sha256(args.source_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"source file differs: {entry['path']}")
    extensions = list(args.source_root.glob("*parallel_reduce_cuda*.so"))
    if len(extensions) != 1 or sha256(extensions[0]) != args.expected_extension_sha:
        raise ValueError("native extension differs from the qualified build")
    native = json.loads(args.native_gate_audit.read_text())
    if (
        native["status"] != "PASS"
        or native["tensor_pairs"] != 40
        or native["failed_elements"] != 0
        or native["raw_sha256"] != sha256(args.native_gate_raw)
        or native["source_manifest_sha256"] != sha256(args.manifest)
        or native["extension_sha256"] != args.expected_extension_sha
    ):
        raise ValueError("native forward/gradient qualification is incomplete")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible or torch.cuda.device_count() != 1:
        raise ValueError("exactly one assigned CUDA device must be visible")
    actual_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if actual_uuid != args.expected_gpu_uuid:
        raise ValueError("CUDA device differs from the admitted device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.source_root))
    from pararnn.rnn_cell.gru_diag_mh import GRUDiagMH, GRUDiagMHConfig
    from pararnn.rnn_cell.lstm_cifg_diag_mh import LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig
    from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode

    classes = {
        "gru": (GRUDiagMH, GRUDiagMHConfig),
        "lstm": (LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig),
    }
    model = PaperModel(*classes[args.cell], RNNCellApplicationMode, 64)
    groups = args.concurrency // args.batch
    generator = torch.Generator().manual_seed(20261009 + args.batch)
    fixtures = [
        torch.randint(0, 512, (args.batch, 128), generator=generator)
        for _ in range(groups)
    ]
    with torch.inference_mode():
        records = qualify(model, fixtures, args.mode)
        torch.save(
            {
                "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": sha256(args.manifest),
                "extension_sha256": args.expected_extension_sha,
                "native_gate_audit_sha256": sha256(args.native_gate_audit),
                "cell": args.cell,
                "mode": args.mode,
                "batch": args.batch,
                "concurrency": args.concurrency,
                "steps": 128,
                "layers": 4,
                "width": 64,
                "budget": BUDGET,
                "records": records,
            },
            qualification,
        )
        audit = subprocess.run(
            [
                sys.executable,
                str(args.qualification_auditor),
                str(qualification),
                "--output",
                str(audit_path),
            ],
            check=False,
        )
        if audit.returncode != 0:
            raise ValueError("actual-shape independent qualification failed")
        report = json.loads(audit_path.read_text())
        if (
            report["status"] != "PASS"
            or report["failed_elements"] != 0
            or report["tensor_pairs"] != groups * 5
            or report["raw_sha256"] != sha256(qualification)
            or report["source_manifest_sha256"] != sha256(args.manifest)
            or report["extension_sha256"] != args.expected_extension_sha
            or report["native_gate_audit_sha256"] != sha256(args.native_gate_audit)
        ):
            raise ValueError("qualification report does not bind the raw arrays")
        for _ in range(2):
            burst(model, fixtures, "sequential")
            burst(model, fixtures, args.mode)
        with args.output.open("w") as stream:
            for block in range(args.blocks):
                order = (
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                )
                timings = {}
                for arm in order:
                    mode = "sequential" if arm == "baseline" else args.mode
                    outputs, timings[arm] = burst(model, fixtures, mode)
                    for output, item in zip(outputs, records):
                        expected = item[arm]["logits"]
                        torch.testing.assert_close(
                            output, expected, atol=BUDGET, rtol=BUDGET
                        )
                base, candidate = timings["baseline"], timings["candidate"]
                stream.write(
                    json.dumps(
                        {
                            "block": block,
                            "status": "PASS",
                            "order": order,
                            "baseline": base,
                            "candidate": candidate,
                            "speedup": base["elapsed_ms"] / candidate["elapsed_ms"],
                        }
                    )
                    + "\n"
                )
                stream.flush()
    meta_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "evidence_version": 2,
                "scope": "SYNTHETIC_PARARNN_FULL_MODEL_PAPER_CONTROL",
                "paired_blocks": args.blocks,
                "raw_sha256": sha256(args.output),
                "qualification_sha256": sha256(qualification),
                "qualification_audit_sha256": sha256(audit_path),
                "qualification_audit_status": report["status"],
                "qualification_audit_tensor_pairs": report["tensor_pairs"],
                "qualification_max_abs": report["max_abs"],
                "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": sha256(args.manifest),
                "extension_sha256": args.expected_extension_sha,
                "native_gate_audit_sha256": sha256(args.native_gate_audit),
                "gpu_uuid": actual_uuid,
                "dtype": "float32",
                "model": {"cell": f"Para{args.cell.upper()}", "layers": 4, "width": 64},
                "workload": {
                    "batch": args.batch,
                    "concurrency": args.concurrency,
                    "steps": 128,
                },
                "baseline": "ParaRNN sequential",
                "candidate": f"ParaRNN {args.mode}",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
