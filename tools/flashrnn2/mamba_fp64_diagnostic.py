"""Compare saved implementations with an explicitly promoted FP64 math reference.

This changes the native precision contract. It diagnoses rounding and never
replaces native qualification, adjusts tolerances, or supplies performance data.
"""

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import mamba_native_runtime as runtime
import torch
from compare_mamba_diagnostics import load
from mamba_script_diagnostic import details
from mamba_script_gate import verified_metadata
from torch import nn


class PromoteFP64(ast.NodeTransformer):
    def __init__(self) -> None:
        self.sites: list[dict] = []

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "float":
            self.sites.append(
                {"line": node.lineno, "from": "float()", "to": "double()"}
            )
            node.func.attr = "double"
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and node.attr == "float32"
        ):
            self.sites.append({"line": node.lineno, "from": "float32", "to": "float64"})
            node.attr = "float64"
        return node


def fp64_functions(path: Path) -> dict:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == runtime.SOURCE_SHA
    classes = {n.name: n for n in ast.parse(data).body if isinstance(n, ast.ClassDef)}
    selected = ast.parse("from __future__ import annotations\n").body
    selected += [classes["MambaCache"], classes["MambaRMSNorm"]]
    for cls, method, name in (
        ("MambaMixer", "slow_forward", "mixer_forward"),
        ("MambaBlock", "forward", "block_forward"),
        ("MambaModel", "forward", "backbone_forward"),
        ("MambaForCausalLM", "forward", "lm_forward"),
    ):
        node = next(
            n
            for n in classes[cls].body
            if isinstance(n, ast.FunctionDef) and n.name == method
        )
        node.name, node.decorator_list = name, []
        selected.append(node)
    promotion = PromoteFP64()
    module = promotion.visit(ast.Module(body=selected, type_ignores=[]))
    assert len(promotion.sites) == 6, promotion.sites
    namespace = {
        "torch": torch,
        "nn": nn,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "promoted_sites": promotion.sites,
        "promoted_ast_sha256": hashlib.sha256(ast.dump(module).encode()).hexdigest(),
    }
    exec(  # noqa: S102 - pinned source, explicit precision transformation
        compile(module, str(path), "exec"), namespace
    )
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--remote-cpu", type=Path, required=True)
    parser.add_argument("--remote-cuda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    started = time.time()
    contracts = verified_metadata(args.goldens)
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    references = {"local_native_fp32": cases}
    saved_provenance = {}
    for name, path in (
        ("remote_cpu_fp32", args.remote_cpu),
        ("remote_cuda_fp32", args.remote_cuda),
    ):
        saved, meta = load(path)
        assert meta["oracles"]["artifact_sha256"] == contracts["artifact_sha256"]
        assert len(saved) == len(cases) == 3
        assert all(
            torch.equal(
                row["prefill"][0][:, -1].argmax(-1, keepdim=True), case["next_ids"]
            )
            for row, case in zip(saved, cases)
        )
        references[name] = saved
        saved_provenance[name] = {
            key: meta[key] for key in ("torch", "device", "tensor_sha256", "model")
        }
    functions = fp64_functions(args.source)
    runtime.native_functions = fp64_functions
    model = runtime.NativeMamba(args.model, args.source, args.config, "cpu").double()
    assert all(
        meta["model"]["artifact_sha256"] == model.metadata["artifact_sha256"]
        for meta in saved_provenance.values()
    )
    assert model.lm_head.weight is model.backbone.embeddings.weight
    assert sum(p.numel() for p in model.parameters()) == 129135360
    assert all(p.dtype == torch.float64 for p in model.parameters())
    rows, snapshots = [], []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            states = [
                torch.zeros_like(t, dtype=torch.float64) for t in case["prefill"][1:]
            ]
            prefill = model.prefill(case["input_ids"], *states)
            prefix = tuple(t.clone() for t in prefill)
            next_ids = prefill[0][:, -1].argmax(-1, keepdim=True)
            decode = model.decode(case["next_ids"], prefill[1], prefill[2])
            snapshot = {"prefill": prefix, "decode": decode, "next_ids": next_ids}
            assert all(
                t.dtype == torch.float64
                for phase in ("prefill", "decode")
                for t in snapshot[phase]
            )
            snapshots.append(snapshot)
            row = {
                "case": index,
                "fp64_prefill_token_equal": torch.equal(next_ids, case["next_ids"]),
                "comparisons": {},
            }
            for name, saved in references.items():
                row["comparisons"][name] = {
                    phase: details(
                        tuple(t.double() for t in saved[index][phase]),
                        snapshot[phase],
                        contracts,
                    )
                    for phase in ("prefill", "decode")
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {
        "scope": "FULL_MODEL_FP64_MATH_DIAGNOSIS_NOT_NATIVE_IMPLEMENTATION_QUALIFICATION",
        "started": started,
        "finished": time.time(),
        "torch": torch.__version__,
        "promoted_sites": functions["promoted_sites"],
        "promoted_ast_sha256": functions["promoted_ast_sha256"],
        "model": model.metadata,
        "oracles": contracts,
        "saved_provenance": saved_provenance,
        "cases": rows,
        "tensor_sha256": tensor_sha,
        "decode_input_policy": "Original oracle next-token IDs; prefill token equality is recorded separately",
        "original_failures_replaced": False,
        "performance_claim": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                args.source,
                args.config,
                Path(runtime.__file__),
                Path(details.__code__.co_filename),
                Path(verified_metadata.__code__.co_filename),
                Path(load.__code__.co_filename),
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
