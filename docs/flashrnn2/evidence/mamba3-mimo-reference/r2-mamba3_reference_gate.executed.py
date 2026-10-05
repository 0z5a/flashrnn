"""Complete pinned Mamba-3 SISO checkpoint: cached versus full-prefix CPU generation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gla_reference_gate import PROMPTS, compare
from mamba3_mimo_torch_reference import Mamba3MIMOReference
from mamba3_torch_reference import Mamba3Reference
from transformers import AutoTokenizer

STATE_NAMES = ("angle", "ssm", "key", "value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("siso", "mimo"), default="siso")
    parser.add_argument("--ssm-budget", type=float, default=1e-5)
    parser.add_argument("--key-budget", type=float, default=1e-5)
    parser.add_argument(
        "--batches", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4]
    )
    args = parser.parse_args()
    assert len(set(args.batches)) == len(args.batches)
    assert args.ssm_budget > 0 and args.key_budget > 0
    assert not args.output.exists()
    assert not list(args.output.parent.glob(args.output.stem + "-b*-c*.pt"))
    torch.set_num_threads(1)
    tokenizer_receipt = json.loads(
        (args.output.parent / f"mamba3-{args.variant}-tokenizer-r1.json").read_text()
    )
    assert tokenizer_receipt["revision"] == "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    for entry in tokenizer_receipt["files"]:
        path = args.model / entry["file"]
        assert path.stat().st_size == entry["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = [tokenizer(text).input_ids[:5] for text in PROMPTS]
    assert all(len(ids) == 5 and max(ids) < 128256 for ids in inputs)
    sources = [Path(__file__), Path(__file__).with_name("mamba3_torch_reference.py")]
    if args.variant == "mimo":
        sources.append(Path(__file__).with_name("mamba3_mimo_torch_reference.py"))
    state_budgets = {
        "angle": 1e-5,
        "ssm": args.ssm_budget,
        "key": args.key_budget,
        "value": 1e-5,
    }
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "scope": f"FULL_MAMBA3_{args.variant.upper()}_187M_CPU_CACHED_VS_FULL_PREFIX_P5_G4",
        "device": "cpu",
        "torch": torch.__version__,
        "performance_claim": False,
        "native_accelerated_mamba_qualified": False,
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {
            name: {"atol": budget, "rtol": budget}
            for name, budget in state_budgets.items()
        },
        "prompts": PROMPTS,
        "input_ids": inputs,
        "tokenizer": tokenizer_receipt,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "snapshots": [],
        "batches": args.batches,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    model_type = Mamba3Reference if args.variant == "siso" else Mamba3MIMOReference
    model = model_type(args.model, args.source)
    metadata["model"] = model.provenance
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in args.batches:
            for case in range(3):
                ids = torch.tensor(inputs[4 * case : 4 * case + batch])
                full_ids = ids.clone()
                logits, cache = model(ids)
                cached_logits, full_logits, tokens = [], [], []
                for step in range(4):
                    expected, full_cache = model(full_ids)
                    left, right = logits[:, -1], expected[:, -1]
                    layers = []
                    for layer in range(12):
                        a = cache.key_value_memory_dict[layer]
                        b = full_cache.key_value_memory_dict[layer]
                        layers.append(
                            {
                                name: compare(x, y, state_budgets[name])
                                for name, x, y in zip(STATE_NAMES, a, b, strict=True)
                            }
                        )
                    checks = {"logits": compare(left, right, 1e-3)}
                    for name in STATE_NAMES:
                        values = [x[name] for x in layers]
                        checks[name] = {
                            "pass": all(x["pass"] for x in values),
                            "max_abs": max(x["max_abs"] for x in values),
                            "worst_normalized": max(
                                x["worst_normalized"] for x in values
                            ),
                            "failed_elements": sum(
                                x["failed_elements"] for x in values
                            ),
                        }
                    next_ids = left.argmax(-1, keepdim=True)
                    equal = torch.equal(next_ids, right.argmax(-1, keepdim=True))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "checks": checks,
                        "layers": layers,
                        "token_equal": equal,
                        "pass": equal and all(x["pass"] for x in checks.values()),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "layers"}),
                        flush=True,
                    )
                    cached_logits.append(left.clone())
                    full_logits.append(right.clone())
                    tokens.append(next_ids[:, 0].clone())
                    if step < 3:
                        full_ids = torch.cat((full_ids, next_ids), dim=1)
                        logits, cache = model(next_ids, cache)
                snapshot = {
                    "batch": batch,
                    "case": case,
                    "input_ids": ids,
                    "cached_logits": torch.stack(cached_logits),
                    "full_prefix_logits": torch.stack(full_logits),
                    "generated_ids": torch.stack(tokens),
                }
                if case == 0:
                    snapshot.update(
                        cached_final_states=cache.key_value_memory_dict,
                        full_prefix_final_states=full_cache.key_value_memory_dict,
                    )
                path = args.output.with_name(f"{args.output.stem}-b{batch}-c{case}.pt")
                partial = path.with_suffix(".pt.partial")
                assert not partial.exists()
                torch.save(snapshot, partial)
                partial.replace(path)
                with path.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                metadata["snapshots"].append(
                    {
                        "file": path.name,
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                        "batch": batch,
                        "case": case,
                    }
                )
                meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
                del snapshot, cache, full_cache
    passed = all(row["pass"] for row in rows)
    metadata.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=3 * len(args.batches),
        batch_steps=len(rows),
        token_choices=sum(r["batch"] for r in rows),
        snapshot_scope="Both logits for every selected step; both final caches for case0 at each selected batch",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
