"""Run the complete Monostich-2 GDN-2/GQA checkpoint on CPU."""

import argparse
import ast
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gdn2_torch_reference import load_reference
from tokenizers import Tokenizer


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual - expected).abs()
    allowed = 1e-4 + 1e-4 * expected.abs()
    failed = (
        ~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)
    )
    return {
        "pass": not failed.any().item(),
        "max_abs": difference.max().item(),
        "worst_normalized": (difference / allowed).max().item(),
        "failed_elements": int(failed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fla-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(4)
    started = time.time()
    prompt_source = Path(__file__).with_name("gla_reference_gate.py")
    prompts = next(
        ast.literal_eval(node.value)
        for node in ast.parse(prompt_source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PROMPTS"
            for target in node.targets
        )
    )
    assert len(prompts) == 12
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    inputs = [tokenizer.encode(prompt).ids[:5] for prompt in prompts]
    assert all(len(ids) == 5 for ids in inputs)
    meta_path = args.output.with_suffix(".meta.json")
    meta = {
        "status": "RUNNING",
        "scope": "MONOSTICH2_BASE_149M_FULL_CPU_BATCH_VS_SERIAL_P5_G4",
        "pid": os.getpid(),
        "started": started,
        "torch": torch.__version__,
        "device": "cpu",
        "model_revision": "7779182b4dfad5939827f40cd1c819843a6825e2",
        "fla_source_revision": "cbb0a72efb55c18ca0ef4f298298317573ad2cb3",
        "input_ids": inputs,
        "budget": {"atol": 1e-4, "rtol": 1e-4},
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("gdn2_torch_reference.py"),
                prompt_source,
                args.model / "tiny_gdn/model.py",
                args.model / "tiny_gdn/config.py",
                args.fla_source / "fla/ops/gdn2/naive.py",
            )
        },
        "performance_claim": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    model = load_reference(args.model, args.fla_source)
    meta.update(
        parameters=sum(p.numel() for p in model.parameters()), checkpoint_tensors=586
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    rows, snapshots = [], []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in (1, 2, 4):
            for case in range(3):
                ids = torch.tensor(inputs[4 * case : 4 * case + batch])
                for step in range(4):
                    batched = model(ids, logits_to_keep=1).logits[:, 0]
                    serial = torch.cat(
                        [
                            model(ids[index : index + 1], logits_to_keep=1).logits[:, 0]
                            for index in range(batch)
                        ]
                    )
                    check = compare(batched, serial)
                    tokens = batched.argmax(-1)
                    token_equal = torch.equal(tokens, serial.argmax(-1))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "logits": check,
                        "token_equal": token_equal,
                        "pass": check["pass"] and token_equal,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps(row), flush=True)
                    snapshots.append(
                        {
                            "batch": batch,
                            "case": case,
                            "step": step,
                            "batched_logits": batched.clone(),
                            "serial_logits": serial.clone(),
                            "tokens": tokens.clone(),
                        }
                    )
                    ids = torch.cat((ids, tokens[:, None]), dim=1)
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    passed = all(row["pass"] for row in rows)
    meta.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=9,
        batch_steps=len(rows),
        token_choices=sum(row["batch"] for row in rows),
        tensor_sha256=tensor_sha,
        snapshot_scope="Both complete 49152-logit arrays and generated IDs at every step",
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
