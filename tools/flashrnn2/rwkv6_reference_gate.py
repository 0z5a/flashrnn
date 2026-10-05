"""Compare complete RWKV6 cached generation with full-prefix evaluation."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
from gla_reference_gate import PROMPTS, compare
from rwkv6_torch_reference import RWKV6Reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    started = time.time()
    tokenizer_path = args.model / "tokenization_rwkv5.py"
    tokenizer_bytes = tokenizer_path.read_bytes()
    assert (
        hashlib.sha1(
            f"blob {len(tokenizer_bytes)}\0".encode() + tokenizer_bytes
        ).hexdigest()
        == "037a570924fe0b56bc096b490b7acd5bcc64be02"
    )
    vocab = args.model / "vocab.txt"
    for line in vocab.read_text().splitlines():
        assert isinstance(ast.literal_eval(line), bytes)
    spec = importlib.util.spec_from_file_location(
        "rwkv6_pinned_tokenizer", tokenizer_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.Rwkv5Tokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = [tokenizer.encode(text, add_special_tokens=False)[:5] for text in PROMPTS]
    assert all(len(ids) == 5 for ids in inputs)
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": started,
        "scope": "FULL_FLA_RWKV6_1_6B_CPU_CACHED_VS_FULL_PREFIX_P5_G4",
        "device": "cpu",
        "torch": torch.__version__,
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {"atol": 1e-5, "rtol": 1e-5},
        "input_ids": inputs,
        "prompts": PROMPTS,
        "performance_claim": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("rwkv6_torch_reference.py"),
                Path(__file__).with_name("gla_reference_gate.py"),
                tokenizer_path,
                vocab,
            )
        },
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    model = RWKV6Reference(args.model, args.source)
    metadata["model"] = model.provenance
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows, snapshots = [], []
    with torch.inference_mode(), args.output.open("x") as handle:
        for batch in (1, 2, 4):
            for case in range(3):
                ids = torch.tensor(inputs[4 * case : 4 * case + batch])
                full_ids = ids.clone()
                logits, cache = model(ids)
                cached_logits, reference_logits, tokens = [], [], []
                for step in range(4):
                    expected, full_cache = model(full_ids)
                    left, right = logits[:, -1], expected[:, -1]
                    checks = {"logits": compare(left, right, 1e-3)}
                    layer_checks = {}
                    for field in ("recurrent_state", "conv_state", "ffn_state"):
                        values = []
                        for a, b in zip(cache.states, full_cache.states, strict=True):
                            assert a[field] is not None and b[field] is not None
                            values.append(compare(a[field], b[field], 1e-5))
                        assert len(values) == 24
                        checks[field] = {
                            "pass": all(x["pass"] for x in values),
                            "max_abs": max(x["max_abs"] for x in values),
                            "worst_normalized": max(
                                x["worst_normalized"] for x in values
                            ),
                            "failed_elements": sum(
                                x["failed_elements"] for x in values
                            ),
                        }
                        layer_checks[field] = values
                    next_ids = left.argmax(-1, keepdim=True)
                    equal = torch.equal(next_ids, right.argmax(-1, keepdim=True))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "checks": checks,
                        "layers": layer_checks,
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
                    reference_logits.append(right.clone())
                    tokens.append(next_ids[:, 0].clone())
                    if step < 3:
                        full_ids = torch.cat((full_ids, next_ids), dim=1)
                        logits, cache = model(next_ids, cache)
                saved = {
                    "batch": batch,
                    "case": case,
                    "input_ids": ids,
                    "cached_logits": torch.stack(cached_logits),
                    "full_prefix_logits": torch.stack(reference_logits),
                    "generated_ids": torch.stack(tokens),
                }
                if case == 0:
                    saved["cached_final_states"] = cache.states
                    saved["full_prefix_final_states"] = full_cache.states
                snapshots.append(saved)
    tensor_path = args.output.with_suffix(".pt")
    torch.save(snapshots, tensor_path)
    with tensor_path.open("rb") as handle:
        tensor_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    passed = all(row["pass"] for row in rows)
    metadata.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=9,
        batch_steps=len(rows),
        token_choices=sum(r["batch"] for r in rows),
        tensor_sha256=tensor_sha,
        snapshot_scope="Both logits at every step; both complete final caches of case0 at every batch",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
