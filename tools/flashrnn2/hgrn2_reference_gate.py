"""Complete HGRN2 checkpoint: cached versus full-prefix CPU generation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gla_reference_gate import PROMPTS, compare
from hgrn2_torch_reference import HGRN2Reference
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4]
    )
    args = parser.parse_args()
    assert len(set(args.batches)) == len(args.batches)
    assert not args.output.exists()
    assert not list(args.output.parent.glob(args.output.stem + "-b*-c*.pt"))
    torch.set_num_threads(1)
    tokenizer_path = args.model / "tokenizer.json"
    data = tokenizer_path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == "43e6daf936dc0f953cb867ec864adab78f92d9ce"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    hf_tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    tokenized = [tokenizer.encode(text).ids for text in PROMPTS]
    assert tokenized == [
        hf_tokenizer.encode(text, add_special_tokens=True) for text in PROMPTS
    ]
    inputs = [ids[:5] for ids in tokenized]
    assert all(len(ids) == 5 for ids in inputs)
    paths = [
        Path(__file__),
        Path(__file__).with_name("hgrn2_torch_reference.py"),
        Path(__file__).with_name("gla_torch_reference.py"),
        Path(__file__).with_name("gla_reference_gate.py"),
        tokenizer_path,
    ]
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "scope": "FULL_HGRN2_1_3B_CPU_CACHED_VS_FULL_PREFIX_P5_G4",
        "device": "cpu",
        "torch": torch.__version__,
        "performance_claim": False,
        "native_accelerated_fla_qualified": False,
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {"atol": 1e-5, "rtol": 1e-5},
        "prompts": PROMPTS,
        "input_ids": inputs,
        "tokenizer_control": "TOKENIZERS_AND_TRANSFORMERS_FAST_MATCH_ALL_12_PROMPTS",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "snapshots": [],
        "batches": args.batches,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    model = HGRN2Reference(args.model, args.source, args.common)
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
                    values = [
                        compare(a["recurrent_state"], b["recurrent_state"], 1e-5)
                        for a, b in zip(cache.states, full_cache.states, strict=True)
                    ]
                    assert len(values) == 24
                    checks = {
                        "logits": compare(left, right, 1e-3),
                        "recurrent_state": {
                            "pass": all(x["pass"] for x in values),
                            "max_abs": max(x["max_abs"] for x in values),
                            "worst_normalized": max(
                                x["worst_normalized"] for x in values
                            ),
                            "failed_elements": sum(
                                x["failed_elements"] for x in values
                            ),
                        },
                    }
                    next_ids = left.argmax(-1, keepdim=True)
                    equal = torch.equal(next_ids, right.argmax(-1, keepdim=True))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "checks": checks,
                        "layers": values,
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
                        cached_final_states=cache.states,
                        full_prefix_final_states=full_cache.states,
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
        snapshot_scope="Both logits for every selected step; both final caches for case0 at each selected batch, saved per case",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
