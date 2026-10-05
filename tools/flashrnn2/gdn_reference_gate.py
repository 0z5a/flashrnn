"""Complete Gated DeltaNet checkpoint: cached versus full-prefix CPU generation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gdn_torch_reference import GatedDeltaNetReference
from gla_reference_gate import PROMPTS, compare
from tokenizers import Tokenizer


def summarize(values: list[dict]) -> dict:
    return {
        "pass": all(item["pass"] for item in values),
        "max_abs": max(item["max_abs"] for item in values),
        "worst_normalized": max(item["worst_normalized"] for item in values),
        "failed_elements": sum(item["failed_elements"] for item in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--delta-source", type=Path, required=True)
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
        == "59604a99f735f090823b4ccfc6bb137b2a485fc4"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenized = [tokenizer.encode(text).ids for text in PROMPTS]
    inputs = [ids[:5] for ids in tokenized]
    assert all(len(ids) == 5 for ids in inputs)
    paths = [
        Path(__file__),
        Path(__file__).with_name("gdn_torch_reference.py"),
        Path(__file__).with_name("deltanet_torch_reference.py"),
        Path(__file__).with_name("gla_torch_reference.py"),
        Path(__file__).with_name("gla_reference_gate.py"),
        tokenizer_path,
    ]
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "scope": "FULL_GATED_DELTANET_340M_CPU_CACHED_VS_FULL_PREFIX_P5_G4",
        "device": "cpu",
        "torch": torch.__version__,
        "performance_claim": False,
        "native_accelerated_fla_qualified": False,
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {"atol": 1e-5, "rtol": 1e-5},
        "prompts": PROMPTS,
        "input_ids": inputs,
        "tokenizer_control": "PINNED_TOKENIZER_JSON_GIT_BLOB_VERIFIED_HF_WRAPPER_NOT_RUN",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "snapshots": [],
        "batches": args.batches,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    model = GatedDeltaNetReference(args.model, args.source, args.delta_source)
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
                    recurrent = [
                        compare(a["recurrent_state"], b["recurrent_state"], 1e-5)
                        for a, b in zip(cache.states, full_cache.states, strict=True)
                    ]
                    convolution = [
                        compare(a, b, 1e-5)
                        for left_state, right_state in zip(
                            cache.states, full_cache.states, strict=True
                        )
                        for a, b in zip(
                            left_state["conv_state"],
                            right_state["conv_state"],
                            strict=True,
                        )
                    ]
                    assert len(recurrent) == 24 and len(convolution) == 72
                    checks = {
                        "logits": compare(left, right, 1e-3),
                        "recurrent_state": summarize(recurrent),
                        "convolution": summarize(convolution),
                    }
                    next_ids = left.argmax(-1, keepdim=True)
                    equal = torch.equal(next_ids, right.argmax(-1, keepdim=True))
                    row = {
                        "batch": batch,
                        "case": case,
                        "step": step,
                        "checks": checks,
                        "recurrent_layers": recurrent,
                        "convolution_layers": convolution,
                        "token_equal": equal,
                        "pass": equal
                        and all(value["pass"] for value in checks.values()),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in row.items()
                                if key not in ("recurrent_layers", "convolution_layers")
                            }
                        ),
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
        token_choices=sum(row["batch"] for row in rows),
        snapshot_scope="Both logits for every step; both final recurrent and convolution caches for case0 at each batch",
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
