"""Cross-check full reverse-mapped BlinkDL CPU generation against saved FLA."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from gla_reference_gate import compare
from rwkv6_blinkdl_reference import blinkdl_reference
from rwkv6_torch_reference import RWKV6Reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".pt").exists()
    torch.set_num_threads(1)
    oracle_meta = json.loads(args.oracle.with_suffix(".meta.json").read_text())
    with args.oracle.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == oracle_meta["tensor_sha256"]
        )
    snapshots = torch.load(args.oracle, weights_only=True, map_location="cpu")
    reference = RWKV6Reference(args.model, args.source)
    model = blinkdl_reference(
        reference.state_dict(), args.source / "ChatRWKV--RWKV_v6_demo.py"
    )
    del reference
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "scope": "FULL_RWKV6_REVERSE_MAPPED_BLINKDL_CPU_B1_P5_G4_THREE_PROMPTS",
        "model": model.provenance,
        "oracle_tensor_sha256": oracle_meta["tensor_sha256"],
        "oracle_checkpoint": oracle_meta["model"]["checkpoint"],
        "logits_budget": {"atol": 1e-3, "rtol": 1e-3},
        "state_budget": {"atol": 1e-5, "rtol": 1e-5},
        "input_policy": "Teacher force identical saved FLA tokens; independently report native greedy choices",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("rwkv6_blinkdl_reference.py"),
                Path(__file__).with_name("rwkv6_torch_reference.py"),
                Path(__file__).with_name("gla_reference_gate.py"),
            )
        },
        "performance_claim": False,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows, saved, state_checks = [], [], []
    with torch.inference_mode(), args.output.open("x") as output:
        for oracle in snapshots:
            if oracle["batch"] != 1:
                continue
            state = None
            for token in oracle["input_ids"][0].tolist():
                logits, state = model(token, state)
            logits_saved = []
            for step in range(4):
                expected = oracle["cached_logits"][step, 0]
                measured = compare(logits, expected, 1e-3)
                token = int(oracle["generated_ids"][step, 0])
                row = {
                    "batch": 1,
                    "case": oracle["case"],
                    "step": step,
                    "logits": measured,
                    "native_greedy": int(logits.argmax()),
                    "fla_greedy": token,
                    "token_equal": int(logits.argmax()) == token,
                }
                row["pass"] = measured["pass"] and row["token_equal"]
                rows.append(row)
                logits_saved.append(logits.clone())
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(json.dumps(row), flush=True)
                if step < 3:
                    logits, state = model(token, state)
            converted = []
            for layer in range(24):
                start = 66 * layer
                converted.append(
                    {
                        "ffn_state": state[start][None].clone(),
                        "conv_state": state[start + 1][None].clone(),
                        "recurrent_state": state[start + 2 : start + 66]
                        .reshape(1, 32, 64, 64)
                        .clone(),
                    }
                )
            saved.append(
                {
                    "batch": 1,
                    "case": oracle["case"],
                    "native_logits": torch.stack(logits_saved),
                    "fla_logits": oracle["cached_logits"][:, 0],
                    "input_ids": oracle["input_ids"],
                    "teacher_forced_ids": oracle["generated_ids"],
                    "native_final_states": converted,
                }
            )
            if oracle["case"] == 0:
                for layer, (native, expected) in enumerate(
                    zip(converted, oracle["cached_final_states"], strict=True)
                ):
                    for field in ("recurrent_state", "conv_state", "ffn_state"):
                        state_checks.append(
                            {
                                "layer": layer,
                                "field": field,
                                **compare(native[field], expected[field], 1e-5),
                            }
                        )
    tensor_path = args.output.with_suffix(".pt")
    torch.save(saved, tensor_path)
    with tensor_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    passed = all(row["pass"] for row in rows + state_checks)
    metadata.update(
        status="PASS" if passed else "NUMERICAL_FAILED",
        finished=time.time(),
        cases=3,
        batch_steps=len(rows),
        matching_tokens=sum(row["token_equal"] for row in rows),
        final_state_checks=state_checks,
        tensor_sha256=digest,
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
