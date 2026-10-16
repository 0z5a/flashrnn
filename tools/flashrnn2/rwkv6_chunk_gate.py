"""Checkpoint-derived RWKV6 chunk-prefix correctness gate on CPU."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from torch.nn import functional as F

SOURCE_SHA = "180ff8eee2e53315e6f2ae2553bdfdb1e7455de03c2d1e762778d4edd5ce85c6"
WEIGHT_SHA = "db508532c15e96d89566080cc39cadfa517d4cba83f782a59faa7f304b00da0a"
LAYER_SHA = "5cd488783ed4d8e4cf5c27c8d42fd0f5025e16245fbae8a54fb324759d7c8af1"
NORM_SHA = "75104b5df93526180293ede0b1ed64ca131a6c663e4c77a1d8be03f36bdcf948"


def official_recurrence(path: Path):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA
    node = next(
        node
        for node in ast.parse(data).body
        if isinstance(node, ast.FunctionDef) and node.name == "naive_recurrent_rwkv6"
    )
    namespace = {"torch": torch, "Optional": Optional}
    exec(  # noqa: S102
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["naive_recurrent_rwkv6"]


def checkpoint_gates(model: Path, ids: list[list[int]]):
    checkpoint = model / "model.safetensors"
    with checkpoint.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice("model.embeddings.weight")
        x = torch.stack(
            [torch.stack([embedding[token].float() for token in row]) for row in ids]
        )
        for norm in ("pre_norm", "attn_norm"):
            root = "model.layers.0." + norm
            x = F.layer_norm(
                x,
                (2048,),
                tensors.get_tensor(root + ".weight").float(),
                tensors.get_tensor(root + ".bias").float(),
                eps=1e-6,
            )
        prefix = "model.layers.0.attn."

        def weight(name: str):
            return tensors.get_tensor(prefix + name).float()

        delta = F.pad(x[:, :-1], (0, 0, 1, 0)) - x
        low = F.linear(
            x + delta * weight("x_proj.0.mu"), weight("x_proj.0.linear.weight")
        )
        mixed = torch.einsum(
            "btnr,hnr->btnh",
            low.reshape(*x.shape[:2], 5, 32).tanh(),
            weight("x_proj.2.weight").reshape(2048, 5, 32),
        )
        r_mu, w_mu, k_mu, v_mu, _ = (mixed + weight("x_bias")).unbind(-2)

        def linear(name: str, mu: torch.Tensor):
            return F.linear(x + delta * mu, weight(name + ".linear.weight"))

        r = linear("r_proj", r_mu)
        k = linear("k_proj", k_mu)
        v = linear("v_proj", v_mu)
        w_input = x + delta * w_mu
        w = F.linear(
            F.linear(w_input, weight("w_proj.linear.lora.0.weight")).tanh(),
            weight("w_proj.linear.lora.2.weight"),
            weight("w_proj.linear.lora.2.bias"),
        )
        reshape = lambda y: y.reshape(*y.shape[:2], 32, 64).transpose(1, 2).contiguous()
        return reshape(r), reshape(k), reshape(v), reshape(-w.exp()), weight("bonus")


def chunk_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    bonus: torch.Tensor,
    initial: torch.Tensor | None,
    chunk_size: int,
):
    batch, heads, steps, key_dim = q.shape
    value_dim = v.shape[-1]
    decay = w.exp()
    summaries = []
    for start in range(0, steps, chunk_size):
        diagonal = q.new_ones(batch, heads, key_dim)
        update = q.new_zeros(batch, heads, key_dim, value_dim)
        end = min(start + chunk_size, steps)
        for t in range(start, end):
            factor = decay[:, :, t]
            update = (
                update * factor[..., None] + k[:, :, t, :, None] * v[:, :, t, None, :]
            )
            diagonal = diagonal * factor
        summaries.append((start, end, diagonal, update))

    state = (
        q.new_zeros(batch, heads, key_dim, value_dim)
        if initial is None
        else initial.clone()
    )
    boundaries = []
    for _, _, diagonal, update in summaries:
        boundaries.append(state)
        state = state * diagonal[..., None] + update

    output = q.new_empty(batch, heads, steps, value_dim)
    for (start, end, _, _), boundary in zip(summaries, boundaries, strict=True):
        local = boundary
        for t in range(start, end):
            kv = k[:, :, t, :, None] * v[:, :, t, None, :]
            output[:, :, t] = (
                (local + bonus[None, ..., None] * kv) * q[:, :, t, :, None]
            ).sum(-2)
            local = local * decay[:, :, t, :, None] + kv
    return output, state


def serial_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    bonus: torch.Tensor,
    initial: torch.Tensor | None,
):
    batch, heads, steps, key_dim = q.shape
    state = (
        q.new_zeros(batch, heads, key_dim, v.shape[-1])
        if initial is None
        else initial.clone()
    )
    output = q.new_empty(batch, heads, steps, v.shape[-1])
    for t in range(steps):
        kv = k[:, :, t, :, None] * v[:, :, t, None, :]
        output[:, :, t] = (
            (state + bonus[None, ..., None] * kv) * q[:, :, t, :, None]
        ).sum(-2)
        state = state * w[:, :, t, :, None].exp() + kv
    return output, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not args.output.exists()
    reference = official_recurrence(args.source / "fla--ops--rwkv6--recurrent_naive.py")
    for name, expected in (
        ("fla--layers--rwkv6.py", LAYER_SHA),
        ("fla--modules--layernorm.py", NORM_SHA),
    ):
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == expected
    metadata = json.loads((args.model / "verified-manifest.json").read_text())
    assert metadata["revision"] == "94302fd437462b5110b06a9e83b32b8a1684e8d4"
    prompt_ids = json.loads(
        (args.output.parent / "rwkv6-reference-r1.meta.json").read_text()
    )["input_ids"]
    stream = [token for prompt in prompt_ids for token in prompt]
    assert len(stream) == 60
    ids = [
        [stream[(position + shift) % 60] for position in range(134)]
        for shift in (0, 7, 19, 31)
    ]
    with torch.inference_mode():
        projected = checkpoint_gates(args.model, ids)
        rows = []
        for dtype, name, atol, rtol in (
            (torch.float32, "pinned_FLA_FP32", 1e-4, 1e-4),
            (torch.float64, "serial_FP64", 1e-8, 1e-10),
        ):
            base = [tensor.to(dtype) for tensor in projected]
            for batch, steps, resumed in (
                (1, 5, False),
                (2, 17, False),
                (4, 127, False),
                (4, 128, True),
                (4, 129, True),
            ):
                start = 5 if resumed else 0
                q, k, v, w = (
                    tensor[:batch, :, start : start + steps] for tensor in base[:4]
                )
                bonus = base[4]
                prefix = [tensor[:batch, :, :5] for tensor in base[:4]]
                initial = (
                    (
                        reference(*prefix, bonus, 1.0, None, True)[1]
                        if dtype == torch.float32
                        else serial_recurrence(*prefix, bonus, None)[1]
                    )
                    if resumed
                    else None
                )
                expected, final = (
                    reference(q, k, v, w, bonus, 1.0, initial, True)
                    if dtype == torch.float32
                    else serial_recurrence(q, k, v, w, bonus, initial)
                )
                for chunk_size in (1, 7, 16, 64):
                    actual, state = chunk_prefix(q, k, v, w, bonus, initial, chunk_size)
                    output_error = (actual - expected).abs()
                    state_error = (state - final).abs()
                    output_tolerance = atol + rtol * expected.abs()
                    state_tolerance = atol + rtol * final.abs()
                    rows.append(
                        {
                            "reference": name,
                            "batch": batch,
                            "steps": steps,
                            "resumed": resumed,
                            "chunk_size": chunk_size,
                            "output_max_abs": output_error.max().item(),
                            "state_max_abs": state_error.max().item(),
                            "output_failed_elements": int(
                                (output_error > output_tolerance).sum()
                            ),
                            "state_failed_elements": int(
                                (state_error > state_tolerance).sum()
                            ),
                            "output_worst_normalized": (output_error / output_tolerance)
                            .max()
                            .item(),
                            "state_worst_normalized": (state_error / state_tolerance)
                            .max()
                            .item(),
                            "output_pass": torch.all(
                                output_error <= output_tolerance
                            ).item(),
                            "state_pass": torch.all(
                                state_error <= state_tolerance
                            ).item(),
                        }
                    )
        q, k, v, w = (tensor[:, :, :5] for tensor in projected[:4])
        bonus = projected[4]
        with_bonus, bonus_state = reference(q, k, v, w, bonus, 1.0, None, True)
        without_bonus, no_bonus_state = reference(
            q, k, v, w, torch.zeros_like(bonus), 1.0, None, True
        )
        bonus_order_pass = not torch.equal(with_bonus, without_bonus) and torch.equal(
            bonus_state, no_bonus_state
        )
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fp32_passed = sum(
        row["output_pass"] and row["state_pass"]
        for row in rows
        if row["reference"] == "pinned_FLA_FP32"
    )
    fp64_passed = sum(
        row["output_pass"] and row["state_pass"]
        for row in rows
        if row["reference"] == "serial_FP64"
    )
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "scope": "RWKV6_LAYER0_CHECKPOINT_PROJECTED_CHUNK_CPU_CORRECTNESS",
                "status": "PASS" if fp32_passed == 20 else "NUMERICAL_FAILED",
                "device": "cpu",
                "torch": torch.__version__,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "checkpoint_sha256": WEIGHT_SHA,
                "source_sha256": SOURCE_SHA,
                "layer_source_sha256": LAYER_SHA,
                "norm_source_sha256": NORM_SHA,
                "gate_inputs": "First-block two norms and attention projections of four synthetic 134-token streams assembled from 12 pinned five-token prompts",
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "tolerances": {
                    "pinned_FLA_FP32": {"atol": 1e-4, "rtol": 1e-4},
                    "serial_FP64": {"atol": 1e-8, "rtol": 1e-10},
                },
                "rows": len(rows),
                "rows_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "fp32_passed": fp32_passed,
                "fp64_passed": fp64_passed,
                "bonus_order_pass": bonus_order_pass,
                "performance_claim": False,
            },
            indent=2,
        )
        + "\n"
    )
    assert len(rows) == 40 and bonus_order_pass
    assert all(
        row["output_pass"] and row["state_pass"]
        for row in rows
        if row["reference"] == "serial_FP64"
    )
    raise SystemExit(
        0 if all(row["output_pass"] and row["state_pass"] for row in rows) else 1
    )


if __name__ == "__main__":
    main()
