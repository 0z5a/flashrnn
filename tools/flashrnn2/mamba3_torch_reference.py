"""Pinned Mamba-3 SISO model bodies with explicit Torch recurrent operations."""

import ast
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

REVISION = "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"
WEIGHT_SHA256 = "b462d55e9f8ab9746be641d4de550b51bf65e55ab662e66835588b5a458c95ec"
SOURCE_SHA256 = {
    "mamba3.py": "930c3dfa04dea8444b1c9ee8b6ac9cbbc7ef492dc5ae1f4c83051ef953eca33c",
    "mlp.py": "d3aa360ae67608d2582975f25ec9adc57613e442f2edefcd88e91363e6de5021",
    "block.py": "b62e755195c277a027c5d9cc8d576a8ae4a1d1317143b91370b2f8ce683b4cc1",
    "mamba3_mimo_rotary_step.py": "e117b5da3d2ddfcfa66673a88dfd88a6d688100291d3b6a1b0b3c24c82ef79d6",
    "mamba3_step_fn.py": "5c82f3936308cfc90bb3bcdfd410c60fc80dfc3afe72c4338831341398cd7ca4",
}


class RMSNormTorch(nn.Module):
    def __init__(self, width, eps=1e-5, device=None, dtype=None, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.bias = None
        self.eps = eps

    def forward(self, x, gate=None):
        y = F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        return y if gate is None else y * F.silu(gate)


def layer_norm_torch(
    x,
    weight,
    bias,
    residual=None,
    prenorm=True,
    residual_in_fp32=True,
    eps=1e-5,
    is_rms_norm=True,
):
    assert bias is None and is_rms_norm
    residual = x if residual is None else x + residual
    y = F.rms_norm(residual.to(weight.dtype), (weight.numel(),), weight, eps)
    return (y, residual.float() if residual_in_fp32 else residual) if prenorm else y


def extract(path: Path, names: set[str], namespace: dict) -> None:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256[path.name]
    nodes = [
        node
        for node in ast.parse(path.read_bytes()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    module = ast.Module(
        body=ast.parse("from __future__ import annotations\n").body + nodes,
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102


def native_namespace(source: Path) -> dict:
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "rearrange": rearrange,
        "repeat": repeat,
        "RMSNorm": RMSNormTorch,
        "RMSNormGated": RMSNormTorch,
        "layer_norm_fn": layer_norm_torch,
        "mamba3_mimo_combined": None,
    }
    extract(
        source / "mamba3_mimo_rotary_step.py",
        {"apply_rotary_qk_inference_reference"},
        namespace,
    )
    extract(
        source / "mamba3_step_fn.py", {"selective_state_update_fused_ref_v2"}, namespace
    )
    rotary = namespace["apply_rotary_qk_inference_reference"]
    update = namespace["selective_state_update_fused_ref_v2"]

    def rotary_step(
        q,
        k,
        angle_state,
        angle_proj,
        dt,
        bias_q,
        bias_k,
        conjugate=False,
        inplace=False,
        rotate_pairwise=True,
    ):
        assert not conjugate and not inplace and rotate_pairwise
        q, k, angle = rotary(q, k, angle_state, angle_proj, dt, bias_q, bias_k)
        return q, k, angle.remainder(2 * math.pi)

    def torch_step(
        state,
        k_state,
        v_state,
        a,
        k,
        q,
        d,
        x,
        dt,
        trap,
        xproj,
        outproj=None,
        state_out=None,
        out=None,
        z=None,
        zproj=None,
        tile_D=64,
        num_warps=4,
    ):
        assert state_out is None and out is not None and tile_D == 64 and num_warps == 4
        value, next_state = update(
            state,
            a,
            k,
            q,
            xproj,
            x,
            zproj,
            z,
            dt,
            k_state,
            v_state,
            trap,
            d,
            outproj,
        )
        state.copy_(next_state)
        out.copy_(value)

    def torch_combined(
        Q,
        K,
        V,
        ADT,
        DT,
        Trap,
        Q_bias,
        K_bias,
        Angles,
        D,
        Z,
        chunk_size=64,
        Input_States=None,
        return_final_states=False,
        cu_seqlens=None,
    ):
        assert Input_States is None and cu_seqlens is None and chunk_size == 64
        batch, length, heads, width = V.shape
        state = V.new_zeros(batch, heads, width, K.shape[-1])
        angle = V.new_zeros(batch, heads, Angles.shape[-1])
        last_k = V.new_zeros(batch, heads, K.shape[-1])
        last_v = V.new_zeros(batch, heads, width)
        outputs = []
        for t in range(length):
            dt = DT[:, :, t]
            trap = Trap[:, :, t].sigmoid()
            q = Q[:, t].expand(-1, heads, -1).unsqueeze(1)
            k = K[:, t].expand(-1, heads, -1).unsqueeze(1)
            q, k, angle = rotary_step(
                q,
                k,
                angle,
                Angles[:, t],
                dt,
                Q_bias.unsqueeze(0),
                K_bias.unsqueeze(0),
            )
            q, k = q[:, 0], k[:, 0]
            alpha = ADT[:, :, t].exp()
            beta = (1 - trap) * dt * alpha
            gamma = trap * dt
            state = (
                state * alpha[:, :, None, None]
                + beta[:, :, None, None] * last_v[:, :, :, None] * last_k[:, :, None, :]
                + gamma[:, :, None, None] * V[:, t, :, :, None] * k[:, :, None, :]
            )
            y = torch.einsum("bhps,bhs->bhp", state, q) + D[None, :, None] * V[:, t]
            outputs.append(y * F.silu(Z[:, t]) if Z is not None else y)
            last_k, last_v = k, V[:, t]
        result = torch.stack(outputs, dim=1)
        return (result, angle, state, last_k, last_v) if return_final_states else result

    namespace.update(
        apply_rotary_qk_inference_fwd=rotary_step,
        mamba3_step_fn=torch_step,
        mamba3_siso_combined=torch_combined,
    )
    extract(source / "mamba3.py", {"heavy_tail_activation", "Mamba3"}, namespace)
    extract(source / "mlp.py", {"GatedMLP"}, namespace)
    extract(source / "block.py", {"Block"}, namespace)
    return namespace


class RecurrentCache:
    def __init__(self):
        self.key_value_memory_dict = {}
        self.seqlen_offset = 0


class Mamba3Reference(nn.Module):
    def __init__(self, model_dir: Path, source_dir: Path):
        super().__init__()
        manifest = json.loads((model_dir / "verified-manifest.json").read_text())
        assert manifest["revision"] == REVISION
        weight_path = model_dir / "pytorch_model.bin"
        entry = next(x for x in manifest["files"] if x["file"] == weight_path.name)
        assert entry["size"] == 373744619 and entry["checksum"] == WEIGHT_SHA256
        with weight_path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA256
        config = json.loads((model_dir / "config.json").read_text())
        assert (config["d_model"], config["n_layer"], config["vocab_size"]) == (
            768,
            12,
            128256,
        )
        assert (
            config["ssm_cfg"]["layer"] == "Mamba3" and not config["ssm_cfg"]["is_mimo"]
        )
        assert (
            config["rms_norm"] and config["fused_add_norm"] and config["tie_embeddings"]
        )
        assert config["d_intermediate"] == 1536 and not config["attn_layer_idx"]
        functions = native_namespace(source_dir)
        native = functions["Mamba3"]

        class Mamba3CPU(native):
            def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
                if inference_params is not None and inference_params.seqlen_offset > 0:
                    assert u.shape[1] == 1 and cu_seqlens is None
                    states = self._get_states_from_cache(inference_params, u.shape[0])
                    out, *_ = self.step(u[:, 0], *states)
                    return out[:, None]
                return super().forward(u, seq_idx, cu_seqlens, inference_params)

        with torch.device("meta"):
            self.backbone = nn.Module()
            self.backbone.embedding = nn.Embedding(
                config["vocab_size"], config["d_model"]
            )
            self.backbone.layers = nn.ModuleList()
            for index in range(config["n_layer"]):
                mixer = partial(
                    Mamba3CPU,
                    layer_idx=index,
                    **{
                        key: value
                        for key, value in config["ssm_cfg"].items()
                        if key != "layer"
                    },
                )
                mlp = partial(
                    functions["GatedMLP"],
                    hidden_features=config["d_intermediate"],
                    out_features=config["d_model"],
                )
                self.backbone.layers.append(
                    functions["Block"](
                        config["d_model"],
                        mixer,
                        mlp,
                        norm_cls=RMSNormTorch,
                        fused_add_norm=True,
                        residual_in_fp32=True,
                    )
                )
            self.backbone.norm_f = RMSNormTorch(config["d_model"])
            self.lm_head = nn.Linear(
                config["d_model"], config["vocab_size"], bias=False
            )
        state = torch.load(
            weight_path, map_location="cpu", weights_only=True, mmap=True
        )
        assert len(state) == 147 and set(state) == set(self.state_dict())
        assert torch.equal(state["lm_head.weight"], state["backbone.embedding.weight"])
        weights = {name: value.float() for name, value in state.items()}
        self.load_state_dict(weights, strict=True, assign=True)
        self.lm_head.weight = self.backbone.embedding.weight
        assert sum(p.numel() for p in self.parameters()) == 186849600
        self.eval()
        self.provenance = {
            "checkpoint": manifest,
            "config_sha256": hashlib.sha256(
                (model_dir / "config.json").read_bytes()
            ).hexdigest(),
            "source_sha256": SOURCE_SHA256,
            "checkpoint_tensors": len(state),
            "effective_tied_parameters": 186849600,
            "layers": len(self.backbone.layers),
            "dtype": "torch.float32",
            "native_cuda_qualified": False,
        }

    def forward(self, input_ids: torch.Tensor, cache: RecurrentCache | None = None):
        if cache is None:
            cache = RecurrentCache()
        hidden = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.backbone.layers:
            hidden, residual = layer(hidden, residual, inference_params=cache)
        hidden = layer_norm_torch(
            hidden, self.backbone.norm_f.weight, None, residual=residual, prenorm=False
        )
        cache.seqlen_offset += input_ids.shape[1]
        return self.lm_head(hidden), cache
