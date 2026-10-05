"""Pinned native Mamba Torch fallback using resident, audited checkpoint weights.

Forward/cache arithmetic is selected unchanged from Transformers source. Only
module construction and tuple-only inference transport are supplied here.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from mamba_script_gate import verified_metadata
from torch import nn

SOURCE_SHA = "ad28b5f1a464a64d1eabf96b42394f41ef322dbe5f4276e85d4f27368b61a796"
CONFIG_BLOB = "65cacc293f83d920b3a79690ce835426950f8b4d"
Output = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def native_functions(path: Path) -> dict:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA
    classes = {
        node.name: node
        for node in ast.parse(data).body
        if isinstance(node, ast.ClassDef)
    }
    selected = ast.parse("from __future__ import annotations\n").body
    selected += [classes["MambaCache"], classes["MambaRMSNorm"]]
    for cls, method, name in (
        ("MambaMixer", "slow_forward", "mixer_forward"),
        ("MambaBlock", "forward", "block_forward"),
        ("MambaModel", "forward", "backbone_forward"),
        ("MambaForCausalLM", "forward", "lm_forward"),
    ):
        node = next(
            node
            for node in classes[cls].body
            if isinstance(node, ast.FunctionDef) and node.name == method
        )
        node.name = name
        node.decorator_list = []
        selected.append(node)
    namespace = {"torch": torch, "nn": nn, "CrossEntropyLoss": nn.CrossEntropyLoss}
    exec(  # noqa: S102 - immutable source hash checked; selected native math only
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


class NativeMixer(nn.Module):
    def __init__(self, config: SimpleNamespace, index: int, functions: dict) -> None:
        super().__init__()
        self.layer_idx = index
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.time_step_rank = config.time_step_rank
        self.use_conv_bias = config.use_conv_bias
        self.use_mambapy = False
        self.act = nn.SiLU()
        self.conv1d = nn.Conv1d(
            config.intermediate_size,
            config.intermediate_size,
            config.conv_kernel,
            groups=config.intermediate_size,
            padding=config.conv_kernel - 1,
            bias=config.use_conv_bias,
        )
        self.in_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=config.use_bias
        )
        self.x_proj = nn.Linear(
            config.intermediate_size,
            config.time_step_rank + 2 * config.state_size,
            bias=False,
        )
        self.dt_proj = nn.Linear(config.time_step_rank, config.intermediate_size)
        self.out_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.use_bias
        )
        self.A_log = nn.Parameter(
            torch.empty(config.intermediate_size, config.state_size)
        )
        self.D = nn.Parameter(torch.empty(config.intermediate_size))
        self.forward = MethodType(functions["mixer_forward"], self)


class NativeBlock(nn.Module):
    def __init__(self, config: SimpleNamespace, index: int, functions: dict) -> None:
        super().__init__()
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = functions["MambaRMSNorm"](
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.mixer = NativeMixer(config, index, functions)
        self.forward = MethodType(functions["block_forward"], self)


class NativeBackbone(nn.Module):
    def __init__(self, config: SimpleNamespace, functions: dict) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            NativeBlock(config, index, functions)
            for index in range(config.num_hidden_layers)
        )
        self.norm_f = functions["MambaRMSNorm"](
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.forward = MethodType(functions["backbone_forward"], self)


class NativeMamba(nn.Module):
    def __init__(
        self, artifact: Path, source: Path, config_path: Path, device: str
    ) -> None:
        super().__init__()
        self.metadata = verified_metadata(artifact)
        raw = config_path.read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == CONFIG_BLOB
        )
        config = json.loads(raw)
        assert config["model_type"] == "mamba" and config["hidden_act"] == "silu"
        config.update(
            use_mambapy=False, use_return_dict=False, output_hidden_states=False
        )
        self.config = SimpleNamespace(**config)
        functions = native_functions(source)
        self.cache_class = functions["MambaCache"]
        with torch.device("meta"):
            self.backbone = NativeBackbone(self.config, functions)
            self.lm_head = nn.Linear(
                self.config.hidden_size, self.config.vocab_size, bias=False
            )
        transport = torch.jit.load(str(artifact), map_location=device).eval()
        weights = transport.state_dict()
        assert all(key.startswith("model.") for key in weights)
        self.load_state_dict(
            {key.removeprefix("model."): value for key, value in weights.items()},
            strict=True,
            assign=True,
        )
        assert (
            self.lm_head.weight.data_ptr() == self.backbone.embeddings.weight.data_ptr()
        )
        self.lm_head.weight = self.backbone.embeddings.weight
        assert (
            sum(p.numel() for p in self.parameters())
            == self.metadata["parameters"]
            == 129135360
        )
        self.forward = MethodType(functions["lm_forward"], self)
        self.eval()

    def run(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor, prefill: bool
    ) -> Output:
        cache = self.cache_class(
            self.config,
            max_batch_size=ids.shape[0],
            dtype=conv.dtype,
            device=ids.device,
        )
        cache.conv_states = list(conv.unbind(0))
        cache.ssm_states = list(ssm.unbind(0))
        position = (
            torch.arange(self.config.conv_kernel, device=ids.device)
            if prefill
            else torch.full((1,), self.config.conv_kernel, device=ids.device)
        )
        output = self(
            ids,
            cache_params=cache,
            cache_position=position,
            use_cache=True,
            return_dict=False,
        )
        return output[0], torch.stack(cache.conv_states), torch.stack(cache.ssm_states)

    def prefill(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor
    ) -> Output:
        return self.run(ids, conv, ssm, True)

    def decode(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor
    ) -> Output:
        return self.run(ids, conv, ssm, False)
