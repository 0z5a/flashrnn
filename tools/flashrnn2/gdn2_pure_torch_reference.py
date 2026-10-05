"""CPU reference for the pinned pure 12-layer GDN-2 checkpoint."""

import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WEIGHT_SHA256 = "48af211d5cbebb8a26bf1c7024e897ccab65bb4f8204b278ce94ff0f4cc7735d"


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight


class RMSNormSwishGate(RMSNorm):
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * F.silu(gate)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(1024, 4352, bias=False)
        self.w2 = nn.Linear(1024, 4352, bias=False)
        self.w3 = nn.Linear(4352, 1024, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LLaMAMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.swiglu = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class GatedDeltaNet2(nn.Module):
    def __init__(self, recurrent) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.q_proj = nn.Linear(1024, 1024, bias=False)
        self.k_proj = nn.Linear(1024, 1024, bias=False)
        self.v_proj = nn.Linear(1024, 1024, bias=False)
        self.q_conv1d = nn.Conv1d(1024, 1024, 4, groups=1024, bias=False)
        self.k_conv1d = nn.Conv1d(1024, 1024, 4, groups=1024, bias=False)
        self.v_conv1d = nn.Conv1d(1024, 1024, 4, groups=1024, bias=False)
        self.f_proj = nn.Sequential(
            nn.Linear(1024, 64, bias=False), nn.Linear(64, 1024, bias=False)
        )
        self.b_proj = nn.Linear(1024, 1024, bias=False)
        self.w_proj = nn.Linear(1024, 1024, bias=False)
        self.A_log = nn.Parameter(torch.zeros(16))
        self.dt_bias = nn.Parameter(torch.zeros(1024))
        self.g_proj = nn.Sequential(
            nn.Linear(1024, 64, bias=False), nn.Linear(64, 1024)
        )
        self.o_norm = RMSNormSwishGate(64)
        self.o_proj = nn.Linear(1024, 1024, bias=False)

    @staticmethod
    def conv(x: torch.Tensor, layer: nn.Conv1d, history: torch.Tensor | None):
        signal = x.transpose(1, 2)
        if history is None:
            signal = F.pad(signal, (3, 0))
        else:
            signal = torch.cat((history, signal), dim=-1)
        output = F.conv1d(signal, layer.weight, groups=1024).transpose(1, 2)
        return F.silu(output), signal[:, :, -3:]

    def forward(self, x: torch.Tensor, state=None):
        batch, steps, _ = x.shape
        recurrent_state = None if state is None else state[0]
        conv_states = (None, None, None) if state is None else state[1]
        q, cq = self.conv(self.q_proj(x), self.q_conv1d, conv_states[0])
        k, ck = self.conv(self.k_proj(x), self.k_conv1d, conv_states[1])
        v, cv = self.conv(self.v_proj(x), self.v_conv1d, conv_states[2])
        q = q.reshape(batch, steps, 16, 64)
        k = k.reshape(batch, steps, 16, 64)
        v = v.reshape(batch, steps, 16, 64)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = -self.A_log.float().exp()[None, None, :, None] * F.softplus(
            self.f_proj(x).float().reshape(batch, steps, 16, 64)
            + self.dt_bias.reshape(1, 1, 16, 64)
        )
        erase = self.b_proj(x).sigmoid().reshape(batch, steps, 16, 64)
        write = self.w_proj(x).sigmoid().reshape(batch, steps, 16, 64)
        output, recurrent_state = self.recurrent(
            q,
            k,
            v,
            decay,
            erase,
            write,
            initial_state=recurrent_state,
            output_final_state=True,
        )
        gate = self.g_proj(x).reshape(batch, steps, 16, 64)
        output = self.o_norm(output, gate).reshape(batch, steps, 1024)
        return self.o_proj(output), (recurrent_state, (cq, ck, cv))


class Block(nn.Module):
    def __init__(self, recurrent) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(1024)
        self.attn = GatedDeltaNet2(recurrent)
        self.norm_2 = RMSNorm(1024)
        self.mlp = LLaMAMLP()

    def forward(self, x: torch.Tensor, state=None):
        attention, state = self.attn(self.norm_1(x), state)
        x = x + attention
        return x + self.mlp(self.norm_2(x)), state


class PureGDN2LM(nn.Module):
    def __init__(self, recurrent) -> None:
        super().__init__()
        self.lm_head = nn.Linear(1024, 32000, bias=False)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(32000, 1024),
                "h": nn.ModuleList([Block(recurrent) for _ in range(12)]),
                "ln_f": RMSNorm(1024),
            }
        )

    def forward(self, ids: torch.Tensor, states=None):
        x = self.transformer["wte"](ids)
        states = [None] * 12 if states is None else states
        next_states = []
        for block, state in zip(self.transformer["h"], states, strict=True):
            x, next_state = block(x, state)
            next_states.append(next_state)
        logits = self.lm_head(self.transformer["ln_f"](x))
        return logits, next_states


def load_reference(checkpoint: Path, fla_source: Path):
    with checkpoint.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_SHA256
    source = fla_source / "fla/ops/gdn2/naive.py"
    spec = importlib.util.spec_from_file_location("pinned_fla_gdn2_naive_pure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.device("meta"):
        model = PureGDN2LM(module.naive_recurrent_gdn2)
    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    assert set(checkpoint_data) == {
        "model",
        "optimizer",
        "hparams",
        "iter_num",
        "step_count",
    }
    weights = checkpoint_data["model"]
    assert len(weights) == 267 and set(weights) == set(model.state_dict())
    model.load_state_dict(weights, strict=True, assign=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 304809920
    return model.eval()
