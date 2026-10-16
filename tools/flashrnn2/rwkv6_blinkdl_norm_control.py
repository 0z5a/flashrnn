"""Repeat the BlinkDL comparison with only norm eps matched to the checkpoint."""

import hashlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import rwkv6_blinkdl_gate as gate
import torch
from rwkv6_blinkdl_reference import blinkdl_reference
from torch import nn
from torch.nn import functional as F


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert eps == 64e-5
    return F.group_norm(x, num_groups, weight, bias, eps=1e-6)


def matched_reference(weights: dict[str, torch.Tensor], source: Path) -> nn.Module:
    model = blinkdl_reference(weights, source)
    namespace = type(model).time_mixing.__globals__
    assert namespace["F"] is F
    namespace["F"] = SimpleNamespace(
        layer_norm=partial(F.layer_norm, eps=1e-6),
        group_norm=group_norm,
        silu=F.silu,
    )
    model.provenance.update(
        scope="BLINKDL_FORWARD_WITH_CHANGED_NORM_EPS_DIAGNOSTIC",
        layer_norm_eps=1e-6,
        group_norm_eps=1e-6,
        original_blinkdl_norms_qualified=False,
        norm_control_source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    )
    return model


if __name__ == "__main__":
    gate.blinkdl_reference = matched_reference
    gate.main()
