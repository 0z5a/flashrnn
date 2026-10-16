"""Explicit recurrent-weight packing; all gates share a coordinate owner."""

import torch


def pack_recurrent(weight: torch.Tensor) -> torch.Tensor:
    gates, heads, output, reduction = weight.shape
    return (
        weight.permute(1, 3, 0, 2).contiguous().view(heads, reduction, gates * output)
    )


def unpack_recurrent(packed: torch.Tensor, gates: int) -> torch.Tensor:
    heads, reduction, columns = packed.shape
    if columns % gates:
        raise ValueError("packed width must contain complete gate bundles")
    return (
        packed.view(heads, reduction, gates, columns // gates)
        .permute(2, 0, 3, 1)
        .contiguous()
    )


def coordinate_owners(width: int, tile: int) -> tuple[tuple[int, int], ...]:
    if width < 1 or tile < 1:
        raise ValueError("width and output tile must be positive")
    return tuple((start, min(start + tile, width)) for start in range(0, width, tile))
