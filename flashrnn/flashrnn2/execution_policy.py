"""Explicit, offline-selected configurations for inference execution.

No request-time benchmarking or unqualified automatic SIMT selection occurs.
Tables must be bound to a GPU/compiler identity by their producing experiment.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelConfig:
    family: str = "mma"
    batch_tile: int = 16
    warps: int = 4
    weight_operand: str = "b"
    # R tiling across CTAs (family "mma_rtiled"): power-of-two output slice per
    # CTA, k = ceil(D / slice_width) CTAs per (batch tile, head), and the
    # per-step synchronization between them.
    slice_width: int = 0
    sync: str = "none"
    pad_k: bool = False

    def __post_init__(self):
        if self.family not in ("mma", "simt", "mma_rtiled", "tilelang"):
            raise ValueError("unknown kernel family")
        if self.family == "tilelang":
            # Thread-block cluster of k = D / slice_width CTAs, one warpgroup each, h exchanged through
            # distributed shared memory with an mbarrier per step (flashrnn2.tilelang_cluster).
            if self.batch_tile != 16 or self.weight_operand != "b" or self.warps != 4 or self.pad_k:
                raise ValueError("the TileLang cluster path uses 16-row batch tiles and one warpgroup")
            if self.slice_width not in (16, 32, 64) or self.sync != "cluster":
                raise ValueError("the TileLang cluster path needs a slice of 16, 32 or 64 and cluster sync")
        elif self.family == "mma_rtiled":
            if self.batch_tile != 16 or self.weight_operand != "b":
                raise ValueError("R tiling uses MMA16 batch tiles and weight operand b")
            if self.slice_width not in (16, 32, 64, 128) or self.sync != "grid":
                raise ValueError("R tiling needs a power-of-two slice in [16, 128] and grid sync")
        elif self.slice_width or self.sync != "none" or self.pad_k:
            raise ValueError("slice width, sync and K padding apply only to R tiling")
        if self.warps not in (4, 8):
            raise ValueError("supported warp counts are 4 and 8")
        if self.family == "mma" and self.batch_tile not in (16, 32):
            raise ValueError("MMA batch tiles must be 16 or 32")
        if self.family == "simt" and (self.batch_tile != 1 or self.weight_operand != "b"):
            raise ValueError("SIMT uses one batch row per CTA and weight operand b")
        if self.weight_operand not in ("a", "b"):
            raise ValueError("unknown weight operand")

    def as_dict(self):
        return asdict(self)


def default_config(width):
    """Conservative unmeasured default: single CTA up to D=128, else R tiling.

    The R-tiled default is spill-light at eight warps in the M0 compile sweep
    and is covered by the M0 protocol/E1 gates; it is not a tuned choice.
    """
    if width <= 128:
        return KernelConfig()
    return KernelConfig(family="mma_rtiled", warps=8, slice_width=64 if width <= 256 else 32, sync="grid")


def runtime_identity():
    """GPU and compiler identity a measured table is bound to; TileLang is None when not installed."""
    import torch
    import triton

    try:
        import tilelang
        tilelang_version = tilelang.__version__
    except ImportError:
        tilelang_version = None
    return dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                triton=triton.__version__, tilelang=tilelang_version)


def shape_key(cell, dtype, batch, steps, heads, width):
    return f"{cell}/{dtype}/b{batch}/t{steps}/h{heads}/d{width}"


TABLES = Path(__file__).resolve().parent / "tables"


class ExecutionPolicy:
    """Exact-shape table with a conservative existing-kernel fallback."""

    def __init__(self, table=None, *, identity=None):
        self.identity = dict(identity or {})
        self.table = {
            key: value if isinstance(value, KernelConfig) else KernelConfig(**value)
            for key, value in (table or {}).items()
        }

    @classmethod
    def from_json(cls, path):
        """Load a measured table file (identity + exact-shape entries), e.g. TABLES / "h100-rtiled-m6-20261014.json"."""
        data = json.loads(Path(path).read_text())
        if data.get("format") != "flashrnn2-execution-policy-table/1":
            raise ValueError("unknown execution-policy table format")
        return cls(data["table"], identity=data["identity"])

    def select(self, cell, dtype, batch, steps, heads, width, *, identity=None):
        if self.table and (not self.identity or identity != self.identity):
            raise ValueError("a measured table requires its matching execution identity")
        key = shape_key(cell, dtype, batch, steps, heads, width)
        config = self.table.get(key, default_config(width))
        if config.family == "tilelang" and not (identity or {}).get("tilelang"):
            # A table entry is chosen before the call; it never falls back to another backend.
            raise ValueError("the table selects the TileLang backend but this runtime has no TileLang")
        return config
