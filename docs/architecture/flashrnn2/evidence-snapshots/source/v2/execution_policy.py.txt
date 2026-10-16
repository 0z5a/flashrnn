"""Explicit, offline-selected configurations for inference execution.

No request-time benchmarking or unqualified automatic SIMT selection occurs.
Tables must be bound to a GPU/compiler identity by their producing experiment.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KernelConfig:
    family: str = "mma"
    batch_tile: int = 16
    warps: int = 4
    weight_operand: str = "b"

    def __post_init__(self):
        if self.family not in ("mma", "simt"):
            raise ValueError("unknown kernel family")
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


def shape_key(cell, dtype, batch, steps, heads, width):
    return f"{cell}/{dtype}/b{batch}/t{steps}/h{heads}/d{width}"


class ExecutionPolicy:
    """Exact-shape table with a conservative existing-kernel fallback."""

    def __init__(self, table=None, *, identity=None):
        self.identity = dict(identity or {})
        self.table = {
            key: value if isinstance(value, KernelConfig) else KernelConfig(**value)
            for key, value in (table or {}).items()
        }

    def select(self, cell, dtype, batch, steps, heads, width, *, identity=None):
        if self.table and (not self.identity or identity != self.identity):
            raise ValueError("a measured table requires its matching execution identity")
        key = shape_key(cell, dtype, batch, steps, heads, width)
        return self.table.get(key, KernelConfig())
