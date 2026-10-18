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
    # TileLang only: load the next step's input projections into registers after each step's epilogue.
    prefetch: bool = False
    # TileLang grid only (large D): reg_k rows of every CTA's R slice live in registers, the rest in shared memory
    # (flashrnn2.tilelang_cluster.build_grid_hybrid / build_grid_hybrid2; S = 16, BM = 16; with eight warps the two
    # warpgroups split K and hold reg_k / 2 rows each).
    reg_k: int = 0
    # TileLang grid only (large D, wide batch tiles): h^T streamed in 64-row chunks instead of staged whole, so that
    # BM = 32 / 64 fits next to R (flashrnn2.tilelang_stream; one warpgroup, S = 16, reg_k a multiple of 64).
    stream_h: bool = False
    # CUDA 2-D split forward only (flashrnn2.hybrid_fw2d, max-width.md section 56): the head-concurrent grid.
    conc: bool = False

    def __post_init__(self):
        if self.family not in ("mma", "simt", "mma_rtiled", "tilelang", "cuda2d"):
            raise ValueError("unknown kernel family")
        if self.conc and self.family != "cuda2d":
            raise ValueError("the head-concurrent grid (conc) exists only on the CUDA 2-D forward")
        if self.family == "cuda2d" and (self.batch_tile not in (16, 32, 64) or self.warps != 8 or self.slice_width
                                        or self.sync != "none" or self.pad_k or self.weight_operand != "b"):
            raise ValueError("the CUDA 2-D forward uses 16/32/64-row batch tiles, eight warps and no R-tiling options")
        if self.prefetch and self.family != "tilelang":
            raise ValueError("input-projection prefetch exists only on the TileLang paths")
        if self.family == "tilelang":
            # k = D / slice_width CTAs, one warpgroup each (flashrnn2.tilelang_cluster): sync "cluster"
            # exchanges h through distributed shared memory with an mbarrier per step (k <= 16); sync "grid"
            # exchanges it through global memory under one cooperative grid barrier per step.
            if self.batch_tile not in (16, 32, 64, 128) or self.weight_operand != "b" or self.pad_k:
                raise ValueError("the TileLang R-tiled paths use 16/32/64/128-row batch tiles and weight operand b")
            # warps: 4 (one warpgroup) or, on the grid path only, 8 (two warpgroups per CTA). Two warpgroups on
            # the cluster path gave wrong, nondeterministic results (screen: 0/12 bitwise twins).
            if self.stream_h and (self.sync != "grid" or self.slice_width != 16 or self.warps != 4
                                  or self.batch_tile not in (32, 64) or self.prefetch or self.reg_k % 64):
                raise ValueError("streamed h (stream_h) is the TileLang grid path with S = 16, four warps, BM = 32/64")
            if self.warps == 8 and (self.sync != "grid" or (self.batch_tile < 32 and not self.reg_k)):
                raise ValueError("two warpgroups per CTA are qualified only on the TileLang grid path with BM >= 32")
            if not ((self.sync == "cluster" and self.slice_width in (16, 32, 64))
                    or (self.sync == "grid" and self.slice_width in (16, 32))):
                raise ValueError("TileLang needs cluster sync with a slice of 16/32/64 or grid sync with 16/32")
            if self.reg_k and not self.stream_h and (self.sync != "grid" or self.slice_width != 16 or self.batch_tile != 16
                               or self.prefetch or self.reg_k % (16 * self.warps)):
                raise ValueError("register residency (reg_k) is the TileLang grid path with S = 16, BM = 16 and reg_k a "
                                 "multiple of 64 per warpgroup (four warps: one warpgroup; eight: K split over two)")
        elif self.family == "mma_rtiled":
            if self.batch_tile != 16 or self.weight_operand != "b":
                raise ValueError("R tiling uses MMA16 batch tiles and weight operand b")
            if self.slice_width not in (16, 32, 64, 128) or self.sync != "grid":
                raise ValueError("R tiling needs a power-of-two slice in [16, 128] and grid sync")
        if (self.reg_k or self.stream_h) and self.family != "tilelang":
            raise ValueError("register residency (reg_k) exists only on the TileLang grid path")
        if self.family in ("mma", "simt") and (self.slice_width or self.sync != "none" or self.pad_k):
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


def _tilelang_available():
    import importlib.util
    return importlib.util.find_spec("tilelang") is not None


def default_config(width, batch=None, heads=1):
    """Conservative unmeasured default: single CTA up to D=128, else R tiling.

    The R-tiled default is spill-light at eight warps in the M0 compile sweep
    and is covered by the M0 protocol/E1 gates; it is not a tuned choice.
    """
    if width <= 128:
        return KernelConfig()
    if batch is not None and batch >= 64:
        # CUDA 2-D split forward where its paired timing held for both cells (max-width.md section 57): D = 2048, H = 1
        # (forward 1.42-1.46x, training step 1.16-1.17x at B = 64); D = 1024 with at least four 64-row groups
        # (heads x batch tiles) on the head-concurrent grid (forward 1.25-1.41x).
        if width == 2048 and heads == 1:
            return KernelConfig(family="cuda2d", warps=8, batch_tile=64)
        if width == 1024 and -(-batch // 64) * heads >= 4:
            return KernelConfig(family="cuda2d", warps=8, batch_tile=64, conc=True)
    if batch is not None and batch > 32 and 1280 <= width <= 1792 and width % 64 == 0:
        # Wide batch tiles with streamed h (job "bm64 v3": 1.6-1.9x over 16-row tiles at B = 64, D = 1280-1792).
        return wide_batch_config(width)
    if width > 1280:
        return large_width_config(width)
    if width >= 1024 and width % 16 == 0 and _tilelang_available():  # measured at D = 1024 and 1280
        # TileLang grid kernel, 16-column slices (max-width.md section 31: 4.8-5.5x faster than the Triton default
        # below at D = 1024, 2.8-2.9x at D = 1280, B = 16).
        return KernelConfig(family="tilelang", slice_width=16, sync="grid")
    # S = 32 needs about 290 KB of shared memory per CTA at D = 1024; above D = 768 only S = 16 fits (compile
    # records: 164 KB at D = 1024, 205 KB at D = 1280, 246 KB at D = 1536 against the 227 KB opt-in limit).
    slice_width = 64 if width <= 256 else 32 if width <= 768 else 16
    return KernelConfig(family="mma_rtiled", warps=8, slice_width=slice_width, sync="grid")


def wide_batch_config(width):
    """TileLang grid with 64-row batch tiles and h^T streamed in 64-row chunks (flashrnn2.tilelang_stream); reg_k is the
    fewest register rows for which shared memory fits (D = 1280: 0, 1536: 128, 1792: 384)."""
    smem = lambda r: 4 * 16 * (width - r) * 2 + 4 * 64 * 64 * 2 + 4 * 16 * 64 * 4  # noqa: E731 (tilelang_stream)
    reg_k = next(r for r in range(0, 1025, 64) if smem(r) <= 232448)
    return KernelConfig(family="tilelang", slice_width=16, sync="grid", batch_tile=64, stream_h=True, reg_k=reg_k)


def large_width_config(width):
    """D > 1280 (no Triton path fits): TileLang grid S = 16 up to D = 1424, register + shared-memory residency of R
    with one warpgroup up to D = 1984, two warpgroups splitting K at D = 2048 (flashrnn2.tilelang_cluster)."""
    if width % 16 or width > 2048:
        raise ValueError("no FlashRNN2 path for this width")
    if width <= 1424:
        return KernelConfig(family="tilelang", slice_width=16, sync="grid")
    if width <= 1984:
        smem = lambda r: 4 * 16 * (width - r) * 2 + width * 16 * 2 + 4 * 16 * 16 * 4  # noqa: E731
        reg_k = next(r for r in range(64, 769, 64) if smem(r) <= 232448)
        return KernelConfig(family="tilelang", slice_width=16, sync="grid", reg_k=reg_k)
    return KernelConfig(family="tilelang", slice_width=16, sync="grid", warps=8, reg_k=896)


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
    def from_json(cls, *paths):
        """Load measured table files (identity + exact-shape entries), e.g. TABLES / "h100-rtiled-m6-20261014.json".

        Several files merge only if they share one identity and no shape key.
        """
        table, identity = {}, None
        for path in paths:
            data = json.loads(Path(path).read_text())
            if data.get("format") != "flashrnn2-execution-policy-table/1":
                raise ValueError("unknown execution-policy table format")
            if identity is not None and data["identity"] != identity:
                raise ValueError("tables measured under different identities cannot be merged")
            if table.keys() & data["table"].keys():
                raise ValueError("tables overlap in exact shape keys")
            identity = data["identity"]
            table.update(data["table"])
        if identity is None:
            raise ValueError("no table file given")
        return cls(table, identity=identity)

    def select(self, cell, dtype, batch, steps, heads, width, *, identity=None):
        if self.table and (not self.identity or identity != self.identity):
            raise ValueError("a measured table requires its matching execution identity")
        key = shape_key(cell, dtype, batch, steps, heads, width)
        config = self.table.get(key, default_config(width, batch, heads))
        if config.family == "tilelang" and not (identity or {}).get("tilelang"):
            # A table entry is chosen before the call; it never falls back to another backend.
            raise ValueError("the table selects the TileLang backend but this runtime has no TileLang")
        return config
