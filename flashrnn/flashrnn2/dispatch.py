"""One-call forward recurrence that serves the exact-shape execution table.

``select_config`` picks a ``KernelConfig`` for one logical call; ``recurrence`` runs one call (or one chunk of a
chunked logical call) on the backend that configuration names, with the same contract as the backends themselves:
FP16/BF16 inputs, full ([S,B,T,H,D]) / hidden ([B,T,H,D]) / last ([B,1,H,D]) output, FP32 compute carry, chunk
continuation through a ``SequenceContext``, and the sLSTM normalizer rule applied at the logical call's first step.

Table keys use the logical call's total steps, so every chunk of a chunked call must run the configuration selected
for the whole call: pass it as ``config``. A continuation chunk without one is rejected rather than re-selected.

The default policy is the packaged H100 table set (M6 + M7 + the FP16/BF16 grid fill), served only when this runtime's identity (GPU name,
capability, Triton and TileLang versions) equals the identity the tables were measured under; otherwise every shape
gets ``default_config``. A table entry that names TileLang on a runtime without TileLang raises; nothing falls back
to another backend silently. Calls that require gradients go to ``training.recurrence`` (whole logical
calls; LSTM/sLSTM; R-tiled Triton backward).
"""
import functools

import torch

from .execution_policy import TABLES, ExecutionPolicy, KernelConfig, runtime_identity, shape_key
from .state_contract import begin_sequence

TABLE_FILES = ("h100-rtiled-m6-20261014.json", "h100-rtiled-m7-20261014.json", "h100-rtiled-grid-20261015.json")
DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


@functools.lru_cache(maxsize=None)
def _packaged():
    return ExecutionPolicy.from_json(*(TABLES / name for name in TABLE_FILES))


def policy_for(identity):
    """The packaged tables when ``identity`` equals theirs, else an empty policy (defaults only)."""
    tables = _packaged()
    return tables if identity == tables.identity else ExecutionPolicy()


@functools.lru_cache(maxsize=None)
def default_policy():
    return policy_for(runtime_identity())


def select_config(cell, dtype, batch, steps, heads, width, *, policy=None):
    """KernelConfig for one logical call and where it came from ("table" or "default")."""
    policy = default_policy() if policy is None else policy
    name = DTYPES.get(dtype, dtype)
    identity = runtime_identity() if policy.table else None
    config = policy.select(cell, name, batch, steps, heads, width, identity=identity)
    source = "table" if shape_key(cell, name, batch, steps, heads, width) in policy.table else "default"
    return config, source


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, sequence_context=None, output_mode="full",
               snapshot_dtype=None, config=None, policy=None):
    """Run one logical call, or one chunk of it, on the backend the configuration selects.

    Returns (output, FP32 compute carry). Without ``sequence_context`` the call is a whole logical call.
    """
    if wx.ndim != 5:
        raise ValueError("expected Wx [B,T,4,H,D]")
    if wx.dtype not in DTYPES:
        raise ValueError("the dispatched recurrence takes FP16/BF16 inputs")
    batch, steps, _, heads, width = wx.shape
    if torch.is_grad_enabled() and any(x.requires_grad for x in (wx, recurrent, bias, initial)):
        # Training: whole logical calls through the differentiable path (forward on the selected backend).
        from . import training

        return training.recurrence(wx, recurrent, bias, initial, cell, output_mode=output_mode,
                                   snapshot_dtype=snapshot_dtype, config=config, sequence_context=sequence_context)
    if sequence_context is None:
        sequence_context = begin_sequence(initial, cell, wx.dtype)
    if config is None:
        if sequence_context.step_offset > 0:
            raise ValueError("a continuation chunk must pass the configuration selected for its logical call")
        config, _ = select_config(cell, wx.dtype, batch, steps, heads, width, policy=policy)
    if not isinstance(config, KernelConfig):
        config = KernelConfig(**config)
    if config.family == "tilelang":
        try:
            from . import tilelang_cluster
        except ImportError as error:  # chosen before the call: no silent switch to another backend
            raise RuntimeError("the selected TileLang backend is not installed") from error
        return tilelang_cluster.recurrence(
            wx, recurrent, bias, initial, cell, sequence_context=sequence_context, slice_width=config.slice_width,
            output_mode=output_mode, snapshot_dtype=snapshot_dtype, sync=config.sync, batch_tile=config.batch_tile,
            threads=32 * config.warps, prefetch=config.prefetch, reg_k=config.reg_k)
    if config.family == "mma_rtiled":
        from . import triton_rtiled
        return triton_rtiled.recurrence(
            wx, recurrent, bias, initial, cell, sequence_context=sequence_context, slice_width=config.slice_width,
            num_warps=config.warps, pad_k=config.pad_k, output_mode=output_mode, snapshot_dtype=snapshot_dtype)
    if config.family == "mma" and config.batch_tile == 16:
        from . import triton_persistent
        return triton_persistent.recurrence(
            wx, recurrent, bias, initial, cell, snapshot_dtype=snapshot_dtype, weight_operand=config.weight_operand,
            return_compute_carry=True, sequence_context=sequence_context, output_mode=output_mode,
            num_warps=config.warps)
    raise ValueError(f"configuration {config.as_dict()} has no one-call path; use PreparedSequence")
