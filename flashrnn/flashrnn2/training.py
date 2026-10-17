"""Differentiable FlashRNN2 recurrence for LSTM/sLSTM training (whole logical calls).

``recurrence`` runs the forward on the backend that ``config`` (or the packaged table) selects, in full output mode
with an FP32 state snapshot, saves that history, and returns the requested output and the FP32 final carry. Its
backward is ``triton_rtiled_bw.backward``: gradients follow ``grad_reference`` (no stabilizer gradient, straight-
through normalizer clamp, as FlashRNN v1). Gradients are returned in the dtypes of the inputs.

Chunk continuation is inference only: a call with a ``sequence_context`` past the first step cannot be
differentiated here.
"""

import importlib.util

import torch

# Defaults from the B = 16 fix (training-step timing against FlashRNN1): the fast backward variant (bitwise identical
# to the base kernel) and TF32 for the preactivation/dR GEMMs (gradient error unchanged: FP16 3.2e-4, BF16 2.4e-3
# relative Frobenius against the FP64 oracle); eight warps, the qualified slice width per D.
BW_DEFAULTS = {"slice_width": None, "num_warps": 8, "fast": True, "tf32": True}
# Gate-batched dh reduction (SPLITG) where the pre-registered rule of the SPLITG batch held: faster than the previous
# best candidates at B = 16 (TileLang forward, lower 95% bound >= 1 for LSTM and sLSTM) at every D = 192-768; for every
# batch size where it was also not slower at B = 64 (D = 192/256/384), otherwise only for B <= 16 (D = 512/768). The
# configuration is the one the SPLITG role selected (TileLang forward, B = 16). It replaces the fused default at D = 256.
SPLITG_DEFAULTS = {192: ({"split_gates": True, "slice_width": 16, "num_warps": 4}, None),
                   256: ({"split_gates": True, "slice_width": 16, "num_warps": 4}, None),
                   384: ({"split_gates": True, "slice_width": 16, "num_warps": 8}, None),
                   512: ({"split_gates": True, "slice_width": 16, "num_warps": 8}, 16),
                   768: ({"split_gates": True, "slice_width": 16, "num_warps": 8}, 16)}


def backward_defaults(width, user=None, batch=None):
    """Backward options for head dimension ``width`` and batch size ``batch``: package defaults, the SPLITG defaults
    where they apply, then the caller's."""
    user = dict(user or {})
    options = dict(BW_DEFAULTS)
    split, max_batch = SPLITG_DEFAULTS.get(width, ({}, None))
    if split and (max_batch is None or (batch is not None and batch <= max_batch)):
        options.update(split)
    options.update(user)
    if not options.get("fast"):  # fused and gate-batched kernels are variants of the fast kernel
        if "fuse_pre" not in user:
            options["fuse_pre"] = 0
        if "split_gates" not in user:
            options["split_gates"] = False
    return options


# Forward for training calls whose shape is not in the packaged tables: the TileLang configuration the gate-batched
# batch selected at B = 16 (H = 1), used when TileLang is installed and the runtime matches the tables' measured
# identity; otherwise the table default (Triton). With the Triton forward the B = 16 training step stays behind
# FlashRNN1 (0.88-1.04x), with these TileLang forwards it is faster at every D = 192-768.
TILELANG_TRAINING_FORWARD = {192: ("cluster", 16), 256: ("cluster", 16), 384: ("cluster", 32), 512: ("cluster", 32),
                             768: ("grid", 16)}


def _tilelang_installed():
    return importlib.util.find_spec("tilelang") is not None


def training_forward(cell, dtype, batch, steps, heads, width, *, policy=None):
    """KernelConfig of a training call's forward and where it came from ("table", "tilelang" or "default")."""
    from . import dispatch
    from .execution_policy import KernelConfig

    config, source = dispatch.select_config(cell, dtype, batch, steps, heads, width, policy=policy)
    if source == "table":
        return config, source
    measured = (dispatch.default_policy() if policy is None else policy).table
    if measured and heads == 1 and width in TILELANG_TRAINING_FORWARD and _tilelang_installed():
        sync, slice_width = TILELANG_TRAINING_FORWARD[width]
        return KernelConfig(family="tilelang", slice_width=slice_width, sync=sync), "tilelang"
    return config, source


HYBRID_BACKWARD_ABOVE = 1280  # widest D the Triton backward fits (S = 16, 224 KB of shared memory)
FUSED_HEADS_MAX_WIDTH = 256  # largest D at which the H > 1 fused-recompute default is qualified (heads v4)


def backward_slice_width(width):
    """Qualified backward slice width per D (eight warps; shared memory fits, spill-light)."""
    if width <= 128:
        return 32
    if width <= 512:
        return 16
    if width <= 768:
        return 32
    return 16  # S = 32 exceeds shared memory above D = 768 (328 KB at D = 1024); S = 16 fits up to D = 1280


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wx, recurrent, bias, initial, cell, output_mode, snapshot_dtype, config, backward_options):
        from . import dispatch

        history, carry = dispatch.recurrence(wx, recurrent, bias, initial, cell, output_mode="full",
                                             snapshot_dtype=torch.float32, config=config)
        ctx.save_for_backward(wx, recurrent, bias, initial, history)
        ctx.cell, ctx.output_mode, ctx.backward_options = cell, output_mode, backward_options
        dtype = wx.dtype if snapshot_dtype is None else snapshot_dtype
        if output_mode == "full":
            output = history.to(dtype)
        elif output_mode == "hidden":
            output = history[0].to(dtype)
        else:
            output = history[0][:, -1:].to(dtype)
        return output, carry

    @staticmethod
    def backward(ctx, d_output, d_carry):
        from . import triton_rtiled_bw

        wx, recurrent, bias, initial, history = ctx.saved_tensors
        d_history = torch.zeros_like(history)
        if d_output is not None:
            if ctx.output_mode == "full":
                d_history += d_output.float()
            elif ctx.output_mode == "hidden":
                d_history[0] += d_output.float()
            else:
                d_history[0][:, -1:] += d_output.float()
        d_final = torch.zeros(initial.shape, device=wx.device, dtype=torch.float32)
        if d_carry is not None:
            d_final += d_carry.float()
        options = dict(ctx.backward_options)
        backend = options.pop("backend", None)
        if backend == "hybrid2d":
            # R split over column blocks and K blocks (max-width.md section 20); opt-in.
            from . import hybrid_bw2d

            dg, d_recurrent, d_bias, d_initial = hybrid_bw2d.backward(
                wx, recurrent, bias, initial, history, d_history, d_final, ctx.cell,
                batch_rows=options.get("batch_rows"), wgrad=options.get("wgrad", "bmm"),
                prof=options.get("prof_phases", False), tf32=options.get("tf32", False),
                diagnostics=options.get("diagnostics"))
            return (dg.to(wx.dtype), d_recurrent.to(recurrent.dtype), d_bias.to(bias.dtype),
                    d_initial.to(initial.dtype), None, None, None, None, None)
        if backend == "hybrid" or (backend is None and wx.shape[-1] > HYBRID_BACKWARD_ABOVE):
            # D > 1280: R does not fit the Triton kernel's shared memory; the CUDA kernel keeps part of it in registers.
            from . import hybrid_bw

            dg, d_recurrent, d_bias, d_initial = hybrid_bw.backward(
                wx, recurrent, bias, initial, history, d_history, d_final, ctx.cell,
                reg_rows_per_warp=options.get("reg_rows_per_warp"), batch_rows=options.get("batch_rows"),
                wgrad=options.get("wgrad", "bmm"), prof=options.get("prof_phases", False),
                cluster=options.get("cluster", 0),
                tf32=options.get("tf32", False),
                diagnostics=options.get("diagnostics"))
            return (dg.to(wx.dtype), d_recurrent.to(recurrent.dtype), d_bias.to(bias.dtype),
                    d_initial.to(initial.dtype), None, None, None, None, None)
        if options.get("fast") and "history_grad_components" not in options:
            full = ctx.output_mode == "full" and d_output is not None
            options["history_grad_components"] = (True, full, full)
        if options.get("slice_width") is None:
            options["slice_width"] = backward_slice_width(wx.shape[-1])
        legacy_heads = options.pop("legacy_heads", False)
        if wx.shape[3] > 1 and not legacy_heads and wx.shape[-1] <= FUSED_HEADS_MAX_WIDTH:
            # H > 1: recompute the gate preactivations inside the kernel (no [B, T, 4, H, D] FP32 preactivation
            # tensor and its permuted copies; heads profiling, docs/flashrnn2/heads.md section 8). Above D = 256 the
            # fused variant does not fit shared memory (heads v4: D = 768 needs 320-416 KB), so the default stays.
            options.setdefault("fast", True)
            options.setdefault("fuse_pre", 2 if options["fast"] else 0)
        dg, d_recurrent, d_bias, d_initial = triton_rtiled_bw.backward(
            wx, recurrent, bias, initial, history, d_history, d_final, ctx.cell, **options)
        return (dg.to(wx.dtype), d_recurrent.to(recurrent.dtype), d_bias.to(bias.dtype), d_initial.to(initial.dtype),
                None, None, None, None, None)


def recurrence(wx, recurrent, bias, initial, cell="lstm", *, output_mode="hidden", snapshot_dtype=None, config=None,
               sequence_context=None, backward_options=None):
    """Differentiable whole-call recurrence. Returns (output, FP32 final carry)."""
    if cell not in ("lstm", "slstm"):
        raise ValueError("training supports LSTM and sLSTM")
    if output_mode not in ("full", "hidden", "last"):
        raise ValueError("output mode must be full, hidden or last")
    if snapshot_dtype not in (None, wx.dtype, torch.float32):
        raise ValueError("snapshot dtype must be input dtype or FP32")
    if sequence_context is not None and sequence_context.step_offset > 0:
        raise ValueError("a continuation chunk cannot be differentiated; run the whole logical call")
    options = backward_defaults(wx.shape[-1], backward_options, batch=wx.shape[0])
    if config is None:
        batch, steps, _, heads, width = wx.shape
        config, _ = training_forward(cell, wx.dtype, batch, steps, heads, width)
    return _Recurrence.apply(wx, recurrent, bias, initial, cell, output_mode, snapshot_dtype, config, options)
