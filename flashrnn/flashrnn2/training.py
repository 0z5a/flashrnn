"""Differentiable FlashRNN2 recurrence for LSTM/sLSTM training (whole logical calls).

``recurrence`` runs the forward on the backend that ``config`` (or the packaged table) selects, in full output mode
with an FP32 state snapshot, saves that history, and returns the requested output and the FP32 final carry. Its
backward is ``triton_rtiled_bw.backward``: gradients follow ``grad_reference`` (no stabilizer gradient, straight-
through normalizer clamp, as FlashRNN v1). Gradients are returned in the dtypes of the inputs.

Chunk continuation is inference only: a call with a ``sequence_context`` past the first step cannot be
differentiated here.
"""

import torch

BW_DEFAULTS = {"slice_width": None, "num_warps": 8}


def backward_slice_width(width):
    """Conservative default: one CTA per group up to D = 64, then slices that fit shared memory spill-light."""
    if width <= 64:
        return width if width in (16, 32, 64) else 64
    if width <= 256:
        return 32
    return 16


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
        if options.get("slice_width") is None:
            options["slice_width"] = backward_slice_width(wx.shape[-1])
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
    options = dict(BW_DEFAULTS, **(backward_options or {}))
    return _Recurrence.apply(wx, recurrent, bias, initial, cell, output_mode, snapshot_dtype, config, options)
