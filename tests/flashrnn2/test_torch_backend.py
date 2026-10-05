"""Public functional and import boundaries, including complete gradients."""

import os
import subprocess
import sys
import unittest
from functools import partial

import torch
from test_reference import inputs

from flashrnn import flashrnn_torch
from flashrnn.flashrnn2.reference import SIZES, recurrence


class TorchBackendTest(unittest.TestCase):
    def test_chunk_gradients_views_and_readonly(self) -> None:
        for cell in SIZES:
            with self.subTest(cell=cell):
                wx, r, b, initial = inputs(cell, steps=6, width=3)
                # A noncontiguous sequence view and mixed sLSTM initialization.
                wx = wx[:, ::2]
                if cell == "slstm":
                    with torch.no_grad():
                        initial[2, 0] = 0
                tensors = (wx, r, b, initial)
                copies = tuple(t.detach().clone() for t in tensors)
                full, final = flashrnn_torch(*tensors, cell=cell)
                first, carry = flashrnn_torch(wx[:, :1], r, b, initial, cell=cell)
                last, chunk_final = flashrnn_torch(wx[:, 1:], r, b, carry, cell=cell)
                joined = torch.cat((first, last), dim=2)
                torch.testing.assert_close(full, joined, atol=0, rtol=0)
                torch.testing.assert_close(final, chunk_final, atol=0, rtol=0)
                full_grad = torch.autograd.grad(
                    full.square().sum() + final.sum(), tensors
                )
                chunk_grad = torch.autograd.grad(
                    joined.square().sum() + chunk_final.sum(), tensors
                )
                for actual, expected in zip(chunk_grad, full_grad):
                    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
                for tensor, copy in zip(tensors, copies):
                    torch.testing.assert_close(tensor, copy, atol=0, rtol=0)
                repeated = flashrnn_torch(*tensors, cell=cell)
                torch.testing.assert_close(repeated[0], full, atol=0, rtol=0)

    def test_aliased_input_and_state(self) -> None:
        initial = torch.randn(1, 2, 1, 1, 3, dtype=torch.float64, requires_grad=True)
        wx = initial.permute(1, 2, 0, 3, 4)
        r = torch.zeros(1, 1, 3, 3, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        copy = initial.detach().clone()
        history, final = flashrnn_torch(wx, r, b, initial, cell="elman")
        torch.testing.assert_close(history, copy.tanh(), atol=0, rtol=0)
        grad = torch.autograd.grad(final.sum(), initial)[0]
        torch.testing.assert_close(grad, 1 - copy.tanh().square())
        torch.testing.assert_close(initial, copy, atol=0, rtol=0)

    def test_bf16_operand_forward_and_gradients(self) -> None:
        for cell in SIZES:
            with self.subTest(cell=cell):
                tensors = tuple(
                    t.detach().float().requires_grad_() for t in inputs(cell)
                )
                wx, r, b, initial = tensors
                actual = flashrnn_torch(
                    *tensors, cell=cell, numerics="fp32_state_bf16_mma"
                )
                expected = recurrence(
                    wx,
                    r.bfloat16().float(),
                    b,
                    initial,
                    cell=cell,
                    mma_dtype=torch.bfloat16,
                )
                for a, e in zip(actual, expected):
                    self.assertEqual(a.dtype, torch.float32)
                    torch.testing.assert_close(a, e, atol=0, rtol=0)
                ag = torch.autograd.grad(
                    actual[0].square().sum() + actual[1].sum(), tensors
                )
                eg = torch.autograd.grad(
                    expected[0].square().sum() + expected[1].sum(), tensors
                )
                for a, e in zip(ag, eg):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_public_gradcheck(self) -> None:
        for cell in SIZES:
            with self.subTest(cell=cell):
                self.assertTrue(
                    torch.autograd.gradcheck(
                        partial(flashrnn_torch, cell=cell),
                        inputs(cell, batch=1, steps=2, heads=1, width=2),
                        fast_mode=True,
                        eps=1e-6,
                        atol=1e-5,
                        rtol=1e-4,
                    )
                )

    def test_dtype_and_device_preserved(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                tensors = tuple(t.detach().to(dtype) for t in inputs("lstm"))
                for output in flashrnn_torch(*tensors):
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(output.device, tensors[0].device)

    def test_ambient_autocast_keeps_explicit_policy(self) -> None:
        tensors = tuple(t.detach().float().requires_grad_() for t in inputs("slstm"))
        for numerics in ("mathematical", "fp32_state_bf16_mma"):
            with self.subTest(numerics=numerics):
                expected = flashrnn_torch(*tensors, cell="slstm", numerics=numerics)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = flashrnn_torch(*tensors, cell="slstm", numerics=numerics)
                for a, e in zip(actual, expected):
                    self.assertEqual(a.dtype, torch.float32)
                    torch.testing.assert_close(a, e, atol=0, rtol=0)
                ag = torch.autograd.grad(actual[0].square().sum(), tensors)
                eg = torch.autograd.grad(expected[0].square().sum(), tensors)
                for a, e in zip(ag, eg):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_reject_invalid_contract(self) -> None:
        wx, r, b, state = inputs("lstm")
        cases = [
            ((wx[0], r, b, state), {}, "ranks"),
            ((wx[:, :0], r, b, state), {}, "positive"),
            ((wx, r.float(), b, state), {}, "same dtype"),
            ((wx, r.to("meta"), b, state), {}, "same device"),
            ((wx.to_sparse(), r, b, state), {}, "strided"),
            ((wx, r, b[:1], state), {}, "bias gate"),
            ((wx, r, b, state[:1]), {}, "initial state"),
            ((wx, r[..., :1], b, state), {}, "recurrent weights"),
            ((wx, r, b, state), {"cell": "unknown"}, "unsupported cell"),
            ((wx, r, b, state), {"numerics": "unknown"}, "unsupported numerical"),
            ((wx, r, b, state), {"numerics": "fp32_state_bf16_mma"}, "FP32"),
            (tuple(t.long() for t in (wx, r, b, state)), {}, "float16"),
        ]
        for tensors, kwargs, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                flashrnn_torch(*tensors, **kwargs)

    def test_import_and_explicit_gpu_boundary(self) -> None:
        code = r"""
import importlib.abc
import importlib.util
import os
import shutil
import subprocess
import sys
from unittest.mock import patch
import torch

blocked = ("triton", "ninja", "torch.utils.cpp_extension",
           "flashrnn.flashrnn.cuda_init", "flashrnn.flashrnn.cuda_init_parametric",
           "flashrnn.flashrnn.gpu_info", "flashrnn.flashrnn.triton_fused")
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("blocked accelerated import: " + fullname)

if os.environ.get("FLASHRNN_REQUIRE_CPU_WHEEL") == "1":
    assert torch.version.cuda is None and torch.version.hip is None
    assert importlib.util.find_spec("triton") is None
    assert importlib.util.find_spec("ninja") is None
    assert shutil.which("nvcc") is None

sys.meta_path.insert(0, Guard())
with patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("GPU query")), \
     patch.object(torch.cuda, "current_device", side_effect=AssertionError("GPU query")), \
     patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")), \
     patch.object(subprocess, "Popen", side_effect=AssertionError("compiler process")):
    from flashrnn import flashrnn, flashrnn_torch, FlashRNNConfig
    x = torch.zeros(1, 2, 4, 1, 3, requires_grad=True)
    r = torch.zeros(4, 1, 3, 3, requires_grad=True)
    b = torch.zeros(4, 1, 3, requires_grad=True)
    s = torch.zeros(2, 1, 1, 1, 3, requires_grad=True)
    h, f = flashrnn_torch(x, r, b, s)
    torch.autograd.grad(h.sum() + f.sum(), (x, r, b, s))
    expected = flashrnn(x, r, b, states=s, function="lstm", backend="vanilla")
    torch.testing.assert_close(h, expected[0])
    assert not any(name in sys.modules for name in blocked)

# Explicit legacy GPU selection still reaches its lazy loader.
import unittest
case = unittest.TestCase()
for backend in ("cuda", "cuda_fused", "triton_fused"):
    with case.assertRaisesRegex(ImportError, "blocked accelerated import"):
        flashrnn(x, r, b, states=s, function="lstm", backend=backend)
"""
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
