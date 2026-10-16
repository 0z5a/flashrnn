"""CPU checks for full-call state precision and initialization boundaries."""

import unittest

import torch

from flashrnn.flashrnn2.reference import recurrence
from flashrnn.flashrnn2.state_contract import (
    SequenceContext,
    begin_sequence,
    state_contract,
)


class PersistentContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(187)

    def fixture(self, cell):
        states = 4 if cell == "slstm" else 2
        wx = (torch.randn(3, 17, 4, 2, 48) * 0.2).to(torch.bfloat16).float()
        r = (torch.randn(4, 2, 48, 48) * 0.03).to(torch.bfloat16).float()
        bias = (torch.randn(4, 2, 48) * 0.1).to(torch.bfloat16).float()
        initial = (torch.randn(states, 3, 1, 2, 48) * 0.1).to(torch.bfloat16).float()
        return wx, r, bias, initial

    def test_ragged_chunks_preserve_full_fp32_carry(self):
        for cell in ("lstm", "slstm"):
            for normalizer in ("zero", "nonzero", "mixed"):
                with self.subTest(cell=cell, normalizer=normalizer):
                    wx, r, bias, initial = self.fixture(cell)
                    if cell == "slstm":
                        initial[2].fill_(0 if normalizer == "zero" else 1)
                        if normalizer == "mixed":
                            initial[2, :1].zero_()
                    context = begin_sequence(initial, cell, torch.bfloat16)
                    expected, final = recurrence(
                        wx,
                        r,
                        bias,
                        initial,
                        cell,
                        torch.bfloat16,
                        sequence_context=context,
                    )
                    outputs, carry, offset = [], initial, 0
                    for length in (1, 3, 13):
                        history, carry = recurrence(
                            wx[:, offset : offset + length],
                            r,
                            bias,
                            carry,
                            cell,
                            torch.bfloat16,
                            sequence_context=context,
                        )
                        outputs.append(history)
                        context = context.advance(length)
                        offset += length
                    torch.testing.assert_close(
                        torch.cat(outputs, 2), expected, atol=0, rtol=0
                    )
                    torch.testing.assert_close(carry, final, atol=0, rtol=0)
                    self.assertEqual(context.step_offset, 17)

    def test_mixed_batch_shards_share_global_predicate(self):
        wx, r, bias, initial = self.fixture("slstm")
        initial[2].zero_()
        initial[2, 1:].fill_(1)
        initial[3].fill_(2)
        context = begin_sequence(initial, "slstm", torch.bfloat16)
        self.assertFalse(context.zero_normalizer.item())
        expected, _ = recurrence(
            wx, r, bias, initial, "slstm", torch.bfloat16, sequence_context=context
        )
        actual, _ = recurrence(
            wx[:1],
            r,
            bias,
            initial[:, :1],
            "slstm",
            torch.bfloat16,
            sequence_context=context,
        )
        # CPU GEMM changes reduction order with batch shape. This new CPU-only
        # budget is below 1e-6; GPU qualification budgets are unchanged.
        torch.testing.assert_close(actual, expected[:, :1], atol=5e-7, rtol=0)
        local = begin_sequence(initial[:, :1], "slstm", torch.bfloat16)
        wrong, _ = recurrence(
            wx[:1],
            r,
            bias,
            initial[:, :1],
            "slstm",
            torch.bfloat16,
            sequence_context=local,
        )
        self.assertGreater((wrong - actual).abs().max().item(), 0.01)

    def test_mixed_head_shards_share_global_predicate(self):
        wx, r, bias, initial = self.fixture("slstm")
        initial[2].zero_()
        initial[2, :, :, 1].fill_(1)
        initial[3].fill_(2)
        context = begin_sequence(initial, "slstm", torch.bfloat16)
        self.assertFalse(context.zero_normalizer.item())
        expected, _ = recurrence(
            wx, r, bias, initial, "slstm", torch.bfloat16, sequence_context=context
        )
        actual, _ = recurrence(
            wx[:, :, :, :1],
            r[:, :1],
            bias[:, :1],
            initial[:, :, :, :1],
            "slstm",
            torch.bfloat16,
            sequence_context=context,
        )
        # CPU GEMM may also change reduction order when the head shape changes.
        # Keep the same sub-1e-6 CPU budget as the batch-shard control above.
        torch.testing.assert_close(actual, expected[:, :, :, :1], atol=5e-7, rtol=0)
        local = begin_sequence(initial[:, :, :, :1], "slstm", torch.bfloat16)
        wrong, _ = recurrence(
            wx[:, :, :, :1],
            r[:, :1],
            bias[:, :1],
            initial[:, :, :, :1],
            "slstm",
            torch.bfloat16,
            sequence_context=local,
        )
        self.assertGreater((wrong - actual).abs().max().item(), 0.01)

    def test_output_rounding_is_not_a_continuation(self):
        wx, r, bias, initial = self.fixture("lstm")
        context = begin_sequence(initial, "lstm", torch.bfloat16)
        _, carry = recurrence(
            wx[:, :4],
            r,
            bias,
            initial,
            "lstm",
            torch.bfloat16,
            sequence_context=context,
        )
        expected, _ = recurrence(
            wx[:, 4:],
            r,
            bias,
            carry,
            "lstm",
            torch.bfloat16,
            sequence_context=context.advance(4),
        )
        wrong, _ = recurrence(
            wx[:, 4:],
            r,
            bias,
            carry.bfloat16().float(),
            "lstm",
            torch.bfloat16,
            sequence_context=context.advance(4),
        )
        self.assertGreater((expected - wrong).abs().max().item(), 0)

    def test_model_contracts_keep_distinct_arithmetic(self):
        generic = state_contract("slstm", torch.bfloat16)
        tirex = state_contract("slstm", torch.bfloat16, "tirex")
        xlstm = state_contract("slstm", torch.float32, "xlstm")
        self.assertEqual(generic.compute_dtype, torch.float32)
        self.assertEqual(generic.output_dtype, torch.bfloat16)
        self.assertEqual(tirex.transport_dtype, torch.bfloat16)
        self.assertEqual(xlstm.normalizer_scope, "elementwise_each_step")
        with self.assertRaises(ValueError):
            state_contract("lstm", torch.bfloat16, "tirex")
        with self.assertRaises(ValueError):
            state_contract("slstm", torch.bfloat16, "xlstm")

    def test_context_validation(self):
        contract = state_contract("slstm", torch.bfloat16)
        for scalar, offset in (
            (torch.zeros(2, dtype=torch.bool), 0),
            (torch.tensor(0.0), 0),
            (torch.tensor(False), -1),
        ):
            with self.assertRaises(ValueError):
                SequenceContext(contract, scalar, offset)
        context = SequenceContext(contract, torch.tensor(False))
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                context.advance(steps)
        wx, r, bias, initial = self.fixture("slstm")
        with self.assertRaisesRegex(ValueError, "rounding"):
            recurrence(wx, r, bias, initial, "slstm", sequence_context=context)


if __name__ == "__main__":
    unittest.main()
