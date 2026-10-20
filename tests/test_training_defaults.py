"""Backward option defaults of the training path (no GPU needed)."""
import unittest

try:
    from flashrnn.flashrnn2 import training
except ImportError:  # torch or triton missing
    training = None


@unittest.skipIf(training is None, "needs torch")
class BackwardDefaults(unittest.TestCase):
    def test_package_default_below_192(self):
        for width in (64, 128):
            self.assertEqual(training.backward_defaults(width, batch=16), training.BW_DEFAULTS)

    def test_split_gates_every_batch_up_to_384(self):
        for width, warps in ((192, 4), (256, 4), (384, 8)):
            for batch in (1, 16, 64, 256):
                o = training.backward_defaults(width, batch=batch)
                self.assertEqual((o["split_gates"], o["slice_width"], o["num_warps"], o.get("fuse_pre", 0)),
                                 (True, 16, warps, 0))

    def test_split_gates_small_batch_only_at_512_768(self):
        for width in (512, 768):
            o = training.backward_defaults(width, batch=16)
            self.assertEqual((o["split_gates"], o["slice_width"], o["num_warps"]), (True, 16, 8))
            for batch in (17, 64, None):
                self.assertEqual(training.backward_defaults(width, batch=batch), training.BW_DEFAULTS)

    def test_caller_overrides(self):
        self.assertEqual(training.backward_defaults(256, {"num_warps": 8}, batch=16)["num_warps"], 8)
        self.assertFalse(training.backward_defaults(384, {"split_gates": False}, batch=16)["split_gates"])
        o = training.backward_defaults(384, {"fast": False}, batch=16)
        self.assertEqual((o["split_gates"], o["fuse_pre"]), (False, 0))


class _Policy:
    """Stand-in for an ExecutionPolicy: one table entry, or none."""

    def __init__(self, table):
        self.table = table

    def select(self, cell, dtype, batch, steps, heads, width, identity=None):
        from flashrnn.flashrnn2.execution_policy import KernelConfig, shape_key
        key = shape_key(cell, dtype, batch, steps, heads, width)
        return self.table.get(key) or KernelConfig(family="mma", batch_tile=16, warps=4)


@unittest.skipIf(training is None, "needs torch")
class TrainingForward(unittest.TestCase):
    def setUp(self):
        import torch
        from flashrnn.flashrnn2.execution_policy import KernelConfig, shape_key
        self.torch, self.KernelConfig = torch, KernelConfig
        self.entry = KernelConfig(family="mma_rtiled", warps=4, slice_width=32, sync="grid")
        self.policy = _Policy({shape_key("lstm", "float16", 32, 64, 1, 384): self.entry})
        from flashrnn.flashrnn2 import dispatch
        self.dispatch, self.identity = dispatch, dispatch.runtime_identity
        dispatch.runtime_identity = lambda: {}  # no GPU needed: the stand-in policy ignores the identity
        self.installed = training._tilelang_installed
        training._tilelang_installed = lambda: True

    def tearDown(self):
        training._tilelang_installed = self.installed
        self.dispatch.runtime_identity = self.identity

    def test_table_entry_wins(self):
        cfg, src = training.training_forward("lstm", self.torch.float16, 32, 64, 1, 384, policy=self.policy)
        self.assertEqual((cfg, src), (self.entry, "table"))

    def test_tilelang_outside_the_table(self):
        for width, (sync, s) in training.TILELANG_TRAINING_FORWARD.items():
            cfg, src = training.training_forward("slstm", self.torch.float16, 16, 256, 1, width, policy=self.policy)
            self.assertEqual((src, cfg.family, cfg.sync, cfg.slice_width), ("tilelang", "tilelang", sync, s))

    def test_tilelang_without_matching_tables(self):
        # A runtime whose identity differs from the packaged tables (empty policy) still gets the TileLang forward,
        # as default_config does at D >= 1024 (max-width.md section 75).
        cfg, src = training.training_forward("lstm", self.torch.float16, 16, 256, 1, 384, policy=_Policy({}))
        self.assertEqual((src, cfg.family), ("tilelang", "tilelang"))

    def test_default_when_not_measured_or_not_installed(self):
        cases = [(self.policy, 320, 1, True),          # width not measured
                 (self.policy, 384, 2, True),          # H > 1 not measured
                 (self.policy, 384, 1, False)]         # TileLang not installed
        for policy, width, heads, installed in cases:
            training._tilelang_installed = lambda installed=installed: installed
            cfg, src = training.training_forward("lstm", self.torch.float16, 16, 256, heads, width, policy=policy)
            self.assertEqual(src, "default")


if __name__ == "__main__":
    unittest.main()
