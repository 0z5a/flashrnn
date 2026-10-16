"""Backward option defaults of the training path (no GPU needed)."""
import unittest

try:
    from flashrnn.flashrnn2 import training
except ImportError:  # torch or triton missing
    training = None


@unittest.skipIf(training is None, "needs torch")
class BackwardDefaults(unittest.TestCase):
    def test_unfused_default_away_from_256(self):
        for width in (64, 192, 384, 512, 768):
            options = training.backward_defaults(width)
            self.assertEqual(options, training.BW_DEFAULTS)
            self.assertFalse(options.get("fuse_pre", 0))

    def test_fused_default_at_256(self):
        options = training.backward_defaults(256)
        self.assertEqual((options["fuse_pre"], options["slice_width"], options["num_warps"]), (2, 16, 4))
        self.assertTrue(options["fast"])

    def test_caller_overrides(self):
        self.assertEqual(training.backward_defaults(256, {"num_warps": 8})["num_warps"], 8)
        self.assertEqual(training.backward_defaults(256, {"fuse_pre": 0})["fuse_pre"], 0)
        # Without the fast variant the fused default is dropped (fused kernels are fast-kernel variants).
        self.assertEqual(training.backward_defaults(256, {"fast": False})["fuse_pre"], 0)


if __name__ == "__main__":
    unittest.main()
