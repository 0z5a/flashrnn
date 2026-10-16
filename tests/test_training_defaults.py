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


if __name__ == "__main__":
    unittest.main()
