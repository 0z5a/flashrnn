"""Policy contract checks; numerical and performance gates run on real CUDA."""
import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "flashrnn/flashrnn2/execution_policy.py"
spec = importlib.util.spec_from_file_location("frnn_execution_policy", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
KernelConfig, ExecutionPolicy, shape_key = module.KernelConfig, module.ExecutionPolicy, module.shape_key


class PolicyContract(unittest.TestCase):
    def test_unknown_shape_uses_existing_mma(self):
        self.assertEqual(ExecutionPolicy().select("slstm", "bfloat16", 2, 64, 4, 128), KernelConfig())

    def test_table_requires_matching_hardware_and_compiler(self):
        identity = {"gpu": "H100", "triton": "3.6.0", "source": "frozen"}
        key = shape_key("lstm", "float16", 16, 128, 2, 64)
        policy = ExecutionPolicy({key: {"warps": 8}}, identity=identity)
        for wrong in (None, {}, dict(identity, triton="different"), dict(identity, gpu="other")):
            with self.assertRaises(ValueError):
                policy.select("lstm", "float16", 16, 128, 2, 64, identity=wrong)
        self.assertEqual(policy.select("lstm", "float16", 16, 128, 2, 64, identity=identity).warps, 8)
        self.assertEqual(policy.select("slstm", "float16", 16, 128, 2, 64, identity=identity), KernelConfig())

    def test_invalid_simt_mapping_is_rejected(self):
        for kwargs in ({"family": "simt"}, {"family": "simt", "batch_tile": 1, "weight_operand": "a"}):
            with self.assertRaises(ValueError):
                KernelConfig(**kwargs)

    def test_mma_tile_must_satisfy_supported_mapping(self):
        for tile in (1, 2, 4, 8, 64):
            with self.assertRaises(ValueError):
                KernelConfig(batch_tile=tile)


class RTiledPolicyContract(unittest.TestCase):
    def test_unknown_large_width_uses_rtiled_default(self):
        self.assertEqual(ExecutionPolicy().select("lstm", "float16", 16, 128, 1, 128), KernelConfig())
        cfg = ExecutionPolicy().select("lstm", "float16", 16, 128, 1, 256)
        self.assertEqual((cfg.family, cfg.slice_width, cfg.sync, cfg.warps), ("mma_rtiled", 64, "grid", 8))
        self.assertEqual(ExecutionPolicy().select("slstm", "bfloat16", 4, 64, 2, 768).slice_width, 32)

    def test_rtiled_fields_are_validated(self):
        self.assertEqual(KernelConfig(family="mma_rtiled", slice_width=32, sync="grid").slice_width, 32)
        for kwargs in (dict(family="mma_rtiled"), dict(family="mma_rtiled", slice_width=48, sync="grid"),
                       dict(family="mma_rtiled", slice_width=64, sync="cluster"),
                       dict(family="mma_rtiled", slice_width=64, sync="grid", batch_tile=32),
                       dict(family="mma_rtiled", slice_width=64, sync="grid", weight_operand="a"),
                       dict(slice_width=64), dict(sync="grid"), dict(pad_k=True)):
            with self.assertRaises(ValueError):
                KernelConfig(**kwargs)


class TileLangPolicyContract(unittest.TestCase):
    def test_tilelang_fields_are_validated(self):
        cfg = KernelConfig(family="tilelang", slice_width=16, sync="cluster")
        self.assertEqual((cfg.family, cfg.slice_width, cfg.warps), ("tilelang", 16, 4))
        for kwargs in (dict(family="tilelang", slice_width=16), dict(family="tilelang", slice_width=128, sync="cluster"),
                       dict(family="tilelang", slice_width=32, sync="grid"),
                       dict(family="tilelang", slice_width=32, sync="cluster", warps=8),
                       dict(family="tilelang", slice_width=32, sync="cluster", pad_k=True), dict(sync="cluster")):
            with self.assertRaises(ValueError):
                KernelConfig(**kwargs)

    def test_tilelang_entry_requires_tilelang_in_identity(self):
        key = shape_key("lstm", "float16", 16, 1024, 1, 256)
        entry = {"family": "tilelang", "slice_width": 16, "sync": "cluster"}
        with_tl = {"gpu": "H100", "triton": "3.6.0", "tilelang": "0.1.15"}
        policy = ExecutionPolicy({key: entry}, identity=with_tl)
        self.assertEqual(policy.select("lstm", "float16", 16, 1024, 1, 256, identity=with_tl).family, "tilelang")
        without = dict(with_tl, tilelang=None)
        with self.assertRaises(ValueError):
            policy.select("lstm", "float16", 16, 1024, 1, 256, identity=without)
        unbound = ExecutionPolicy({key: entry}, identity=without)
        with self.assertRaises(ValueError):
            unbound.select("lstm", "float16", 16, 1024, 1, 256, identity=without)

    def test_unmeasured_large_width_keeps_triton_default(self):
        self.assertEqual(ExecutionPolicy().select("lstm", "float16", 16, 1024, 1, 256).family, "mma_rtiled")


if __name__ == "__main__":
    unittest.main()
