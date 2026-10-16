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


if __name__ == "__main__":
    unittest.main()
