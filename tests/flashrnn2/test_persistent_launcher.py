"""Check kernel ABI without importing a CUDA/Triton runtime."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PersistentLauncherTest(unittest.TestCase):
    def test_both_kernel_signatures_match_the_real_launcher(self):
        trees = [
            ast.parse((ROOT / "flashrnn/flashrnn2" / name).read_text())
            for name in ("triton_persistent.py", "gluon_persistent.py")
        ]
        signatures = [
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_sequence"
            ).args
            for tree in trees
        ]
        names = [[arg.arg for arg in args.args] for args in signatures]
        # Prepared execution adds optional Triton-only addressing/output flags.
        # The public launcher still binds the common required ABI, including
        # the Gluon path, without passing those optional flags.
        self.assertEqual(names[0][: len(names[1])], names[1])
        self.assertEqual(names[0][len(names[1]) :],
                         ["TOTAL_T", "STEP_BASE", "OUTPUT_MODE"])
        self.assertEqual(len(signatures[0].defaults), 3)
        self.assertEqual([ast.literal_eval(value)
                          for value in signatures[0].defaults], [0, 0, 0])
        self.assertEqual(signatures[1].defaults, [])
        launch = next(
            node
            for node in ast.walk(trees[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sequence"
        )
        self.assertEqual(len(launch.args), len(names[1]))
        self.assertIn("FINAL", names[0])
        self.assertIn("SEQUENCE_START", names[0])
        prepared = ast.parse(
            (ROOT / "flashrnn/flashrnn2/prepared.py").read_text()
        )
        prepared_launch = next(
            node for node in ast.walk(prepared)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_sequence"
        )
        self.assertEqual(len(prepared_launch.args), len(names[1]))
        self.assertEqual(
            [keyword.arg for keyword in prepared_launch.keywords
             if keyword.arg not in ("num_warps", "enable_fp_fusion")],
            ["TOTAL_T", "STEP_BASE", "OUTPUT_MODE"],
        )


if __name__ == "__main__":
    unittest.main()
