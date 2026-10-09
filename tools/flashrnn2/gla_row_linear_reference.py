"""Diagnose full GLA continuation with a fixed Linear row shape."""

import argparse
import hashlib
import json
from pathlib import Path

from deltanet_row_linear_reference import token_linear
from gla_reference_gate import main as reference_main
from torch.nn import functional as F


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", choices=("gla",), default="gla")
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    sidecar = args.output.with_suffix(".arithmetic.json")
    assert not sidecar.exists() and not args.output.exists()
    sidecar.write_text(
        json.dumps(
            {
                "scope": "FULL_GLA_CHANGED_LINEAR_ROW_SHAPE_DIAGNOSTIC",
                "arithmetic_changed": True,
                "linear_input_rows": 1,
                "original_gate_replaced": False,
                "performance_claim": False,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), Path(token_linear.__code__.co_filename))
                },
            },
            indent=2,
        )
        + "\n"
    )
    F.linear = token_linear
    reference_main()


if __name__ == "__main__":
    main()
