"""RetNet numerical diagnostic with a fixed Linear row shape."""

import argparse
import hashlib
import json
from pathlib import Path

from deltanet_row_linear_reference import token_linear
from retnet_reference_gate import main as reference_main
from torch.nn import functional as F

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()
sidecar = args.output.with_suffix(".arithmetic.json")
assert not sidecar.exists() and not args.output.exists()
sidecar.write_text(
    json.dumps(
        {
            "scope": "FULL_RETNET_CHANGED_LINEAR_ROW_SHAPE_DIAGNOSTIC",
            "arithmetic_changed": True,
            "linear_input_rows": 1,
            "retention_and_rotary_unchanged": True,
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
