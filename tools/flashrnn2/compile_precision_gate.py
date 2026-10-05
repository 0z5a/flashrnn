"""Run the unchanged compile gate with verified eager-precision preservation."""

import json
import runpy
from pathlib import Path

from torch._inductor import config

assert config.emulate_precision_casts
print(
    json.dumps({"emulate_precision_casts": config.emulate_precision_casts}), flush=True
)
runpy.run_path(str(Path(__file__).with_name("compile_gate.py")), run_name="__main__")
