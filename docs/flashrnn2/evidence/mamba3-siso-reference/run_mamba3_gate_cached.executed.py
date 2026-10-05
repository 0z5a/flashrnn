"""Run Mamba-3 SISO reference with the existing local Python package cache."""

import runpy
import sys
from pathlib import Path

from uv_cache_runtime import activate

activate()
gate = Path(__file__).resolve().parent / "source/tools/flashrnn2/mamba3_reference_gate.py"
sys.path.insert(0, str(gate.parent))
sys.argv[0] = str(gate)
runpy.run_path(str(gate), run_name="__main__")
