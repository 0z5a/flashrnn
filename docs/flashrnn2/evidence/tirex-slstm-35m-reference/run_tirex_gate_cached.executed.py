"""Run the TiRex gate with existing cached Python packages only."""

import runpy
import sys
from pathlib import Path

from uv_cache_runtime import activate

activate()
gate = (
    Path(__file__).resolve().parent
    / "source/tools/flashrnn2/tirex_slstm_reference_gate.py"
)
sys.argv[0] = str(gate)
runpy.run_path(str(gate), run_name="__main__")
