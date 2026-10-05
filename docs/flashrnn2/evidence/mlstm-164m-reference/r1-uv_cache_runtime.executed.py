"""Use already-cached, local CPython 3.12 packages without installing anything."""

import sys
from pathlib import Path


ARCHIVES = Path("/Users/0z5a/.cache/uv/archive-v0")
PINS = {
    "torch": "q1Y6anFxkpcrIg3liZ4R9",  # 2.14.1
    "numpy": "XVJVtRewlUKTuU6PiqVyj",  # 2.3.5, cp312
    "safetensors": "GXx6fIVR-Cz6apk5CsLe1",  # 0.8.0
    "tokenizers": "eEJepzv_ZhkSpmBDdOfAO",  # 0.21.4
    "transformers": "a4R-zCEhdV2k87pLRfvR3",  # 4.54.1
    "einops": "hwvrg-ocguDIK9AGIV43N",  # 0.8.2
}


def activate() -> dict[str, str]:
    preferred = [ARCHIVES / value for value in PINS.values()]
    others = []
    for path in sorted(ARCHIVES.iterdir()):
        if not path.is_dir() or path in preferred:
            continue
        if any((path / package).exists() for package in PINS):
            continue
        libraries = list(path.rglob("*.so"))
        if any("cpython-311" in library.name for library in libraries) and not any(
            "cpython-312" in library.name for library in libraries
        ):
            continue
        others.append(path)
    sys.path[0:0] = [str(path) for path in preferred + others]
    return {name: str(ARCHIVES / archive) for name, archive in PINS.items()}
