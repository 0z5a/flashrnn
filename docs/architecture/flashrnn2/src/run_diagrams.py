"""Convenience launcher; discovers optional desktop Cairo without fixed user paths."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["render","validate","inspect","all"])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    cairo=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/lib"
    if sys.platform=="darwin" and cairo.is_dir():
        old=env.get("DYLD_FALLBACK_LIBRARY_PATH")
        env["DYLD_FALLBACK_LIBRARY_PATH"]=str(cairo)+(":"+old if old else "")
    commands={
        "render":["render_figures.py","--model",str(root/"architecture.json"),
                  "--layout",str(root/"layouts.json"),"--style",str(root/"style.json"),
                  "--out",str(root/"figures"),"--formats","svg,png,pdf"],
        "validate":["validate_figures.py","--root",str(root)],
        "inspect":["inspect_exports.py","--root",str(root)],
    }
    for command in ["render","validate","inspect"] if args.command=="all" else [args.command]:
        argv=commands[command]
        subprocess.run([sys.executable,str(root/"src"/argv[0]),*argv[1:]],env=env,check=True)


if __name__=="__main__":
    main()
