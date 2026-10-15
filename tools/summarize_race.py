"""Summarize a race-verification batch (science/tasks/*/race.json).

Usage: summarize_race.py <batch-evidence-dir> [--out FILE]. One row per task and variant: runs, flagged runs,
runs whose output differs from the reference, silent corruptions, errors and the verdict under the pre-registered
rule (tools/race_verify.py).
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    rows = []
    for f in sorted(a.batch.glob("science/tasks/*/race.json")):
        rec = json.loads(f.read_text())
        t = rec["task"]
        c = t["config"]
        label = f"{t['path']} {t['cell']} D{c['width']} S{c['slice_width']}" + \
            (f" BM{c['bm']} T{c['threads']}" if "bm" in c else f" W{c['warps']}")
        if rec.get("status") == "ERROR":
            rows.append(dict(task=label, variant="-", error=rec.get("error", "")[-300:]))
            continue
        for variant, s in rec.get("summary", {}).items():
            max_viol = max((r.get("violations", 0) for r in rec["runs"][variant] if "error" not in r), default=0)
            rows.append(dict(task=label, variant=variant, runs=s["runs"], flagged=s["flagged"],
                             outputs_differ=s["outputs_differ"], silent=s["silent_corruption"], errors=s["errors"],
                             max_violations=max_viol, verdict="PASS" if s["pass_"] else "FAIL"))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        if "error" in r:
            print(f"{r['task']:<42} ERROR {r['error'][-160:]}")
        else:
            print(f"{r['task']:<42} {r['variant']:<22} flagged {r['flagged']:>2}/{r['runs']}  differ {r['outputs_differ']:>2}"
                  f"  silent {r['silent']}  err {r['errors']}  maxviol {r['max_violations']:>6}  {r['verdict']}")


if __name__ == "__main__":
    main()
