"""Summarize a backward qualification batch (science/cases/*/bw.json, science/sanitizer/*, control/driver-result.json).

Usage: summarize_bw.py <batch-evidence-dir> [--out FILE]. Prints per-gate counts, the worst gradient error per
(cell, dtype, D) against the FP64 oracle and FlashRNN1's error on the same cases, and every failing case.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    cases = sorted(a.batch.glob("science/cases/*/bw.json"))
    gates = defaultdict(lambda: defaultdict(int))
    worst = {}
    v1 = defaultdict(list)
    failing = []
    for f in cases:
        rec = json.loads(f.read_text())
        c = rec["case"]
        key = (c["cell"], c["dtype"], c["width"])
        for g, v in rec.get("gates", {}).items():
            gates[g][v] += 1
        for loss in ("hist", "full"):
            m = rec.get("grads", {}).get(loss, {}).get("metrics", {})
            for name, x in m.items():
                k = key + (name,)
                worst[k] = max(worst.get(k, 0.0), x["rel_fro"])
        if rec.get("v1", {}).get("status") == "AVAILABLE":
            ours = rec["grads"]["hist"]["metrics"]
            for name, x in rec["v1"]["metrics"].items():
                v1[key + (name,)].append((ours[name]["rel_fro"], x["rel_fro"]))
        elif "v1" in rec:
            v1[key + ("unavailable",)].append(rec["v1"].get("status"))
        if rec.get("status") != "PASS":
            failing.append(dict(name=f.parent.name, gates=rec.get("gates"),
                                errors={k: v[-400:] for k, v in rec.items() if k.endswith("_error")},
                                checks={loss: {n: ok for n, ok in rec.get("grads", {}).get(loss, {}).get("checks",
                                        {}).items() if not ok} for loss in ("hist", "full")},
                                modes=rec.get("modes", {}).get("bitwise"), repeat=rec.get("repeat")))
    table = defaultdict(dict)
    for (cell, dtype, width, name), w in worst.items():
        table[(cell, dtype, width)][name] = w
    rows = []
    for key in sorted(table):
        r = dict(cell=key[0], dtype=key[1], width=key[2], worst_rel_fro=table[key])
        pairs = {n: v for (c, d, w, n), v in v1.items() if (c, d, w) == key and n != "unavailable"}
        if pairs:
            r["ours_over_v1_max"] = {n: max(o / max(b, 1e-300) for o, b in v) for n, v in pairs.items()}
            r["v1_rel_fro_max"] = {n: max(b for _, b in v) for n, v in pairs.items()}
        un = v1.get(key + ("unavailable",))
        if un:
            r["v1_unavailable_cases"] = len(un)
        rows.append(r)
    driver = json.loads((a.batch / "control/driver-result.json").read_text()) if \
        (a.batch / "control/driver-result.json").exists() else {}
    san = [dict(name=r.get("name"), clean=r.get("clean"), exit=r.get("exit"))
           for r in driver.get("phases", {}).get("sanitizer", {}).get("records", [])]
    out = dict(cases=len(cases), gates={g: dict(v) for g, v in gates.items()}, per_shape=rows, failing=failing,
               sanitizer=san, driver_status=driver.get("status"))
    text = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(text)
    print(json.dumps(dict(cases=len(cases), gates=out["gates"], failing=len(failing), sanitizer=san,
                          driver=driver.get("status")), indent=1))
    for r in rows:
        print(r["cell"], r["dtype"], r["width"], "worst", {k: f"{v:.2e}" for k, v in r["worst_rel_fro"].items()},
              "ours/v1", {k: f"{v:.2f}" for k, v in r.get("ours_over_v1_max", {}).items()},
              "v1_unavail", r.get("v1_unavailable_cases", 0))


if __name__ == "__main__":
    main()
