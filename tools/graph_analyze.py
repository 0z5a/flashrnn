"""Analysis of batch "graph" (docs/flashrnn2/max-width.md section 72.1).

Usage: graph_analyze.py <results dir with science/claims> [--out FILE]
Per claim and arm: other / one (the library default, eager), 95% hierarchical bootstrap (10,000 draws: starts, then
blocks); > 1 means the default is faster. Derived per shape (point ratios of means):
  graph_vs_graph  v1cuda_graph / one_graph  (patched alternating, graphed, against FlashRNN2 graphed)
  alt_graph_gain  v1cuda / v1cuda_graph     (what a graph does for the patched alternating kernel)
"""
import argparse
import json
from pathlib import Path

from heads_analyze import bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = {}
    for cdir in sorted((a.batch / "science/claims").glob("*")):
        per, errors, case, stable = {}, {}, None, {}
        for sf in sorted(cdir.glob("start*/paired.json")):
            d = json.loads(sf.read_text())
            case = d["case"]
            if d.get("status") != "DONE":
                errors[sf.parent.name] = (d.get("error") or "")[-300:]
                continue
            errors.update({f"{sf.parent.name}:{k}": v[-200:] for k, v in d.get("arm_errors", {}).items()})
            stable.update({f"{sf.parent.name}:{k}": v for k, v in d.get("outputs_stable", {}).items() if not v})
            for b in d["blocks"]:
                for arm, v in b.items():
                    per.setdefault(arm, {}).setdefault(sf.parent.name, []).append((v["one_ms"], v["other_ms"]))
        group = "patched" if "graphp" in cdir.name else "orig"
        key = (case["cell"], case["dtype"], case["width"], case["batch"])
        row = rows.setdefault(key, dict(cell=key[0], dtype=key[1], width=key[2], batch=key[3], ratios={}, means={},
                                        errors={}, unstable={}))
        row["errors"][group] = errors
        row["unstable"][group] = stable
        for arm, starts in per.items():
            r = bootstrap(list(starts.values()), hash((cdir.name, arm)) & 0xFFFF)
            name = f"{arm}@{group}"
            row["ratios"][name] = dict(ratio=r[0], lo=r[1], hi=r[2])
            flat = [p for s in starts.values() for p in s]
            row["means"][f"one@{group}"] = sum(p[0] for p in flat) / len(flat)
            row["means"][name] = sum(p[1] for p in flat) / len(flat)
    for row in rows.values():
        m = row["means"]
        if "v1cuda_graph@patched" in m and "one_graph@patched" in m:
            row["graph_vs_graph"] = m["v1cuda_graph@patched"] / m["one_graph@patched"]
            row["alt_graph_gain"] = m["v1cuda@patched"] / m["v1cuda_graph@patched"]
    out = dict(rows=[dict(r, key=list(k)) for k, r in rows.items()])
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=str))
    f = lambda x: f"{x['ratio']:.2f} [{x['lo']:.2f},{x['hi']:.2f}]" if x else "missing"  # noqa: E731
    for k, r in sorted(rows.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][2])):
        rr = {k: r["ratios"].get(k) for k in ("one_graph@orig", "v1@orig", "v1cuda@orig", "v1cuda@patched",
                                                "v1cuda_graph@patched")}
        print(f"{r['cell']:5s} D{r['width']} B{r['batch']}: one={r['means'].get('one@orig', 0):.2f}ms "
              f"graph={f(rr['one_graph@orig'])} fused={f(rr['v1@orig'])} alt={f(rr['v1cuda@orig'])} | "
              f"altP={f(rr['v1cuda@patched'])} altP_graph={f(rr['v1cuda_graph@patched'])} "
              f"altG/oneG={r.get('graph_vs_graph', 0):.2f} altGain={r.get('alt_graph_gain', 0):.2f} "
              f"err={sum(len(v) for v in r['errors'].values())} unstable={sum(len(v) for v in r['unstable'].values())}")


if __name__ == "__main__":
    main()
