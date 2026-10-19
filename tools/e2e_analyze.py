"""Analysis of batch "e2e" (docs/flashrnn2/max-width.md section 70).

Usage: e2e_analyze.py <results dir with science/> [--out FILE]
Per case: other / frnn2 time for every other arm (v1 = FlashRNN cuda_fused, v1cuda = FlashRNN cuda, cudnn = cuDNN),
95% hierarchical bootstrap (10,000 draws: starts, then blocks); > 1 means frnn2 is faster, and a speedup (or a loss)
holds when the whole interval is above (or below) 1. Checks (start 0): layer -> relative Frobenius difference of
output and gradients to frnn2; lm -> largest relative loss difference to frnn2 over the first steps.
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
    rows = []
    for cdir in sorted((a.batch / "science").glob("*")):
        starts, errors, check, case, cfg = {}, {}, None, None, None
        for sf in sorted(cdir.glob("start*/paired.json")):
            d = json.loads(sf.read_text())
            case = d["case"]
            if d.get("status") != "DONE":
                errors[sf.parent.name] = (d.get("error") or "")[-300:]
                continue
            errors.update({f"{sf.parent.name}:{k}": v[-300:] for k, v in d.get("arm_errors", {}).items()})
            check = d.get("check", check)
            cfg = d.get("config", cfg)
            for b in d["blocks"]:
                for arm, v in b.items():
                    starts.setdefault(arm, {}).setdefault(sf.parent.name, []).append((v["one_ms"], v["other_ms"]))
        row = dict(name=cdir.name, mode=case and case["mode"], cell=case and case["cell"],
                   dtype=case and case["dtype"], width=case and case["width"], batch=case and case["batch"],
                   config=cfg, errors=errors, ratios={})
        frnn2 = [p[0] for s in starts.get("v1", starts.get("cudnn", {})).values() for p in s]
        row["frnn2_ms"] = sum(frnn2) / len(frnn2) if frnn2 else None
        for arm, per in starts.items():
            r = bootstrap(list(per.values()), hash((cdir.name, arm)) & 0xFFFF)
            if r:
                row["ratios"][arm] = dict(ratio=r[0], lo=r[1], hi=r[2],
                                          verdict="faster" if r[1] >= 1 else "slower" if r[2] <= 1 else "unresolved")
        if check:
            row["check"] = (check.get("max_rel_vs_frnn2") if "losses" in check
                            else {k: max(v.values()) for k, v in check.items()})
        rows.append(row)
    out = dict(rows=rows)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        f = lambda x: f"{x['ratio']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]"  # noqa: E731
        rs = "  ".join(f"{k} {f(v)}" for k, v in sorted(r["ratios"].items()))
        chk = {k: f"{v:.1e}" for k, v in (r.get("check") or {}).items()}
        ms = f"{r['frnn2_ms']:.2f}" if r["frnn2_ms"] else "-"
        print(f"{r['name']:28s} frnn2={ms} ms  {rs}  check={chk}  errors={list(r['errors'])}")


if __name__ == "__main__":
    main()
