"""Analysis of batch "b16pf" (docs/flashrnn2/max-width.md sections 65 and 66.2).

Usage: b16pf_analyze.py <results dir with control/ and science/> [--out FILE]
  P1     tasks/p1-bitwise-*: prefetch bitwise equal to the default (dG, d_initial, dR, dbias), every case
  R-pf   tasks/rpf-race-*: correct kernel 0 violations and bitwise stable; no_b1 and no_b2 detected, never silent
  W3     v1 (cuda_fused) / one (library default), training step, 95% hierarchical bootstrap (10,000 draws: starts,
         then blocks within the drawn starts); holds when the lower bound >= 1
  W4     v1cuda (cuda) / one, same interval and rule
  W6-pf  one (default) / alt (prefetch): the ratio printed is default time / prefetch time (> 1: prefetch faster);
         holds when the lower bound >= 1
  prof   sweep profile rows (report only)
A start whose outputs were not stable, or with an arm error, is listed and excluded from that arm's interval.
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
    sci = a.batch / "science"
    rec = dict(batch=str(a.batch), driver=json.loads((a.batch / "control/driver-result.json").read_text())["status"])
    p1 = [r for f in sorted(sci.glob("tasks/p1-bitwise-*/bitwise.json")) for r in json.loads(f.read_text())["rows"]]
    rec["P1"] = dict(cases=len(p1), passed=sum(r["pass_"] for r in p1),
                     failed=[(r["cell"], r["width"]) for r in p1 if not r["pass_"]])
    races = {}
    for f in sorted(sci.glob("tasks/rpf-race-*/race.json")):
        d = json.loads(f.read_text())
        races[f.parent.name] = dict(pass_=d.get("pass_"), summary=d.get("summary"), error=d.get("error"))
    rec["R-pf"] = races
    claims = []
    for cdir in sorted(sci.glob("claims/*")):
        starts = {"v1": [], "v1cuda": [], "alt": []}
        excluded = []
        case = None
        for sf in sorted(cdir.glob("start*/paired.json")):
            d = json.loads(sf.read_text())
            case = d["case"]
            if d.get("status") != "DONE":
                excluded.append((sf.parent.name, "status " + str(d.get("status"))))
                continue
            for arm in starts:
                if d.get("arm_errors", {}).get(arm) or d.get("outputs_stable", {}).get(arm) is False:
                    excluded.append((sf.parent.name, arm))
                    continue
                starts[arm].append([(b[arm]["one_ms"], b[arm]["other_ms"]) for b in d["blocks"] if arm in b])
        row = dict(name=cdir.name, cell=case["cell"], dtype=case["dtype"], width=case["width"], excluded=excluded)
        for arm, key in (("v1", "W3"), ("v1cuda", "W4")):
            r = bootstrap(starts[arm], hash((cdir.name, arm)) & 0xFFFF)
            row[key] = None if r is None else dict(ratio=r[0], lo=r[1], hi=r[2], holds=r[1] >= 1)
        # W6-pf: default / prefetch = one / alt, the inverse of the bootstrap's alt / one
        r = bootstrap([[(b, a_) for a_, b in s] for s in starts["alt"]], hash((cdir.name, "alt")) & 0xFFFF)
        row["W6-pf"] = None if r is None else dict(ratio=r[0], lo=r[1], hi=r[2], holds=r[1] >= 1)
        flat = [p for s in starts["v1"] for p in s]
        row["one_ms"] = sum(p[0] for p in flat) / len(flat) if flat else None
        claims.append(row)
    rec["claims"] = claims
    for key in ("W3", "W4", "W6-pf"):
        rec[key + "_holds"] = f"{sum(bool(c[key] and c[key]['holds']) for c in claims)}/{len(claims)}"
    prof = {}
    for f in sorted(sci.glob("sweep/*/sweep.json")):
        d = json.loads(f.read_text())
        prof[f.parent.name] = dict(arms={k: v.get("ms") for k, v in d.get("arms", {}).items()},
                                   profile=d.get("profile"))
    rec["prof"] = prof
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    for c in claims:
        f = lambda x: "-" if not x else f"{x['ratio']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]{'' if x['holds'] else ' (no)'}"  # noqa: E731
        print(f"{c['cell']:5s} {c['dtype'][:4]} D={c['width']:4d} one={c['one_ms']:.3f} ms  W3 {f(c['W3'])}  "
              f"W4 {f(c['W4'])}  W6-pf {f(c['W6-pf'])}  excl={c['excluded']}")
    print({k: rec[k] for k in ("driver", "P1", "W3_holds", "W4_holds", "W6-pf_holds")})
    print(json.dumps(races, default=str)[:1500])


if __name__ == "__main__":
    main()
