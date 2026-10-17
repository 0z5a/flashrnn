"""Analysis of the "mg" batch against docs/flashrnn2/scale-out-plan.md section 11 (F1, F2, G1-G3, N1).

Usage: scale_mg_analyze.py <batch dir with control/ and science/> [--out FILE]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scale_symm_analyze import bootstrap, load


def verdict(lo, hi, faster, slower):
    return faster if lo >= 1.02 else slower if hi <= 0.98 else "not resolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sci = a.batch / "science"
    rec = dict(batch=str(a.batch), driver=(load(a.batch / "control" / "driver-result.json") or {}).get("status"))
    # F1: three runs of the forward qualification.
    rows = defaultdict(list)
    for f in sorted(sci.glob("fwd-qualify-run*-k*/qualify.json")):
        run = f.parent.name.split("-")[2]
        for r in load(f)["configs"]:
            rows[(r["name"], r["cell"])].append(dict(r, run=run))
    digests = defaultdict(set)
    for key, rs in rows.items():
        for r in rs:
            if r.get("out_sha256"):
                digests[(r["cell"], r["width"], r["slice_width"])].add(r["out_sha256"])
    f1 = []
    for key, rs in sorted(rows.items()):
        r0 = rs[0]
        agree = len(digests[(r0["cell"], r0["width"], r0["slice_width"])]) == 1
        ok = len(rs) == 3 and all(r.get("pass_") for r in rs) and agree
        f1.append(dict(name=key[0], cell=key[1], runs=len(rs), pass_=ok, all_tool_pass=all(r.get("pass_") for r in rs),
                       digests_agree=agree, errors=[(r.get("error") or "")[-200:] for r in rs if r.get("error")],
                       fp64=[r.get("max_rel_vs_fp64") for r in rs],
                       symm_bitwise=[(r.get("symm") or {}).get("bitwise_vs_uva") for r in rs]))
    rec["F1"] = dict(passed=sum(x["pass_"] for x in f1), cases=len(f1), rows=f1)
    # F2
    starts, excl = defaultdict(list), defaultdict(list)
    for f in sorted(sci.glob("fwd-paired-k*-start*/paired.json")):
        x = load(f)
        for r in x["configs"]:
            key = (r["name"], r["cell"])
            if r.get("error") or not r.get("blocks") or not all(r["outputs_stable"].values()):
                excl[key].append(dict(start=x["start"], error=(r.get("error") or "")[-200:],
                                      stable=r.get("outputs_stable")))
                continue
            starts[key].append([(b["uva_ms"], b["nccl_symm_ms"]) for b in r["blocks"]])
    f2 = []
    for i, key in enumerate(sorted(set(starts) | set(excl))):
        res = bootstrap(starts.get(key, []), 20261019 + i)
        row = dict(name=key[0], cell=key[1], starts=len(starts.get(key, [])), excluded=excl.get(key, []))
        if res:
            row.update(ratio_uva_over_symm=round(res[0], 4), lo=round(res[1], 4), hi=round(res[2], 4),
                       verdict=verdict(res[1], res[2], "symm faster", "uva faster"))
        else:
            row["verdict"] = "no data"
        f2.append(row)
    rec["F2"] = dict(rows=f2, counts={v: sum(r["verdict"] == v for r in f2)
                                      for v in ("symm faster", "uva faster", "not resolved", "no data")})
    # G1
    g1 = []
    for f in sorted(sci.glob("bw-qualify*/qualify.json")):
        for c in load(f)["cases"]:
            g1.append(dict({k: c.get(k) for k in ("cell", "width", "peers", "batch", "batch_rows", "pass_")},
                           uva={k: (c.get("uva") or {}).get(k) for k in ("dg_bitwise", "d_init_bitwise", "error")},
                           symm={k: (c.get("nccl_symm") or {}).get(k) for k in ("dg_bitwise", "d_init_bitwise", "error")},
                           error=(c.get("error") or "")[-300:] or None))
    rec["G1"] = dict(passed=sum(bool(x["pass_"]) for x in g1), cases=len(g1), rows=g1)
    # G2
    g2 = []
    for f in sorted(sci.glob("bw-race*/race.json")):
        for c in load(f)["cases"]:
            g2.append(dict({k: c.get(k) for k in ("cell", "width", "peers", "batch", "batch_rows", "pass_")},
                           variants={k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                                     for k, v in c.get("variants", {}).items()},
                           error=(c.get("error") or "")[-300:] or None))
    rec["G2"] = dict(passed=sum(bool(x["pass_"]) for x in g2), cases=len(g2), rows=g2)
    # G3
    st = defaultdict(lambda: defaultdict(list))
    ex = defaultdict(list)
    for f in sorted(sci.glob("bw-paired-start*/paired.json")):
        x = load(f)
        for c in x["cases"]:
            key = json.dumps({k: c[k] for k in ("cell", "width", "peers", "batch", "steps", "batch_rows")},
                             sort_keys=True)
            if c.get("error") or not c.get("blocks") or not all(c["outputs_stable"].values()):
                ex[key].append(dict(start=x["start"], error=(c.get("error") or "")[-200:], stable=c.get("outputs_stable")))
                continue
            st[key]["single_vs_mg"].append([(b["single_ms"], b["mg_uva_ms"]) for b in c["blocks"]])
            st[key]["uva_vs_symm"].append([(b["mg_uva_ms"], b["mg_symm_ms"]) for b in c["blocks"]])
    g3 = []
    for i, key in enumerate(sorted(set(st) | set(ex))):
        row = dict(case=json.loads(key), excluded=ex.get(key, []))
        for name, (fast, slow) in (("single_vs_mg", ("multi-GPU faster", "multi-GPU slower")),
                                   ("uva_vs_symm", ("symm faster", "uva faster"))):
            res = bootstrap(st[key][name], 20261020 + 2 * i + (name == "uva_vs_symm")) if key in st else None
            row[name] = (dict(ratio=round(res[0], 4), lo=round(res[1], 4), hi=round(res[2], 4),
                              verdict=verdict(res[1], res[2], fast, slow)) if res else dict(verdict="no data"))
        g3.append(row)
    rec["G3"] = g3
    # N1
    n1 = []
    for f in sorted(sci.glob("nvshmem-p*/nvshmem.json")):
        x = load(f)
        for r in x.get("rows", []):
            ms = sorted(r.get("nvshmem_ms") or [])
            n1.append(dict(task=f.parent.name, case=r["case"], pass_=r.get("pass_"), bitwise=r.get("bitwise"),
                           nvshmem_ms_median=ms[len(ms) // 2] if ms else None,
                           reference_ms_median=r.get("reference_ms_median"), reference=r.get("reference"),
                           error=(r.get("error") or "")[-300:] or None))
        if not x.get("rows"):
            n1.append(dict(task=f.parent.name, pass_=False, workers=x.get("workers")))
    rec["N1"] = dict(passed=sum(bool(x["pass_"]) for x in n1), cases=len(n1), rows=n1)
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(json.dumps(dict(driver=rec["driver"], F1=f'{rec["F1"]["passed"]}/{rec["F1"]["cases"]}', F2=rec["F2"]["counts"],
                          G1=f'{rec["G1"]["passed"]}/{rec["G1"]["cases"]}', G2=f'{rec["G2"]["passed"]}/{rec["G2"]["cases"]}',
                          G3=[(r["case"], r["single_vs_mg"]["verdict"], r["uva_vs_symm"]["verdict"]) for r in g3],
                          N1=f'{rec["N1"]["passed"]}/{rec["N1"]["cases"]}')))


if __name__ == "__main__":
    main()
