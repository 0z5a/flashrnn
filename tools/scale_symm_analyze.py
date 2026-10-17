"""Analysis of the symmetric-buffer batch against docs/flashrnn2/scale-out-plan.md section 9.

Usage: scale_symm_analyze.py <batch dir with control/ and science/> [--out FILE]
  preflight  NCCL versions, multicast support, symmetric session probe (reported)
  micro      pair 0-1 against the median pair at 16 B, without and with warm-up (repair 2 hypothesis), uva vs symm
  S1         qualification per configuration and cell (rule of section 9, including cross-P digest agreement)
  S2         uva / nccl_symm per configuration and cell: 95% hierarchical bootstrap (10,000 draws, starts then
             blocks); symmetric faster if lo >= 1.02, uva faster if hi <= 0.98, else not resolved
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

DRAWS = 10000


def bootstrap(starts, seed):
    starts = [s for s in starts if s]
    if not starts:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(DRAWS):
        num = den = 0.0
        for s in (rng.choice(starts) for _ in starts):
            for u, m in (rng.choice(s) for _ in s):
                num += u
                den += m
        ratios.append(num / den)
    ratios.sort()
    flat = [p for s in starts for p in s]
    return sum(u for u, _ in flat) / sum(m for _, m in flat), ratios[int(0.025 * DRAWS)], ratios[int(0.975 * DRAWS) - 1]


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def micro_summary(micro, alloc):
    pairs = (micro or {}).get("pair_by_alloc", {}).get(alloc) or ((micro or {}).get("pair") if alloc == "uva" else None)
    if not pairs:
        return None
    a16 = {k: v["16"]["one_way_us"] for k, v in pairs.items()}
    med = statistics.median(a16.values())
    first = a16.get("0-1")
    return dict(alpha16_us=a16, median_us=med, pair01_over_median=first / med if first and med else None,
                GBps_1MB={k: v["bandwidth_GBps_at_1MB"] for k, v in pairs.items()},
                barrier_errors=sum(x.get("barrier_error", 0) for v in pairs.values() for x in v.values()
                                   if isinstance(x, dict)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sci = a.batch / "science"
    rec = dict(batch=str(a.batch), driver=(load(a.batch / "control" / "driver-result.json") or {}).get("status"))
    rec["preflight"] = load(sci / "preflight-symm" / "symm.json")
    nowarm = load(sci / "micro-nowarm" / "micro.json")
    warm = load(sci / "micro" / "micro.json")
    rec["micro"] = dict(nowarm_uva=micro_summary(nowarm, "uva"), warm_uva=micro_summary(warm, "uva"),
                        warm_symm=micro_summary(warm, "nccl_symm"), errors=dict(nowarm=(nowarm or {}).get("errors"),
                                                                                  warm=(warm or {}).get("errors")))
    nw, w = rec["micro"]["nowarm_uva"], rec["micro"]["warm_uva"]
    rec["micro"]["repair2_hypothesis_supported"] = bool(
        nw and w and nw["pair01_over_median"] and w["pair01_over_median"]
        and nw["pair01_over_median"] >= 3 and w["pair01_over_median"] <= 1.5)
    # S1
    rows = []
    for f in sorted(sci.glob("qualify-k*/qualify.json")):
        rows += load(f)["configs"]
    digests = defaultdict(set)
    for r in rows:
        if r.get("out_sha256"):
            digests[(r["cell"], r["width"], r["slice_width"])].add(r["out_sha256"])
    s1 = []
    for r in rows:
        key = (r["cell"], r["width"], r["slice_width"])
        sym = r.get("symm") or {}
        cross_p = len(digests[key]) == 1
        ok = bool(r.get("pass_")) and cross_p
        s1.append(dict(name=r["name"], cell=r["cell"], pass_=ok, tool_pass=r.get("pass_"), cross_p_agree=cross_p,
                       bitwise_vs_p1=r.get("bitwise_vs_p1"), symm_bitwise_vs_uva=sym.get("bitwise_vs_uva"),
                       symm_error=(sym.get("error") or "")[-300:] or None, fp64=r.get("max_rel_vs_fp64"),
                       warmup_error=(r.get("warmup_error") or "")[-300:] or None,
                       error=(r.get("error") or "")[-300:] or None,
                       bitwise_vs_single_gpu=r.get("bitwise_vs_single_gpu")))
    rec["S1"] = dict(passed=sum(x["pass_"] for x in s1), cases=len(s1), rows=s1)
    # S2
    starts = defaultdict(list)
    unstable = defaultdict(list)
    arms_bitwise = defaultdict(list)
    for f in sorted(sci.glob("paired-k*-start*/paired.json")):
        x = load(f)
        for r in x["configs"]:
            key = (r["name"], r["cell"])
            if r.get("error") or not r.get("blocks"):
                unstable[key].append(dict(start=x["start"], error=(r.get("error") or "no blocks")[-200:]))
                continue
            if not all(r["outputs_stable"].values()):
                unstable[key].append(dict(start=x["start"], outputs_stable=r["outputs_stable"]))
                continue
            arms_bitwise[key].append(r.get("arms_bitwise"))
            starts[key].append([(b["uva_ms"], b["nccl_symm_ms"]) for b in r["blocks"]])
    s2 = []
    for i, key in enumerate(sorted(set(starts) | set(unstable))):
        res = bootstrap(starts.get(key, []), 20261018 + i)
        row = dict(name=key[0], cell=key[1], starts=len(starts.get(key, [])), excluded=unstable.get(key, []),
                   arms_bitwise=arms_bitwise.get(key))
        if res:
            row.update(ratio_uva_over_symm=round(res[0], 4), lo=round(res[1], 4), hi=round(res[2], 4),
                       verdict="symm faster" if res[1] >= 1.02 else "uva faster" if res[2] <= 0.98 else "not resolved")
        else:
            row.update(verdict="no data")
        s2.append(row)
    rec["S2"] = dict(rows=s2, counts={v: sum(r["verdict"] == v for r in s2)
                                      for v in ("symm faster", "uva faster", "not resolved", "no data")})
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(json.dumps(dict(driver=rec["driver"], symm_ok=(rec["preflight"] or {}).get("symm_ok"),
                          repair2=rec["micro"]["repair2_hypothesis_supported"],
                          S1=f'{rec["S1"]["passed"]}/{rec["S1"]["cases"]}', S2=rec["S2"]["counts"])))


if __name__ == "__main__":
    main()
