"""Summarize the fused-preactivation backward batch: admission of each fused level, qualification errors against the
FP64 oracle (and FlashRNN1's), race and sanitizer results, the D = 384/512 breakdown, selections, and the paired
timing (tools/rtiled_m4_analyze.py output) evaluated against the criterion pre-registered in plan.json.

Usage: summarize_fuse.py <batch-dir with control/ and science/> --timing <m4-summary.json> [--out FILE]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def qualification(batch):
    out = defaultdict(lambda: dict(cases=0, pass_=0, gates=defaultdict(lambda: defaultdict(int)), worst={},
                                   ours_over_v1=0.0, v1_cases=0, failing=[]))
    for f in sorted(batch.glob("science/phase0/qualify/*/bw.json")):
        rec = json.loads(f.read_text())
        c = rec["case"]
        level = out[f"fuse{c.get('fuse', 0)}"]
        level["cases"] += 1
        level["pass_"] += rec.get("status") == "PASS"
        for g, v in rec.get("gates", {}).items():
            level["gates"][g][v] += 1
        key = f"{c['dtype']}"
        for loss in ("hist", "full"):
            for name, m in rec.get("grads", {}).get(loss, {}).get("metrics", {}).items():
                level["worst"][key] = max(level["worst"].get(key, 0.0), m["rel_fro"])
        if rec.get("v1", {}).get("status") == "AVAILABLE":
            level["v1_cases"] += 1
            ours = rec["grads"]["hist"]["metrics"]
            for name, m in rec["v1"]["metrics"].items():
                level["ours_over_v1"] = max(level["ours_over_v1"], ours[name]["rel_fro"] / max(m["rel_fro"], 1e-300))
        if rec.get("status") != "PASS":
            level["failing"].append(dict(name=f.parent.name, gates=rec.get("gates"),
                                         errors={k: v[-300:] for k, v in rec.items() if k.endswith("_error")}))
    return {k: dict(v, gates={g: dict(x) for g, x in v["gates"].items()}) for k, v in out.items()}


def race(batch):
    rows = []
    for f in sorted(batch.glob("science/phase0/race/*/race.json")):
        rec = json.loads(f.read_text())
        for group, s in rec.get("summary", {}).items():
            rows.append(dict(name=f.parent.name, group=group, runs=s["runs"], flagged=s["flagged"],
                             outputs_differ=s["outputs_differ"], silent=s["silent_corruption"], errors=s["errors"],
                             pass_=s["pass_"]))
    return rows


def profile(batch):
    rows = []
    for f in sorted(batch.glob("science/phase0/profile/*/profile.json")):
        rec = json.loads(f.read_text())
        phases = rec.get("bw_phases", {})

        def best(pred):
            names = [n for n in phases if pred(n)]
            return min(names, key=lambda n: phases[n]["total"]) if names else None
        bt, bf = best(lambda n: n.endswith("_fast_tf32")), best(lambda n: "_fuse" in n)
        rows.append(dict(case=rec["case"], v1_step=rec.get("v1_step"), step=rec.get("step"),
                         best_unfused=(bt, phases.get(bt)), best_fused=(bf, phases.get(bf)),
                         errors=list(rec.get("errors", {}))))
    return rows


def selections(batch):
    rows = []
    for f in sorted(batch.glob("science/phase1/*/select.json")):
        rec = json.loads(f.read_text())
        rows.append(dict(case=rec["case"]["name"],
                         selected={r: v.get("selected") for r, v in rec.get("candidates", {}).items()},
                         not_qualified={r: [k for k, q in v.get("qualification", {}).items() if q["status"] != "PASS"]
                                        for r, v in rec.get("candidates", {}).items()}))
    return rows


def criterion(timing, plan):
    rows = timing["rows"]
    widths = sorted({c["width"] for c in plan["select_cases"]})
    verdict = {}
    for env in ("tl", "mod"):
        cells = {}
        for r in rows:
            if r["experiment"] == "TRAIN_STEP_FRNN2_VS_V1" and r["batch"] == 16 and f"fuse-{env}-" in r["name"]:
                cells[(r["cell"], r["width"])] = r["ci_low"]
        complete = all((c, w) in cells for c in ("lstm", "slstm") for w in widths)
        verdict[f"b16_{env}"] = dict(complete=complete, all_lower_ge_1=complete and all(v >= 1.0 for v in cells.values()),
                                     lower_bounds={f"{c}-d{w}": round(v, 4) for (c, w), v in sorted(cells.items())})
    b64 = {(r["cell"], r["width"]): r["ci_low"] for r in rows
           if r["experiment"] == "TRAIN_STEP_FRNN2_VS_V1" and r["batch"] == 64}
    verdict["b64_no_regression"] = dict(all_lower_ge_1=len(b64) == 2 * len(widths) and all(v >= 1.0 for v in b64.values()),
                                        lower_bounds={f"{c}-d{w}": round(v, 4) for (c, w), v in sorted(b64.items())})
    verdict["b16_fixed"] = any(verdict[f"b16_{e}"]["all_lower_ge_1"] for e in ("tl", "mod")) and \
        verdict["b64_no_regression"]["all_lower_ge_1"]
    fusion = defaultdict(dict)
    for r in rows:
        if r["experiment"] == "TRAIN_STEP_FUSED_VS_UNFUSED":
            env = "tl" if "fuse-tl-" in r["name"] else "mod"
            fusion[r["width"]][f"{env}-b{r['batch']}-{r['cell']}"] = (round(r["speedup"], 4), round(r["ci_low"], 4),
                                                                      round(r["ci_high"], 4))
    default = {}
    for w in widths:
        f = fusion.get(w, {})
        b16 = all(f.get(f"tl-b16-{c}", (0, 0, 0))[1] >= 1.0 for c in ("lstm", "slstm"))
        b64 = all(f.get(f"mod-b64-{c}", (0, 0, 0))[2] >= 1.0 for c in ("lstm", "slstm"))
        default[w] = "all batch sizes" if b16 and b64 else ("B <= 16" if b16 else "not default")
    return verdict, dict(fusion), default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    plan = json.loads((a.batch / "control/plan.json").read_text())
    driver = json.loads((a.batch / "control/driver-result.json").read_text())
    timing = json.loads(a.timing.read_text())
    sanitizer = [dict(name=r.get("name"), clean=r.get("clean")) for r in driver["phases"]["phase0"]["records"]
                 if "/sanitizer/" in str(r.get("out", ""))]
    verdict, fusion, default = criterion(timing, plan)
    rec = dict(driver_status=driver.get("status"), admission=driver.get("fused_admission"),
               qualification=qualification(a.batch), race=race(a.batch), sanitizer=sanitizer,
               profile=profile(a.batch), selections=selections(a.batch),
               timing=dict(complete=timing["comparisons"], incomplete=len(timing["incomplete"]),
                           unavailable=len(timing["unavailable"])),
               criterion=verdict, fusion_effect=fusion, fuse_default_rule=default)
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
