"""Analysis of a scale-out phase A batch against the pre-registered checks (docs/flashrnn2/scale-out-plan.md).

Usage: scale_analyze.py <batch dir with science/> --out FILE
Reads science/micro/micro.json, science/qualify-*/qualify.json, science/time-*/time.json, science/heads/heads.json.
  check 1  qualification: every configuration passes (bitwise vs P = 1, vs the single-GPU kernel where it exists,
           finite, FP64 within 2e-2)
  check 2  model: measured per-step cost within +-20% of the prediction; predicted schedule ordering holds for every
           pair the model separates by more than 10%
  check 3  crossover: resident multi-GPU configuration against the one-GPU streamed baseline at the same D; accepted
           for a D when the measured ratio >= 1.1 and the model predicted >= 1.0
Also tabulates alpha/beta per pair, the per-step phases, and the many-head rows.
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows += d.get("configs", d.get("rows", []))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    sci = a.batch / "science"
    micro = json.loads((sci / "micro" / "micro.json").read_text()) if (sci / "micro" / "micro.json").exists() else {}
    qual = load(glob.glob(str(sci / "qualify-*" / "qualify.json")))
    timing = load(glob.glob(str(sci / "time-*" / "time.json")))
    heads = load(glob.glob(str(sci / "heads" / "heads.json")))
    rec = dict(batch=str(a.batch))
    # Microbenchmarks.
    if micro:
        rec["links"] = {k: dict(alpha_us=round(v["alpha_us"], 3), GBps_1MB=round(v["bandwidth_GBps_at_1MB"], 1))
                        for k, v in micro.get("pair", {}).items()}
        rec["hbm_GBps"] = {k: round(v["copy_GBps"]) for k, v in micro.get("hbm", {}).items()}
        rec["pcie_GBps"] = {k: round(v["h2d_GBps"], 1) for k, v in micro.get("pcie", {}).items()}
    # Check 1.
    q_rows = [dict(name=r["name"], cell=r["cell"], pass_=r.get("pass_"), bitwise_single=r.get("bitwise_vs_single_gpu"),
                   bitwise_p1=r.get("bitwise_vs_p1"), fp64=r.get("max_rel_vs_fp64"), error=(r.get("error") or "")[-200:])
              for r in qual]
    rec["check1_qualification"] = dict(configs=len(q_rows), passed=sum(bool(r["pass_"]) for r in q_rows),
                                       failing=[r for r in q_rows if not r["pass_"]],
                                       pass_=bool(q_rows) and all(r["pass_"] for r in q_rows))
    # Check 2.
    model_rows, within, orderings = [], 0, []
    by_key = defaultdict(dict)
    for r in timing:
        if "per_step_us" not in r:
            continue
        pred = r["model"]["total_us"]
        ratio = r["per_step_us"] / pred if pred else None
        ok = ratio is not None and 0.8 <= ratio <= 1.2
        within += ok
        model_rows.append(dict(name=r["name"], cell=r["cell"], measured_us=round(r["per_step_us"], 3),
                               predicted_us=round(pred, 3), ratio=round(ratio, 3) if ratio else None, within_20pct=ok,
                               phases_us={k: round(v, 3) for k, v in r["phases_mean_us"].items()}))
        by_key[(r["cell"], r["peers"], r["width"], r["slice_width"], r["rmode"])][(r["schedule"], r["stride"])] = (r["per_step_us"], pred)
    for key, scheds in by_key.items():
        items = list(scheds.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (sa, ma), (sb, mb) = items[i], items[j]
                if min(ma[1], mb[1]) <= 0 or abs(ma[1] - mb[1]) / min(ma[1], mb[1]) <= 0.10:
                    continue  # the model does not separate them
                predicted_faster = sa if ma[1] < mb[1] else sb
                measured_faster = sa if ma[0] < mb[0] else sb
                orderings.append(dict(key=list(map(str, key)), pair=[list(sa), list(sb)],
                                      predicted_faster=list(predicted_faster), measured_faster=list(measured_faster),
                                      holds=predicted_faster == measured_faster))
    rec["check2_model"] = dict(configs=len(model_rows), within_20pct=within, rows=model_rows, orderings=orderings,
                               orderings_hold=sum(o["holds"] for o in orderings),
                               pass_=bool(model_rows) and within == len(model_rows) and all(o["holds"] for o in orderings))
    # Check 3: resident multi-GPU vs one-GPU streamed at the same D (per cell).
    cross = []
    for r in timing:
        if r.get("rmode") != "smem" or r.get("peers", 1) == 1 or "per_step_us" not in r:
            continue
        base = [b for b in timing if b.get("rmode") == "stream" and b.get("peers") == 1 and b.get("width") == r["width"]
                and b.get("cell") == r["cell"] and "per_step_us" in b]
        if not base:
            continue
        b = base[0]
        measured = b["per_step_us"] / r["per_step_us"]
        predicted = b["model"]["total_us"] / r["model"]["total_us"] if r["model"]["total_us"] else None
        cross.append(dict(name=r["name"], cell=r["cell"], width=r["width"], baseline=b["name"],
                          measured_ratio=round(measured, 3), predicted_ratio=round(predicted, 3) if predicted else None,
                          accepted=measured >= 1.1 and (predicted or 0) >= 1.0))
    accepted_widths = sorted({c["width"] for c in cross if c["accepted"]})
    rec["check3_crossover"] = dict(rows=cross, accepted_widths=accepted_widths)
    # Heads.
    rec["heads"] = [dict(cell=r["cell"], width=r["width"], heads=r["heads"], batch=r["batch"],
                         forward_ms=r.get("forward_ms"), train_step_ms=r.get("train_step_ms"),
                         forward_ms_heads_over_gpus=r.get("forward_ms_heads_over_gpus"), error=(r.get("error") or "")[-120:])
                    for r in heads]
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(json.dumps(dict(check1=rec["check1_qualification"]["pass_"], check2=rec["check2_model"]["pass_"],
                          within=f'{within}/{len(model_rows)}', orderings=f'{rec["check2_model"]["orderings_hold"]}/{len(orderings)}',
                          crossover_accepted=accepted_widths)))


if __name__ == "__main__":
    main()
