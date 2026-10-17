"""Analysis of the many-heads batch against docs/flashrnn2/heads.md (checks H1-H5).

Usage: heads_analyze.py <batch dir with control/ and science/> [--out FILE]
  H1  qualification per path (every case passes)
  H2  forward: loop_graph / one at the claim shapes, 95% hierarchical bootstrap (10,000 draws: starts, then blocks
      within the drawn starts); holds for a shape when the lower bound >= 1
  H3  forward and training: v1 / one, same interval and rule
  H4  BF16 training at H = 1: v1 / one, same rule
  H5  sweep rows (per-call median ms of every arm against H), reported only
The ratio of a draw is mean(other) / mean(one) over the drawn blocks. A start whose outputs were not stable
(bitwise against the arm's first call) is reported and excluded from the interval.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DRAWS = 10000


def bootstrap(starts, seed):
    """starts: list of lists of (one_ms, other_ms) block pairs. Returns (point, lo, hi)."""
    starts = [s for s in starts if s]
    if not starts:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(DRAWS):
        one = other = 0.0
        for s in (rng.choice(starts) for _ in starts):
            for a, b in (rng.choice(s) for _ in s):
                one += a
                other += b
        ratios.append(other / one)
    ratios.sort()
    flat = [p for s in starts for p in s]
    point = sum(b for _, b in flat) / sum(a for a, _ in flat)
    return point, ratios[int(0.025 * DRAWS)], ratios[int(0.975 * DRAWS) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sci = a.batch / "science"
    plan = json.loads((a.batch / "control" / "plan.json").read_text())
    driver = json.loads((a.batch / "control" / "driver-result.json").read_text())
    rec = dict(batch=str(a.batch), driver_status=driver.get("status"))
    # H1
    q_rows = []
    for q in plan["qualify"]:
        f = sci / "qualify" / q["name"] / "qualify.json"
        r = json.loads(f.read_text()) if f.exists() else {}
        q_rows.append(dict(name=q["name"], path=q["path"], pass_=bool(r.get("pass_")),
                           error=(r.get("error") or "")[-300:] or None,
                           forward_fp64=(r.get("forward") or {}).get("fp64"),
                           forward_bitwise=(r.get("forward") or {}).get("bitwise_ok"),
                           training_bitwise=(r.get("training") or {}).get("bitwise_ok"),
                           training_fp64_ok=(r.get("training") or {}).get("fp64_ok"),
                           d_r_bitwise=[h.get("d_r_bitwise") for h in (r.get("training") or {}).get("per_head", [])]))
    by_path = defaultdict(list)
    for r in q_rows:
        by_path[r["path"]].append(r)
    rec["H1"] = {p: dict(passed=sum(r["pass_"] for r in v), cases=len(v), pass_=all(r["pass_"] for r in v),
                         failing=[r for r in v if not r["pass_"]]) for p, v in by_path.items()}
    # Repro (diagnostic)
    rec["repro"] = []
    for r in plan["repro"]:
        f = sci / "repro" / r["name"] / "repro.json"
        x = json.loads(f.read_text()) if f.exists() else {}
        rec["repro"].append(dict(name=r["name"], status=x.get("status"), error=(x.get("error") or "")[-400:] or None))
    # H2-H4
    claims = []
    for i, c in enumerate(plan["claims"]):
        starts = defaultdict(list)
        unstable, statuses = [], []
        for s in range(plan["starts"]):
            f = sci / "claims" / c["name"] / f"start{s}" / "paired.json"
            if not f.exists():
                statuses.append("missing")
                continue
            x = json.loads(f.read_text())
            statuses.append(x.get("status"))
            stable = x.get("outputs_stable", {})
            for other in [n for n in c["arms"] if n != "one"]:
                if not stable.get("one", False) or not stable.get(other, False):
                    unstable.append(dict(start=s, arm=other))
                    continue
                pairs = [(b[other]["one_ms"], b[other]["other_ms"]) for b in x.get("blocks", []) if other in b]
                starts[other].append(pairs)
        for other in [n for n in c["arms"] if n != "one"]:
            res = bootstrap(starts[other], 20261017 + i)
            if c.get("experiment", "").startswith("W"):  # large-D batches: W3 against cuda_fused, W4 against cuda
                check = ("W3" if other == "v1" else "W4" if other == "v1cuda" else "W6" if other == "alt" else other) \
                    + c["experiment"][4:]
            else:
                check = "H2" if other == "loop_graph" else "H4" if c["experiment"] == "H4" else "H3" if other == "v1" \
                    else "H6" if other == "alt" else "loop"
            row = dict(name=c["name"], check=check, kind=c["kind"], cell=c["cell"], dtype=c["dtype"], width=c["width"],
                       heads=c["heads"], batch=c["batch"], steps=c["steps"], other=other, starts=statuses,
                       unstable=unstable)
            if res:
                row.update(ratio=round(res[0], 4), lo=round(res[1], 4), hi=round(res[2], 4), holds=res[1] >= 1.0)
            else:
                row.update(ratio=None, holds=False)
            claims.append(row)
    rec["claims"] = claims
    for check in sorted({r["check"] for r in claims} | {"H2", "H3", "H4", "loop"}):
        rows = [r for r in claims if r["check"] == check]
        rec[check] = dict(shapes=len(rows), holds=sum(r["holds"] for r in rows), measured=sum(r["ratio"] is not None for r in rows))
    # H5
    sweep = []
    for c in plan["sweep"]:
        f = sci / "sweep" / c["name"] / "sweep.json"
        if not f.exists():
            continue
        x = json.loads(f.read_text())
        sweep.append(dict(kind=c["kind"], cell=c["cell"], dtype=c["dtype"], width=c["width"], heads=c["heads"],
                          batch=c["batch"], status=x.get("status"), config=x.get("config"),
                          arms={k: v.get("ms") if "ms" in v else "error" for k, v in x.get("arms", {}).items()},
                          loop_bitwise={k: v.get("bitwise_vs_one") for k, v in x.get("arms", {}).items()
                                        if "bitwise_vs_one" in v}))
    rec["H5_sweep"] = sweep
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(json.dumps(dict(H1={p: f'{v["passed"]}/{v["cases"]}' for p, v in rec["H1"].items()},
                          checks={k: rec[k] for k in sorted({r["check"] for r in claims})},
                          repro={r["name"]: r["status"] for r in rec["repro"]}, sweep_rows=len(sweep))))


if __name__ == "__main__":
    main()
