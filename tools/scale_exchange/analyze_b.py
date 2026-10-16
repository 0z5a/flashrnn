"""Analysis of the scale-out phase B batch against the pre-registered checks (docs/flashrnn2/scale-out-plan.md §7).

Usage: analyze_b.py <batch dir with control/ and science/> [--out FILE]

Written before any phase B result. Model parameters come only from the micro launches of the same backend:
  repeat:R at P = 2 within a node  -> o_nv, alpha_nv, beta_nv(1)   (per_step = o + R*alpha + R*s*beta, least squares)
  repeat:R at P = 2 across nodes   -> o_net, alpha_net, beta_net
  flat at P = 3 / 4 within a node  -> beta_nv(2) / beta_nv(3)      (slope of per_step over the slice sweep)
  fanout:k at P = 8, s in {16, 8192} -> g = slope_k(per_step) - beta_net*s, averaged
The per-step overhead o of a two-node prediction is o_net (same launch shape: GIN flush / NVSHMEM quiet present).
Prediction: per_step = o + gamma + sum_k max_i max(alpha_nv + beta_nv(n)*b_local_max,
                                                  alpha_net + (r - 1)*g + beta_net*B_net)   (absent terms dropped)
over the schedule's rounds k and ranks i (the Python port of schedule.h below, cross-checked against the
schedule_check loads written by the batch preflight).
Checks: B1 correctness, B2 model (+-20 %, ordering where the model separates by > 10 %), B3 topology claim
(ratio >= 1.1 measured, >= 1.0 predicted), B4 against the host all-gather (graph) (ratio >= 1.1), B5 additivity
(|per_step(gamma) - gamma - per_step(0)| <= max(10 %, 1 us)).
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

CLAIM8 = (8192, 16384)
CLAIM7 = (9376, 18736)
DEVICE = ("gin", "shmem-ibrc", "shmem-ibgda")


# ---- Python port of schedule.h -------------------------------------------------------------------------------------
def build(sched, stride, P, L, rank):
    nodes = (P + L - 1) // L
    q = P.bit_length() - 1
    nround = dict(flat=1, cyclic=P - 1, ring=P - 1, doubling=q, hier=2, fanout=1, repeat=stride)[sched]
    rounds = [[] for _ in range(nround)]

    def add(r, peer, block, n):
        rounds[r].append((peer, block, n))

    from math import gcd
    if sched == "flat":
        if gcd(stride, P) != 1:
            return None
        for k in range(1, P):
            add(0, (rank + k * stride) % P, rank, 1)
    elif sched == "cyclic":
        if gcd(stride, P) != 1:
            return None
        for k in range(1, P):
            add(k - 1, (rank + k * stride) % P, rank, 1)
    elif sched == "ring":
        for k in range(P - 1):
            add(k, (rank + 1) % P, (rank - k) % P, 1)
    elif sched == "doubling":
        if P & (P - 1):
            return None
        for k in range(q):
            add(k, rank ^ (1 << k), rank & ~((1 << k) - 1), 1 << k)
    elif sched == "hier":
        if P % L or nodes < 2:
            return None
        local, node = rank % L, rank // L
        for n in range(nodes):
            if n != node:
                add(0, n * L + local, rank, 1)
        for j in range(L):
            if j != local:
                for n in range(nodes):
                    add(1, node * L + j, n * L + local, 1)
    elif sched == "fanout":
        if P % L or nodes < 2 or not 1 <= stride <= L:
            return None
        local, nn = rank % L, (rank // L + 1) % nodes
        for j in range(stride):
            add(0, nn * L + (local + j) % L, rank, 1)
    elif sched == "repeat":
        for k in range(stride):
            add(k, (rank + 1) % P, rank, 1)
    else:
        raise ValueError(sched)
    return rounds


def loads(sched, stride, P, L):
    """Per round, per rank: (distinct same-node peers, max blocks to one same-node peer, network sends, network
    blocks). Same-node cost is charged per peer link (amendment of 2026-10-16 21:50, before any result)."""
    tabs = [build(sched, stride, P, L, r) for r in range(P)]
    if tabs[0] is None:
        return None
    out = []
    for k in range(len(tabs[0])):
        per = []
        for i in range(P):
            sends = tabs[i][k] if k < len(tabs[i]) else []
            per_peer = defaultdict(int)
            for peer, _, n in sends:
                if peer // L == i // L:
                    per_peer[peer] += n
            net = [n for peer, _, n in sends if peer // L != i // L]
            per.append((len(per_peer), max(per_peer.values(), default=0), len(net), sum(net)))
        out.append(per)
    return out


def crosscheck_loads(path):
    """Compare the port's per-schedule maxima with schedule_check's JSON (written by the batch preflight)."""
    bad, n = [], 0
    for row in json.loads(Path(path).read_text()):
        if row["P"] > 16:
            continue
        ld = loads(row["schedule"], row["stride"], row["P"], row["L"])
        if ld is None:
            bad.append(row)
            continue
        n += 1
        net_msgs = max(p[2] for rnd in ld for p in rnd)
        net_blocks = max(p[3] for rnd in ld for p in rnd)
        total_net = sum(p[3] for rnd in ld for p in rnd)
        if (len(ld), net_msgs, net_blocks, total_net) != (row["rounds"], row["max_net_msgs_per_nic_round"],
                                                          row["max_net_blocks_per_nic_round"], row["net_blocks_total"]):
            bad.append(row)
    return dict(compared=n, mismatches=len(bad), first=bad[:3])


# ---- fits ------------------------------------------------------------------------------------------------------------
def lstsq(X, y):
    """Ordinary least squares via normal equations (small systems, no numpy on the analysis host needed)."""
    m = len(X[0])
    A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(m)] for i in range(m)]
    b = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(m)]
    for c in range(m):  # Gauss-Jordan
        p = max(range(c, m), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(m):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [A[r][j] - f * A[c][j] for j in range(m)]
                b[r] -= f * b[c]
    coef = [b[i] / A[i][i] for i in range(m)]
    pred = [sum(c * x for c, x in zip(coef, row)) for row in X]
    ss_res = sum((a - p) ** 2 for a, p in zip(y, pred))
    mean = sum(y) / len(y)
    ss_tot = sum((a - mean) ** 2 for a in y) or 1.0
    return coef, 1 - ss_res / ss_tot


def fit_backend(rows):
    """rows: case dicts of one backend. Returns the model parameters (microseconds, us per byte)."""
    par, notes = {}, []
    for cls, tag in (("nv", "micro-intra2"), ("net", "micro-cross2")):
        r = [x for x in rows if x["tag"] == tag and x["schedule"] == "repeat" and x["per_step_us"] > 0]
        if len(r) < 4:
            notes.append(f"{tag}: {len(r)} rows")
            continue
        (o, a, b), r2 = lstsq([[1, x["stride"], x["stride"] * x["slice_bytes"]] for x in r], [x["per_step_us"] for x in r])
        par[f"o_{cls}"], par[f"alpha_{cls}"], par[f"beta_{cls}"], par[f"r2_{cls}"] = o, a, b, r2
    par["beta_nv_n"] = {1: par.get("beta_nv")}
    for n, tag in ((2, "micro-intra3"), (3, "micro-intra4")):
        r = [x for x in rows if x["tag"] == tag and x["per_step_us"] > 0]
        if len(r) >= 3:
            (c, b), r2 = lstsq([[1, x["slice_bytes"]] for x in r], [x["per_step_us"] for x in r])
            par["beta_nv_n"][n] = b
    r = [x for x in rows if x["tag"] == "micro-fanout" and x["per_step_us"] > 0]
    gaps = []
    for s in sorted({x["slice_bytes"] for x in r}):
        pts = [x for x in r if x["slice_bytes"] == s]
        if len(pts) >= 2 and "beta_net" in par:
            (c, slope), _ = lstsq([[1, x["stride"]] for x in pts], [x["per_step_us"] for x in pts])
            gaps.append(slope - par["beta_net"] * s)
    if gaps:
        par["g_net"] = sum(gaps) / len(gaps)
        par["g_net_per_slice"] = gaps
    par["notes"] = notes
    return par


def predict(par, sched, stride, P, L, slice_bytes, gamma_us):
    ld = loads(sched, stride, P, L)
    need = ("o_net", "alpha_nv", "alpha_net", "beta_net", "g_net")
    if ld is None or any(k not in par for k in need):
        return None
    total = par["o_net"] + gamma_us
    for rnd in ld:
        worst = 0.0
        for n_loc, b_loc, n_net, b_net in rnd:   # b_loc: blocks to the busiest same-node peer
            t = 0.0
            if n_loc:
                beta = par["beta_nv_n"].get(min(n_loc, 3)) or par["beta_nv_n"][1]
                t = max(t, par["alpha_nv"] + beta * b_loc * slice_bytes)
            if n_net:
                t = max(t, par["alpha_net"] + (n_net - 1) * par["g_net"] + par["beta_net"] * b_net * slice_bytes)
            worst = max(worst, t)
        total += worst
    return total


# ---- checks ----------------------------------------------------------------------------------------------------------
def correct(row):
    return row.get("wrong_words") in (0, -1) and row.get("timeout_step") == 0 and row.get("step_mismatch", 0) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--post-hoc-p8-only", action="store_true",
                    help="POST HOC (not pre-registered): drop the P = 7 launches, which no library could place "
                         "unevenly over two nodes, from every check")
    a = ap.parse_args()
    plan = json.loads((a.batch / "control" / "plan.json").read_text())
    driver = json.loads((a.batch / "control" / "driver-result.json").read_text())
    rows = []
    if a.post_hoc_p8_only:
        plan["launches"] = [s for s in plan["launches"] if "--ranks" not in s["argv"]]
    for spec in plan["launches"]:
        f = a.batch / "science" / spec["name"] / "stdout"
        if not f.exists():
            continue
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "case":
                    r["launch"], r["backend"] = spec["name"], spec["backend"]
                    rows.append(r)
    rec = dict(batch=str(a.batch), post_hoc_p8_only=a.post_hoc_p8_only, driver_status=driver.get("status"), gates=driver.get("gates"),
               launches={x["name"]: dict(status=x.get("status"), exit=x.get("exit"), cases=x.get("cases"),
                                         correct=x.get("correct_cases")) for x in driver.get("launches", [])})
    loads_file = next((a.batch / "science" / "preflight").glob("schedule-loads.json"), None) \
        if (a.batch / "science" / "preflight").exists() else None
    rec["port_crosscheck"] = crosscheck_loads(loads_file) if loads_file else "no schedule-loads.json"
    by_backend = defaultdict(list)
    for r in rows:
        by_backend[r["backend"]].append(r)
    # B1
    b1 = {}
    for b in DEVICE + ("host",):
        launches = [s for s in plan["launches"] if s["backend"] == b and s["stage"] in ("smoke", "qualify")]
        qrows = [r for r in by_backend[b] if r["launch"] in {s["name"] for s in launches}]
        exits = [rec["launches"].get(s["name"], {}).get("exit") for s in launches]
        b1[b] = dict(smoke=bool((driver.get("gates") or {}).get(f"{b}-smoke")), launches=len(launches),
                     exits=exits, cases=len(qrows), incorrect=[dict(launch=r["launch"], schedule=r["schedule"],
                     stride=r["stride"], slice=r["slice_bytes"], wrong=r["wrong_words"], mismatch=r.get("step_mismatch"),
                     timeout=r["timeout_step"]) for r in qrows if not correct(r)])
        b1[b]["pass_"] = b1[b]["smoke"] and all(e == 0 for e in exits) and qrows != [] and not b1[b]["incorrect"]
    rec["B1"] = b1
    passing = [b for b in DEVICE if b1[b]["pass_"]]
    # Fits and B2
    rec["fits"], b2 = {}, {}
    for b in passing:
        par = fit_backend(by_backend[b])
        rec["fits"][b] = par
        cmp_rows = [r for r in by_backend[b] if r["tag"] in ("cmp8", "cmp7") and r["gamma_ns"] == 0]
        model_rows, groups = [], defaultdict(dict)
        for r in cmp_rows:
            pred = predict(par, r["schedule"], r["stride"], r["P"], r["L"], r["slice_bytes"], 0.0)
            ratio = r["per_step_us"] / pred if pred else None
            model_rows.append(dict(P=r["P"], schedule=r["schedule"], stride=r["stride"], slice=r["slice_bytes"],
                                   measured_us=round(r["per_step_us"], 3), predicted_us=round(pred, 3) if pred else None,
                                   ratio=round(ratio, 3) if ratio else None, within_20pct=ratio is not None and 0.8 <= ratio <= 1.2,
                                   phases_us=[r.get("phase_compute_us"), r.get("phase_publish_us"), r.get("phase_wait_us")]))
            groups[(r["P"], r["slice_bytes"])][(r["schedule"], r["stride"])] = (r["per_step_us"], pred)
        orderings = []
        for key, sch in groups.items():
            for (sa, ma), (sb, mb) in itertools.combinations(sorted(sch.items()), 2):
                if not ma[1] or not mb[1] or abs(ma[1] - mb[1]) / min(ma[1], mb[1]) <= 0.10:
                    continue
                pf = sa if ma[1] < mb[1] else sb
                mf = sa if ma[0] < mb[0] else sb
                orderings.append(dict(P=key[0], slice=key[1], pair=[list(sa), list(sb)], predicted_faster=list(pf),
                                      measured_faster=list(mf), holds=pf == mf))
        within = sum(m["within_20pct"] for m in model_rows)
        b2[b] = dict(cases=len(model_rows), within_20pct=within, orderings=len(orderings),
                     orderings_hold=sum(o["holds"] for o in orderings), rows=model_rows, ordering_rows=orderings,
                     pass_=bool(model_rows) and within == len(model_rows) and all(o["holds"] for o in orderings))
    rec["B2"] = b2
    # B3 / B4 / B5
    b3, b4, b5 = {}, {}, {}
    host = {(r["P"], r["slice_bytes"]): r["per_step_us"] for r in by_backend["host"] if r["graph"] == 1 and r["gamma_ns"] == 0}
    for b in passing:
        par = rec["fits"][b]
        t = {(r["P"], r["slice_bytes"], r["schedule"], r["stride"]): r["per_step_us"]
             for r in by_backend[b] if r["tag"] in ("cmp8", "cmp7") and r["gamma_ns"] == 0}
        res = []
        for P, slices, topo in ((8, CLAIM8, ("hier", "doubling")), (7, CLAIM7, ("cyclic",)))[:1 if a.post_hoc_p8_only else 2]:
            for s in slices:
                flat = {k: v for k, v in t.items() if k[0] == P and k[1] == s and k[2] == "flat"}
                cand = {k: v for k, v in t.items() if k[0] == P and k[1] == s and k[2] in topo}
                if not flat or not cand:
                    res.append(dict(P=P, slice=s, missing=True))
                    continue
                fk, tk = min(flat, key=flat.get), min(cand, key=cand.get)
                pf = predict(par, fk[2], fk[3], P, 4, s, 0.0)
                pt = predict(par, tk[2], tk[3], P, 4, s, 0.0)
                m = flat[fk] / cand[tk]
                pr = pf / pt if pf and pt else None
                res.append(dict(P=P, slice=s, flat=list(fk[2:]), flat_us=round(flat[fk], 3), topo=list(tk[2:]),
                                topo_us=round(cand[tk], 3), measured_ratio=round(m, 3),
                                predicted_ratio=round(pr, 3) if pr else None,
                                accepted=m >= 1.1 and (pr or 0) >= 1.0))
        b3[b] = dict(rows=res, pass_=all(x.get("accepted") for x in res))
        res4 = []
        for P, slices in ((8, CLAIM8), (7, CLAIM7))[:1 if a.post_hoc_p8_only else 2]:
            for s in slices:
                dev = {k: v for k, v in t.items() if k[0] == P and k[1] == s}
                if not dev or (P, s) not in host:
                    res4.append(dict(P=P, slice=s, missing=True))
                    continue
                k = min(dev, key=dev.get)
                res4.append(dict(P=P, slice=s, best=list(k[2:]), best_us=round(dev[k], 3), host_graph_us=round(host[(P, s)], 3),
                                 ratio=round(host[(P, s)] / dev[k], 3), accepted=host[(P, s)] / dev[k] >= 1.1))
        b4[b] = dict(rows=res4, pass_=all(x.get("accepted") for x in res4))
        res5 = []
        for r in by_backend[b]:
            if r["tag"] not in ("add-g2", "add-g5"):
                continue
            base = t.get((8, r["slice_bytes"], r["schedule"], r["stride"]))
            if base is None:
                res5.append(dict(tag=r["tag"], missing=True))
                continue
            g = r["gamma_ns"] / 1000.0
            dev = r["per_step_us"] - (g + base)
            tol = max(0.1 * (g + base), 1.0)
            res5.append(dict(tag=r["tag"], schedule=r["schedule"], slice=r["slice_bytes"], gamma_us=g,
                             measured_us=round(r["per_step_us"], 3), additive_us=round(g + base, 3),
                             deviation_us=round(dev, 3), tolerance_us=round(tol, 3), holds=abs(dev) <= tol))
        b5[b] = dict(rows=res5, pass_=bool(res5) and all(x.get("holds") for x in res5))
    rec.update(B3=b3, B4=b4, B5=b5, backends_passing_B1=passing,
               host_rows=[dict(P=r["P"], slice=r["slice_bytes"], graph=r["graph"], per_step_us=r["per_step_us"],
                               wrong=r["wrong_words"]) for r in by_backend["host"]])
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(json.dumps(dict(B1={b: v["pass_"] for b, v in b1.items()}, B2={b: v["pass_"] for b, v in b2.items()},
                          B2_within={b: f'{v["within_20pct"]}/{v["cases"]}' for b, v in b2.items()},
                          B2_orderings={b: f'{v["orderings_hold"]}/{v["orderings"]}' for b, v in b2.items()},
                          B3={b: v["pass_"] for b, v in b3.items()}, B4={b: v["pass_"] for b, v in b4.items()},
                          B5={b: v["pass_"] for b, v in b5.items()}, port=rec["port_crosscheck"])))


if __name__ == "__main__":
    main()
