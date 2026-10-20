"""Analysis of barrier_trace.py output (docs/flashrnn2/max-width.md section 82).

Usage: barrier_trace_analyze.py TRACE.json [--out FILE]
Per arm (steps >= 1 of every recorded call; a_i = arrival stamp of CTA i, r_i = its release stamp):
  N                 CTAs
  std               mean over steps of std_i(a_i)                       [ns]
  spread            mean over steps of max_i(a_i) - mean_i(a_i)          [ns]
  ratio             mean over steps of (max - mean) / std, against the Gaussian E[max of N standard normals] (exact,
                    numerical integral), and against the half-normal form of the section 6 claim
  wait              mean over steps and CTAs of r_i - a_i                [ns]
  latency           mean over steps of min_i(r_i) - max_i(a_i)           [ns]: the barrier's own cost after the last
                    arrival
  resolution        smallest positive difference between any two stamps  [ns]: globaltimer granularity
Section 82 P4 (pre-registered): ratio within 0.85-1.15 of the Gaussian value at every N (16, 32, 64 of the fw2d grids);
wait - latency within 0.8-1.2 of spread (the wait is the extreme-value term plus the barrier latency).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def gaussian_emax(n, grid=200001):
    x = np.linspace(-10, 10, grid)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    integrate = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    return float(integrate(x * n * phi * cdf ** (n - 1), x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = json.loads(a.trace.read_text())
    out = dict(gpu=d.get("gpu"), arms={})
    emax = {}
    for name, ent in d["arms"].items():
        if "error" in ent or not ent.get("calls"):
            out["arms"][name] = dict(error=(ent.get("error") or "no calls")[-300:])
            print(f"{name:22s} ERROR")
            continue
        tr = np.array(ent["calls"], dtype=np.float64)  # [calls, T, N, 2]
        n = tr.shape[2]
        arr, rel = tr[:, 1:, :, 0], tr[:, 1:, :, 1]
        std = arr.std(axis=2, ddof=1)
        spread = arr.max(axis=2) - arr.mean(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(std > 0, spread / std, np.nan)
        wait = (rel - arr).mean(axis=2)
        latency = rel.min(axis=2) - arr.max(axis=2)
        flat = np.unique(tr.reshape(-1))
        diffs = np.diff(flat)
        res = float(diffs[diffs > 0].min()) if (diffs > 0).any() else float("nan")
        if n not in emax:
            emax[n] = gaussian_emax(n)
        r = dict(N=n, calls=tr.shape[0], steps=tr.shape[1],
                 std_ns=float(np.nanmean(std)), spread_ns=float(np.nanmean(spread)),
                 ratio=float(np.nanmean(ratio)), ratio_median=float(np.nanmedian(ratio)),
                 gaussian_emax=emax[n], ratio_over_gaussian=float(np.nanmean(ratio) / emax[n]),
                 wait_ns=float(np.nanmean(wait)), latency_ns=float(np.nanmean(latency)),
                 wait_minus_latency_over_spread=float((np.nanmean(wait) - np.nanmean(latency)) / max(np.nanmean(spread), 1e-9)),
                 resolution_ns=res,
                 steps_zero_std=int((std == 0).sum()))
        out["arms"][name] = r
        print(f"{name:22s} N={n:3d} std {r['std_ns']:6.0f} spread {r['spread_ns']:6.0f} ratio {r['ratio']:.2f} "
              f"(gauss {emax[n]:.2f}, x{r['ratio_over_gaussian']:.2f}) wait {r['wait_ns']:6.0f} latency {r['latency_ns']:6.0f} "
              f"(wait-lat)/spread {r['wait_minus_latency_over_spread']:.2f} res {res:.0f} ns zero-std steps {r['steps_zero_std']}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
