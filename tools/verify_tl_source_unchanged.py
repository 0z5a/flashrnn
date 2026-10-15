"""Login-node check of the TileLang race-verification build: with _verify off, the cluster and grid kernels generate
the same CUDA source as the uninstrumented module; every verification variant (checks, delays, broken controls)
compiles. Usage: verify_tl_source_unchanged.py --baseline FILE --out DIR
"""
import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import tilelang


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source(fn):
    return tilelang.compile(fn, target={"kind": "cuda", "arch": "sm_90a"}).get_kernel_source()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import flashrnn.flashrnn2.tilelang_cluster as new
    old = load(a.baseline, "flashrnn.flashrnn2.tilelang_cluster_baseline")
    rows = []
    for cell in ("lstm", "slstm"):
        for kw in (dict(sync="cluster", width=256, slice_width=32, bm=16), dict(sync="cluster", width=256,
                   slice_width=16, bm=32), dict(sync="grid", width=768, slice_width=16, bm=16),
                   dict(sync="grid", width=768, slice_width=16, bm=64, threads=256)):
            kw = dict(kw)
            sync = kw.pop("sync")
            common = dict(cell=cell, dtype="float16", mode="hidden", snapshot="float32", bm=kw["bm"],
                          threads=kw.get("threads", 128))
            if sync == "grid":
                common["sms"] = 132
            build_old = old.build_grid if sync == "grid" else old.build
            build_new = new.build_grid if sync == "grid" else new.build
            shape = (5, 9, 1, kw["width"], kw["slice_width"])
            row = dict(cell=cell, sync=sync, **kw)
            try:
                row["off_equal"] = source(build_old(*shape, **common)) == source(build_new(*shape, **common))
                variants = [dict(check=True), dict(check=True, delay_seed=3)]
                variants += [dict(check=True, broken=b) for b in (("single", "no_wait", "no_sync") if sync == "cluster"
                                                                  else ("single", "no_barrier", "arrive_first"))]
                compiled = []
                for v in variants:
                    try:
                        source(build_new(*shape, **common, _verify=v))
                        compiled.append(True)
                    except Exception:  # noqa: BLE001
                        compiled.append(traceback.format_exc()[-800:])
                row["variants_compiled"] = compiled
            except Exception:  # noqa: BLE001
                row["error"] = traceback.format_exc()[-1500:]
            rows.append(row)
            print(json.dumps({k: (v if not isinstance(v, list) else [x is True for x in v]) for k, v in row.items()
                              if k != "error"}), row.get("error", "")[-600:], flush=True)
    if True:  # 256-thread cluster positive control compiles under verification only
        try:
            source(new.build(5, 9, 1, 256, 32, cell="lstm", dtype="float16", mode="hidden", snapshot="float32",
                             threads=256, bm=32, _verify=dict(check=True, broken="threads256")))
            rows.append(dict(positive_control_threads256=True))
        except Exception:  # noqa: BLE001
            rows.append(dict(positive_control_threads256=traceback.format_exc()[-800:]))
    ok = all(r.get("off_equal", True) and all(x is True for x in r.get("variants_compiled", [True])) and
             "error" not in r and r.get("positive_control_threads256", True) is True for r in rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "verify-tl-source.json").write_text(json.dumps(dict(rows=rows, status="PASS" if ok else "FAIL"),
                                                                indent=1))
    print("STATUS", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
