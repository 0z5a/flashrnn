"""Login-node check that every TileLang kernel a batch plan will launch traces and compiles (sm_90a, no GPU).

Usage: tl_trace_check.py <plan.json> <out.json>
For every distinct (config, cell, dtype, D, B, T, output mode) of the plan's qualification and timing cases whose
FlashRNN2 configuration is a TileLang one, build the kernel exactly as the dispatch would (forward "hidden"; training
cases also "full" with an FP32 snapshot) and compile it with tilelang.compile. Catches errors that only show when the
kernel is traced (e.g. an undefined name inside a prim_func), before an allocation is spent on them.
"""
import json
import sys
import time
import traceback

from flashrnn.flashrnn2 import tilelang_cluster as TC
from flashrnn.flashrnn2.execution_policy import KernelConfig

SMS = 132


def build(cfg, cell, dtype, width, batch, steps, mode, snapshot):
    if cfg.stream_h:
        from flashrnn.flashrnn2 import tilelang_stream
        return tilelang_stream.kernel(batch, steps, 1, width, cfg.reg_k, cfg.batch_tile, cell=cell, dtype=dtype,
                                      mode=mode, snapshot=snapshot, sms=SMS)
    return TC.kernel(batch, steps, 1, width, cfg.slice_width, cell=cell, dtype=dtype, mode=mode, snapshot=snapshot,
                     sync=cfg.sync, sms=SMS, bm=cfg.batch_tile, threads=32 * cfg.warps, prefetch=cfg.prefetch,
                     reg_k=cfg.reg_k)


def main():
    plan = json.loads(open(sys.argv[1]).read())
    keys = {}
    for case in plan.get("qualify", []) + plan.get("claims", []):
        for field in ("config", "alt_config"):
            raw = case.get(field)
            if not raw:
                continue
            cfg = KernelConfig(**raw)
            if cfg.family != "tilelang":
                continue
            train = case.get("train") or case.get("kind") == "train"
            modes = [("hidden", "float32")] + ([("full", "float32")] if train else [])
            for mode, snap in modes:
                key = json.dumps([raw, case["cell"], case["dtype"], case["width"], case["batch"], case["steps"], mode, snap],
                                 sort_keys=True)
                keys[key] = (cfg, case["cell"], case["dtype"], case["width"], case["batch"], case["steps"], mode, snap)
    rows = []
    for key, args in sorted(keys.items()):
        t0 = time.time()
        rec = dict(key=json.loads(key))
        try:
            k = build(*args)
            # The launch must keep the configured block size: an automatically added warpgroup (warp specialization)
            # would not take part in the cooperative grid barrier.
            import re
            m = re.search(r"__launch_bounds__\((\d+)", k.get_kernel_source())
            want = 128 if args[0].stream_h else 32 * args[0].warps
            rec.update(launch_threads=int(m.group(1)) if m else None, expected_threads=want)
            rec["ok"] = rec["launch_threads"] == want
            if not rec["ok"]:
                rec["error"] = f"block size {rec['launch_threads']} != configured {want} (warp specialization?)"
        except Exception:
            rec.update(ok=False, error=traceback.format_exc()[-1200:])
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(json.dumps(dict(ok=rec["ok"], key=rec["key"][:4], seconds=rec["seconds"])), flush=True)
    out = dict(kernels=len(rows), failed=sum(not r["ok"] for r in rows), rows=rows,
               status="PASS" if rows and all(r["ok"] for r in rows) else ("NONE" if not rows else "FAIL"))
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
    print(json.dumps(dict(status=out["status"], kernels=out["kernels"], failed=out["failed"])))


if __name__ == "__main__":
    main()
