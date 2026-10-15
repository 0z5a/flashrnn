"""Cluster-side start command for HC round 1 (SM-cost plane: TileLang batch tile BM > 16).

phase 0: qualification of every BM > 16 variant (first alone) plus compute-sanitizer memcheck/synccheck;
phase 1: split-0 selection per (cell, D, C) with tools/hc_bench.py select; frozen comparison table;
phase 2: split-1 paired timing with tools/hc_bench.py start (5 starts x 20 ABBA/BAAB blocks per comparison).
Phases are separated by barriers. Each lane is pinned to CPU cores local to its GPU. Runs with no client.
"""
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
BASE = CONTROL.parent
SOURCE = BASE / "source"
SCIENCE = BASE / "science"
RUNTIME = CONTROL / "runtime.sh"
PLAN = json.loads((CONTROL / "plan.json").read_text())
LOCK = threading.Lock()
RESULT = dict(job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, started=time.time(), plan=PLAN,
              phases={}, scientific_pass=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save():
    with LOCK:
        tmp = CONTROL / "driver-result.json.partial"
        tmp.write_text(json.dumps(RESULT, indent=2, default=str))
        tmp.replace(CONTROL / "driver-result.json")


def cpulist(text):
    cores = set()
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            cores.update(range(int(a), int(b) + 1))
        elif part:
            cores.add(int(part))
    return cores


def gpus():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used",
                        "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    rows = [dict(zip(("index", "uuid", "bus", "name", "driver", "memory_used"), map(str.strip, l.split(","))))
            for l in q.strip().splitlines()]
    allowed = os.sched_getaffinity(0)
    for r in rows:
        bus = r["bus"].lower()
        bus = bus[4:] if bus.startswith("0000") and len(bus) > 12 else bus
        path = Path(f"/sys/bus/pci/devices/{bus}/local_cpulist")
        r["local_cpus"] = sorted(cpulist(path.read_text()) & allowed) if path.exists() else sorted(allowed)
    # Split cores evenly between GPUs that share the same local CPU set.
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r["local_cpus"]), []).append(r)
    for cores, members in groups.items():
        n = len(members)
        for i, r in enumerate(members):
            r["lane_cpus"] = list(cores[i * len(cores) // n:(i + 1) * len(cores) // n])
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    return rows, apps


def run_task(gpu, argv, out, tilelang=False, timeout=None, tool="tools/rtiled_m4.py", prefix=None):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], FRNN_GPU_UUID=gpu["uuid"], FRNN_CACHE_TAG="node",
               FRNN_TK_ROOT=PLAN["tk_root"],
               FRNN_TK_CACHE=PLAN["tk_cache"].replace("{job}", os.environ.get("SLURM_JOB_ID", "dry")),
               FRNN_TILELANG="1" if tilelang else "0")
    if prefix:
        env["FRNN_PREFIX"] = prefix
    cmd = ["taskset", "-c", ",".join(map(str, gpu["lane_cpus"])), "bash", str(RUNTIME), tool] + argv
    rec = dict(argv=cmd, out=str(out), gpu=gpu["index"], started=time.time())
    with open(out / "task.log", "x") as log:
        p = subprocess.Popen(cmd, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rec["exit"] = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 15)
            try:
                rec["exit"] = p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                rec["exit"] = p.wait()
            rec["timed_out"] = True
    rec.update(finished=time.time(), joined=True, done=(out / "DONE.json").exists())
    return rec


def parallel(lanes, work, per_lane=1):
    """work: list of (lane_index, callable) -> records; at most per_lane concurrent per lane."""
    queues = {i: [w for l, w in work if l == i] for i in range(len(lanes))}
    records = []

    def worker(i):
        while True:
            with LOCK:
                if not queues[i]:
                    return
                job = queues[i].pop(0)
            rec = job()
            with LOCK:
                records.append(rec)
            save()

    threads = [threading.Thread(target=worker, args=(i,)) for i in queues for _ in range(per_lane)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return records


def main():
    rows, apps = gpus()
    RESULT.update(gpus_at_start=rows, compute_apps_at_start=apps, gpfs_free_bytes=shutil.disk_usage(BASE).free)
    save()
    if len(rows) < 4 or apps:
        RESULT["status"] = "ADMISSION_FAIL"
        save()
        return 2
    lanes = rows[:4]
    deadline = time.time() + 60 * PLAN["minutes"]

    # Phase 0: qualification of the BM > 16 variants, the first alone; compute-sanitizer on two small cases.
    quals = PLAN["qualifications"]
    def q_out(q):
        return SCIENCE / "phase0" / f"{q[0]}-{q[1]}-d{q[2]}-s{q[3]}-{q[4]}-bm{q[5]}"
    def q_task(g, q, prefix=None, stages=None, name=None):
        out = SCIENCE / "phase0" / name if name else q_out(q)
        argv = ["--out", str(out), "--cell", q[0], "--dtype", q[1], "--width", str(q[2]), "--slice", str(q[3]),
                "--sync", q[4], "--bm", str(q[5]), "--stages", stages or q[6]]
        return run_task(g, argv, out, tilelang=True, timeout=PLAN["qualification_timeout_s"],
                        tool="tools/rtiled_m6_tilelang.py", prefix=prefix)
    first = q_task(lanes[0], quals[0])
    RESULT["phases"]["phase0_first"] = first
    save()
    f0 = q_out(quals[0]) / "m6.json"
    first_gates = json.loads(f0.read_text()).get("gates", {}) if f0.exists() else {}
    first_ok = bool(first_gates) and "ERROR" not in first_gates.values()
    work = [(n % 4, lambda g=lanes[n % 4], q=q: q_task(g, q)) for n, q in enumerate(quals) if n > 0 and first_ok]
    for n, run in enumerate(PLAN["sanitizer_runs"]):
        work.append((n % 4, lambda g=lanes[n % 4], r=run: q_task(g, r["case"] + ["sanity"], prefix=r["prefix"],
                                                                  stages=r["stages"], name=f"sanitizer-{r['tool']}")))
    RESULT["phases"]["phase0"] = dict(records=parallel(lanes, work, per_lane=2), finished=time.time())
    status = {}
    for q in quals:
        f = q_out(q) / "m6.json"
        status[tuple(q)] = json.loads(f.read_text()).get("status", "INCOMPLETE") if f.exists() else (
            "NOT_RUN_FIRST_FAILED" if not first_ok else "MISSING_OR_TIMEOUT")
    RESULT["phases"]["phase0"]["qualification"] = {"/".join(map(str, k[:6])): v for k, v in status.items()}
    sanitizer = {}
    for run in PLAN["sanitizer_runs"]:
        out = SCIENCE / "phase0" / f"sanitizer-{run['tool']}"
        rec = next((r for r in RESULT["phases"]["phase0"]["records"] if r["out"] == str(out)), {})
        log = (out / "task.log").read_text(errors="replace") if (out / "task.log").exists() else ""
        f = out / "m6.json"
        gates = json.loads(f.read_text()).get("gates", {}) if f.exists() else {}
        summaries = [l.strip() for l in log.splitlines() if "SUMMARY:" in l and l.startswith("=========")]
        errors = [l for l in summaries if "ERROR SUMMARY" in l]
        clean = (rec.get("exit") == 0 and bool(errors) and gates.get("sanity") == "PASS"
                 and all(l == "========= ERROR SUMMARY: 0 errors" for l in errors))
        sanitizer[run["tool"]] = dict(exit=rec.get("exit"), summaries=summaries, sanity=gates.get("sanity"),
                                      blocking=run["blocking"], status="PASS" if clean else "FAIL")
    RESULT["phases"]["phase0"]["sanitizer"] = sanitizer
    save()
    # A BM > 16 configuration enters selection only if its FP16 qualification (incl. bitwise = BM 16) passed.
    available = {(q[0], q[2], q[3], q[4], q[5]) for q in quals if q[1] == "float16" and status[tuple(q)] == "PASS"}
    # A sanitizer finding excludes the configuration it ran on (all cells), not the whole batch.
    excluded = set()
    for run in PLAN["sanitizer_runs"]:
        if run["blocking"] and sanitizer[run["tool"]]["status"] != "PASS":
            excluded.add(tuple(run["case"][2:6]))
    available = {a for a in available if a[1:] not in excluded}
    RESULT["phases"]["phase0"]["excluded_by_sanitizer"] = sorted(map(list, excluded))
    stop = []
    if not first_ok:
        stop.append("first qualification raised an ERROR")
    if not available:
        stop.append("no FP16 BM > 16 configuration qualified")
    if stop:
        RESULT.update(status="PHASE0_GATE_STOP", stop_reasons=stop, finished=time.time())
        save()
        return 1

    # Phase 1: split-0 selection, one process per case.
    work = []
    for case in PLAN["select_cases"]:
        keep = []
        for c in case["candidates"]:
            cfg = c["config"]
            if c.get("wide") and (case["cell"], case["width"], cfg["slice_width"], cfg["sync"],
                                  cfg["batch_tile"]) not in available:
                continue
            keep.append(c)
        labels = {c["label"] for c in keep}
        case["candidates"] = keep
        case["roles"] = {r: [l for l in v if l in labels] for r, v in case["roles"].items()}
        out = SCIENCE / "phase1" / case["name"]
        work.append((case["lane"], lambda g=lanes[case["lane"]], c=case, o=out: run_task(
            g, ["select", "--out", str(o), "--case", json.dumps(c), "--reps", str(PLAN["select_reps"])], o,
            tilelang=True, timeout=2400, tool="tools/hc_bench.py")))
    RESULT["phases"]["phase1"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    # Freeze the comparison table before any split-1 run.
    comps, unavailable, ident = [], [], None
    selected = {}
    for case in PLAN["select_cases"]:
        f = SCIENCE / "phase1" / case["name"] / "select.json"
        if f.exists():
            sel = json.loads(f.read_text())
            selected[case["name"]] = sel["roles"]
            ident = ident or sel["identity"]
    for spec in PLAN["comparisons"]:
        roles = selected.get(spec["select_case"], {})
        base = roles.get(spec["baseline_role"], {}).get("selected")
        cand = roles.get(spec["candidate_role"], {}).get("selected")
        if base is None or cand is None:
            unavailable.append(dict(spec, reason="selection incomplete"))
            continue
        comp = {k: spec[k] for k in ("name", "experiment", "cell", "width", "batch", "concurrency", "steps", "lane")}
        comp.update(baseline=base, candidate=cand,
                    baseline_label=roles[spec["baseline_role"]]["label"], candidate_label=roles[spec["candidate_role"]]["label"])
        comps.append(comp)
    table = dict(identity=ident, comparisons=comps, unavailable=unavailable, frozen=time.time(),
                 selection_split=0, timing_split=1)
    (CONTROL / "frozen-table.json").write_text(json.dumps(table, indent=2))
    table_sha = sha(CONTROL / "frozen-table.json")
    RESULT["frozen_table"] = dict(path=str(CONTROL / "frozen-table.json"), sha256=table_sha,
                                  comparisons=len(comps), unavailable=len(unavailable))
    save()
    if not comps:
        RESULT.update(status="NO_COMPARISON_QUALIFIED", finished=time.time())
        save()
        return 1

    # Phase 2: split-1 paired timing; canary first.
    durations = []
    def start_task(g, comp, start):
        out = SCIENCE / "phase2" / comp["name"] / f"start{start}"
        rec = run_task(g, ["start", "--out", str(out), "--table", str(CONTROL / "frozen-table.json"),
                           "--table-sha256", table_sha, "--comparison", comp["name"], "--start", str(start),
                           "--blocks", str(PLAN["blocks"])], out, tilelang=True, timeout=1200, tool="tools/hc_bench.py")
        rec.update(comparison=comp["name"], start=start)
        with LOCK:
            durations.append(rec["finished"] - rec["started"])
        return rec
    ordered = sorted(comps, key=lambda c: c["name"])
    canary = start_task(lanes[ordered[0]["lane"]], ordered[0], 0)
    RESULT["phases"]["phase2_canary"] = canary
    save()
    if canary["exit"] != 0:
        RESULT.update(status="TIMING_CANARY_FAILED", finished=time.time())
        save()
        return 1

    def comparison_job(gpu, comp):
        recs = []
        for start in range(PLAN["starts"]):
            if comp is ordered[0] and start == 0:
                recs.append(dict(canary))
                continue
            mean = sum(durations) / len(durations)
            if time.time() + 1.5 * mean > deadline:
                recs.append(dict(comparison=comp["name"], start=start, status="NOT_STARTED_DEADLINE"))
                continue
            rec = start_task(gpu, comp, start)
            recs.append(rec)
            if rec["exit"] != 0:
                break  # keep the real failure; never repeat a failed start as another start
        return dict(comparison=comp["name"], starts=recs)
    work = [(c["lane"], lambda g=lanes[c["lane"]], c=c: comparison_job(g, c)) for c in ordered]
    RESULT["phases"]["phase2"] = dict(records=parallel(lanes, work), finished=time.time())
    RESULT["gpus_at_end"], RESULT["compute_apps_at_end"] = gpus()
    done = [s for r in RESULT["phases"]["phase2"]["records"] for s in r["starts"] if s.get("exit") == 0]
    planned = len(comps) * PLAN["starts"]
    RESULT.update(finished=time.time(), completed_starts=len(done), planned_starts=planned,
                  concurrent_lanes="four GPU lanes timed concurrently on one node, each pinned to GPU-local cores",
                  status="ALL_STARTS_COMPLETED" if len(done) == planned else "PARTIAL_STARTS")
    save()
    return 0 if len(done) == planned else 1


if __name__ == "__main__":
    sys.exit(main())
