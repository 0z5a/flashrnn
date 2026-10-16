"""Cluster-side start command for the fused-preactivation backward batch: (0) pinned FlashRNN1 prebuilt in the module
(CUDA 11.8) environment; (1) qualification of the fused backward levels (gradient gate with FlashRNN1 calibration,
persistent == launch, repeats), their race verification and compute-sanitizer memcheck/synccheck, then the
pre-registered admission of each fused level; (2) training-step breakdown at D = 384/512 (tools/bw_profile.py, timing
only); (3) split-0 selection over unfused and admitted fused candidates, frozen table and split-1 paired timing
(tools/bw_bench.py). Four single-GPU lanes pinned to GPU-local cores.
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


def run_task(gpu, argv, out, tilelang=False, timeout=None, tool="tools/bw_bench.py", prefix=None):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], FRNN_GPU_UUID=gpu["uuid"], FRNN_CACHE_TAG="node",
               FRNN_V1_BUILD_ROOT=PLAN["v1_build_root"].replace("{job}", os.environ.get("SLURM_JOB_ID", "dry")),
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
            try:
                rec = job()
            except Exception as error:  # noqa: BLE001 - one failing task must not end its lane
                rec = dict(status="DRIVER_ERROR", error=repr(error)[-800:], finished=time.time())
            with LOCK:
                records.append(rec)
            save()

    threads = [threading.Thread(target=worker, args=(i,)) for i in queues for _ in range(per_lane)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return records


def sanitizer_clean(log, exit_code):
    text = log.read_text(errors="replace") if log.exists() else ""
    summaries = [l for l in text.splitlines() if "ERROR SUMMARY:" in l]
    return exit_code == 0 and bool(summaries) and all(" 0 errors" in l for l in summaries)


def admitted_roles(case, admitted):
    """Drop fused candidates of levels that were not admitted (pre-registered gate); the rest is unchanged."""
    roles = {}
    for role, specs in case["roles"].items():
        roles[role] = [s for s in specs if not s.get("fuse") or s["fuse"] in admitted]
    return dict(case, roles={r: v for r, v in roles.items() if v})


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

    # Phase 0a: FlashRNN1 prebuild (module environment): selection among the FlashRNN1 role only.
    def prebuild(g, c):
        out = SCIENCE / "phase0" / "prebuild" / c["name"]
        return run_task(g, ["tselect", "--out", str(out), "--case", json.dumps(c)], out, tilelang=False, timeout=2400)
    work = [(c["lane"], lambda g=lanes[c["lane"]], c=c: prebuild(g, c)) for c in PLAN.get("prebuild", [])]
    RESULT["phases"]["prebuild"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    # Phase 0b: qualification of the fused levels; phase 0c: race verification; phase 0d: compute-sanitizer.
    def qualify(g, q):
        out = SCIENCE / "phase0" / "qualify" / q["name"]
        rec = run_task(g, ["--out", str(out), "--case", json.dumps(q["case"]), "--stages", q["stages"]], out,
                       tilelang=False, timeout=1800, tool="tools/bw_qualify.py")
        f = out / "bw.json"
        rec.update(name=q["name"], status=json.loads(f.read_text()).get("status") if f.exists() else "NO_RESULT")
        return rec
    def race(g, r):
        out = SCIENCE / "phase0" / "race" / r["name"]
        rec = run_task(g, ["--out", str(out), "--task", json.dumps(r["task"]), "--variants", r["variants"]], out,
                       tilelang=False, timeout=1800, tool="tools/race_verify.py")
        f = out / "race.json"
        rec.update(name=r["name"], status=json.loads(f.read_text()).get("status") if f.exists() else "NO_RESULT")
        return rec
    def sanitize(g, c):
        out = SCIENCE / "phase0" / "sanitizer" / c["name"]
        prefix = f"/apps/ACC/CUDA/12.8/bin/compute-sanitizer --tool {c['tool']} --error-exitcode 99"
        rec = run_task(g, ["--out", str(out), "--case", json.dumps(c["case"]), "--stages", "modes"], out,
                       tilelang=False, timeout=2400, tool="tools/bw_qualify.py", prefix=prefix)
        rec.update(name=c["name"], tool=c["tool"], clean=sanitizer_clean(out / "task.log", rec["exit"]))
        return rec
    work = [(q["lane"], lambda g=lanes[q["lane"]], q=q: qualify(g, q)) for q in PLAN.get("qualify", [])]
    work += [(r["lane"], lambda g=lanes[r["lane"]], r=r: race(g, r)) for r in PLAN.get("race", [])]
    work += [(c["lane"], lambda g=lanes[c["lane"]], c=c: sanitize(g, c)) for c in PLAN.get("sanitizer", [])]
    RESULT["phases"]["phase0"] = dict(records=parallel(lanes, work), finished=time.time())
    recs = {r.get("name"): r for r in RESULT["phases"]["phase0"]["records"]}
    admission = {}
    for level in sorted({q["fuse"] for q in PLAN["qualify"]}):
        q_ok = [recs.get(q["name"], {}).get("status") == "PASS" for q in PLAN["qualify"] if q["fuse"] == level]
        r_ok = []
        for r in PLAN["race"]:
            if r["fuse"] != level or not r["gating"]:
                continue
            f = SCIENCE / "phase0" / "race" / r["name"] / "race.json"
            summary = json.loads(f.read_text()).get("summary", {}) if f.exists() else {}
            r_ok.append(bool(summary.get("correct", {}).get("pass_")))
        s_ok = [bool(recs.get(c["name"], {}).get("clean")) for c in PLAN["sanitizer"] if c["fuse"] == level]
        admission[level] = dict(qualify=f"{sum(q_ok)}/{len(q_ok)}", race_correct=f"{sum(r_ok)}/{len(r_ok)}",
                                sanitizer=f"{sum(s_ok)}/{len(s_ok)}",
                                admitted=all(q_ok) and all(r_ok) and all(s_ok) and bool(q_ok and r_ok and s_ok))
    admitted = {level for level, a in admission.items() if a["admitted"]}
    RESULT["fused_admission"] = admission
    q_records = [r for r in RESULT["phases"]["phase0"]["records"] if "/qualify/" in str(r.get("out", ""))]
    RESULT["qualification_pass"] = sum(r.get("status") == "PASS" for r in q_records)
    RESULT["qualification_planned"] = len(PLAN.get("qualify", []))
    save()

    # Phase 0e: training-step breakdown (timing only; TileLang environment, FlashRNN1 loaded with the module toolkit).
    def profile(g, c):
        out = SCIENCE / "phase0" / "profile" / c["name"]
        rec = run_task(g, ["--out", str(out), "--case", json.dumps(c["case"]), "--tilelang"], out, tilelang=True,
                       timeout=1800, tool="tools/bw_profile.py")
        rec.update(name=c["name"], done=(out / "profile.json").exists())
        return rec
    work = [(c["lane"], lambda g=lanes[c["lane"]], c=c: profile(g, c)) for c in PLAN.get("profile", [])]
    RESULT["phases"]["profile"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    # Phase 1: split-0 selection; the first case runs alone as a canary.
    def sel(g, c):
        out = SCIENCE / "phase1" / c["name"]
        return run_task(g, ["tselect", "--out", str(out), "--case", json.dumps(c)], out,
                        tilelang=bool(c.get("tilelang")), timeout=2400)
    cases = [admitted_roles(c, admitted) for c in PLAN["select_cases"]]
    first = sel(lanes[cases[0]["lane"]], cases[0])
    RESULT["phases"]["phase1_first"] = first
    save()
    if first["exit"] != 0:
        RESULT.update(status="SELECTION_CANARY_FAILED", finished=time.time())
        save()
        return 1
    work = [(c["lane"], lambda g=lanes[c["lane"]], c=c: sel(g, c)) for c in cases[1:]]
    RESULT["phases"]["phase1"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    # Freeze the comparison table before any split-1 run.
    comps, unavailable, ident = [], [], None
    for spec in PLAN["comparisons"]:
        f = SCIENCE / "phase1" / spec["select_case"] / "select.json"
        roles = json.loads(f.read_text()) if f.exists() else {}
        ident = ident or roles.get("identity")
        base = roles.get("candidates", {}).get(spec["baseline_role"], {}).get("selected")
        cand = roles.get("candidates", {}).get(spec["candidate_role"], {}).get("selected")
        if base is None or cand is None:
            unavailable.append(dict(spec, reason="selection incomplete"))
            continue
        comp = {k: spec[k] for k in ("name", "experiment", "level", "cell", "dtype", "width", "batch", "concurrency",
                                     "steps", "lane", "tilelang")}
        comp.update(baseline=base, candidate=cand)
        comps.append(comp)
    table = dict(identity=ident, comparisons=comps, unavailable=unavailable, frozen=time.time(), selection_split=0,
                 timing_split=1)
    (CONTROL / "frozen-table.json").write_text(json.dumps(table, indent=2))
    table_sha = sha(CONTROL / "frozen-table.json")
    RESULT["frozen_table"] = dict(path=str(CONTROL / "frozen-table.json"), sha256=table_sha, comparisons=len(comps),
                                  unavailable=len(unavailable))
    save()
    if not comps:
        RESULT.update(status="NO_COMPARISON_SELECTED", finished=time.time())
        save()
        return 1

    # Phase 2: split-1 paired timing; canary first.
    durations = []
    def start_task(g, comp, start):
        out = SCIENCE / "phase2" / comp["name"] / f"start{start}"
        rec = run_task(g, ["tstart", "--out", str(out), "--table", str(CONTROL / "frozen-table.json"),
                           "--table-sha256", table_sha, "--comparison", comp["name"], "--start", str(start),
                           "--blocks", str(PLAN["blocks"])], out, tilelang=bool(comp.get("tilelang")), timeout=1200)
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
                break
        return dict(comparison=comp["name"], starts=recs)
    work = [(c["lane"], lambda g=lanes[c["lane"]], c=c: comparison_job(g, c)) for c in ordered]
    RESULT["phases"]["phase2"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    RESULT["gpus_at_end"], RESULT["compute_apps_at_end"] = gpus()
    done = [s_ for r in RESULT["phases"]["phase2"]["records"] for s_ in r["starts"] if s_.get("exit") == 0]
    planned = len(comps) * PLAN["starts"]
    RESULT.update(finished=time.time(), completed_starts=len(done), planned_starts=planned,
                  concurrent_lanes="four GPU lanes timed concurrently on one node, each pinned to GPU-local cores",
                  status="ALL_STARTS_COMPLETED" if len(done) == planned else "PARTIAL_STARTS")
    save()
    return 0 if len(done) == planned else 1


if __name__ == "__main__":
    sys.exit(main())
