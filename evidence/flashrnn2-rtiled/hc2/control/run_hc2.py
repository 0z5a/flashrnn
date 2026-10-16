"""Cluster-side start command for HC round 2: profiling, occupancy-plane screen, launch-plane paired timing.

P0: one ncu probe alone (counter permission); P1: the other profiles with ncu in parallel, or nsys timelines of the
s0p forward if ncu cannot read counters; P2: split-0 screening with tools/hc_bench.py select (256- vs 128-thread
TileLang, twin bitwise checks); P3: split-1 paired timing of the CUDA Graph forward vs eager. No phase overlaps another.
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

    def prof_task(g, prof, tool_prefix, mode):
        out = SCIENCE / "profile" / f"{prof['name']}-{mode}"
        argv = [mode, "--out", str(out), "--config", json.dumps(prof["config"]), "--cell", prof["cell"],
                "--width", str(prof["width"]), "--steps", "128", "--calls", "5" if mode == "profile" else "3"]
        return run_task(g, argv, out, tilelang=True, timeout=1200, tool="tools/hc_bench.py",
                        prefix=tool_prefix.replace("{out}", str(out)))

    # P0 + P1: profiling, before any timing.
    profs = PLAN["profiles"]
    probe = prof_task(lanes[0], profs[0], PLAN["ncu"], "profile")
    log = Path(probe["out"]) / "task.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    ncu_ok = probe["exit"] == 0 and "Section: GPU Speed Of Light" in text and "ERR_NVGPUCTRPERM" not in text
    RESULT["phases"]["P0_ncu_probe"] = dict(probe, ncu_ok=ncu_ok, counter_permission_error="ERR_NVGPUCTRPERM" in text)
    save()
    if ncu_ok:
        work = [(n % 4, lambda g=lanes[n % 4], pr=pr: prof_task(g, pr, PLAN["ncu"], "profile"))
                for n, pr in enumerate(profs) if n > 0]
    else:
        work = [(n % 4, lambda g=lanes[n % 4], pr=pr: prof_task(g, pr, PLAN["nsys"], "forward"))
                for n, pr in enumerate(profs)]
    RESULT["phases"]["P1_profiles"] = dict(tool="ncu" if ncu_ok else "nsys", records=parallel(lanes, work),
                                           finished=time.time())
    save()

    # P2: split-0 screening.
    work = []
    for case in PLAN["select_cases"]:
        out = SCIENCE / "screen" / case["name"]
        work.append((case["lane"], lambda g=lanes[case["lane"]], c=case, o=out: run_task(
            g, ["select", "--out", str(o), "--case", json.dumps(c), "--reps", str(PLAN["select_reps"])], o,
            tilelang=True, timeout=2400, tool="tools/hc_bench.py")))
    RESULT["phases"]["P2_screen"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    # P3: the comparison table is fixed by the plan (round-1 configurations); freeze it before timing.
    ident = None
    for case in PLAN["select_cases"]:
        f = SCIENCE / "screen" / case["name"] / "select.json"
        if f.exists():
            ident = json.loads(f.read_text())["identity"]
            break
    table = dict(identity=ident, comparisons=PLAN["comparisons"], unavailable=[], frozen=time.time(),
                 selection="round-1 frozen table (run HC1 v2)", timing_split=1)
    (CONTROL / "frozen-table.json").write_text(json.dumps(table, indent=2))
    table_sha = sha(CONTROL / "frozen-table.json")
    RESULT["frozen_table"] = dict(path=str(CONTROL / "frozen-table.json"), sha256=table_sha,
                                  comparisons=len(PLAN["comparisons"]), unavailable=0)
    save()
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
    ordered = sorted(PLAN["comparisons"], key=lambda c: c["name"])
    canary = start_task(lanes[ordered[0]["lane"]], ordered[0], 0)
    RESULT["phases"]["P3_canary"] = canary
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
    RESULT["phases"]["P3"] = dict(records=parallel(lanes, work), finished=time.time())
    RESULT["gpus_at_end"], RESULT["compute_apps_at_end"] = gpus()
    done = [s for r in RESULT["phases"]["P3"]["records"] for s in r["starts"] if s.get("exit") == 0]
    planned = len(PLAN["comparisons"]) * PLAN["starts"]
    RESULT.update(finished=time.time(), completed_starts=len(done), planned_starts=planned,
                  concurrent_lanes="four GPU lanes timed concurrently on one node, each pinned to GPU-local cores",
                  status="ALL_STARTS_COMPLETED" if len(done) == planned else "PARTIAL_STARTS")
    save()
    return 0 if len(done) == planned else 1


if __name__ == "__main__":
    sys.exit(main())
