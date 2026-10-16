"""Cluster-side start command for HC round 0: decomposition of high-concurrency classifier serving.

Phase A runs one tools/hc_bench.py process per (cell, D) on four GPU lanes in parallel (full decomposition,
recurrence microbench, row-independence checks). Phase B repeats the C = 1024 timing of every (cell, D)
alone on one lane, so the effect of the concurrent lanes on host-side measurements is visible.

Phases are separated by barriers, so no compiler runs while any lane is timing.
Each lane is pinned to CPU cores local to its GPU. Runs with no client.
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
    tasks = PLAN["tasks"]

    def task(g, t, phase, extra):
        out = SCIENCE / phase / f"{t['cell']}-d{t['width']}"
        argv = ["--out", str(out), "--cell", t["cell"], "--width", str(t["width"]), "--steps", str(PLAN["steps"]),
                "--group", str(PLAN["group"])] + extra
        return run_task(g, argv, out, tilelang=True, timeout=PLAN["task_timeout_s"], tool="tools/hc_bench.py")

    # Phase A: the first task alone (compiles, exercises every path); a crash there stops the batch.
    first = task(lanes[0], tasks[0], "phaseA", ["--concurrency", PLAN["concurrency"], "--reps", str(PLAN["reps"])])
    RESULT["phases"]["phaseA_first"] = first
    save()
    if first["exit"] != 0 or not first["done"]:
        RESULT.update(status="FIRST_TASK_FAILED", finished=time.time())
        save()
        return 1
    work = [(n % 4, lambda g=lanes[n % 4], t=t: task(g, t, "phaseA", ["--concurrency", PLAN["concurrency"],
                                                                     "--reps", str(PLAN["reps"])]))
            for n, t in enumerate(tasks) if n > 0]
    RESULT["phases"]["phaseA"] = dict(records=parallel(lanes, work), finished=time.time())
    save()
    # Phase B: C = 1024 timing alone, one task at a time on lane 0.
    RESULT["phases"]["phaseB"] = dict(records=[task(lanes[0], t, "phaseB", ["--concurrency", "1024", "--reps",
                                                                            str(PLAN["reps"]), "--no-micro"])
                                               for t in tasks], finished=time.time())
    RESULT["gpus_at_end"], RESULT["compute_apps_at_end"] = gpus()
    done = [r for ph in ("phaseA", "phaseB") for r in RESULT["phases"][ph]["records"]] + [first]
    ok = all(r["exit"] == 0 and r["done"] for r in done)
    RESULT.update(finished=time.time(), status="ALL_TASKS_COMPLETED" if ok else "PARTIAL")
    save()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
