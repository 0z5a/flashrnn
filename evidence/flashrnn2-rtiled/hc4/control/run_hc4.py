"""Cluster-side start command for HC round 4: ablation ceilings (A), then the whole-node NUMA experiment (B).

A: one tools/hc_bench.py ablate task per GPU lane. B: four numa_worker processes, one per GPU, launched together for
each (round, condition, case), bound with numactl to the GPU's NUMA node ("local"), to the farthest node ("remote"),
or not bound ("unbound"); a go file starts them together once all four are warm. A and B never overlap.
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


def node_topology(lanes):
    """NUMA node of each lane's GPU, CPUs per node (within this job's affinity) and node distances."""
    allowed = os.sched_getaffinity(0)
    nodes = {}
    for d in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        n = int(d.name[4:])
        nodes[n] = dict(cpus=sorted(cpulist((d / "cpulist").read_text()) & allowed),
                        distance=[int(x) for x in (d / "distance").read_text().split()])
    gpus_ = []
    for g in lanes:
        bus = g["bus"].lower()
        bus = bus[4:] if bus.startswith("0000") and len(bus) > 12 else bus
        f = Path("/sys/bus/pci/devices") / bus / "numa_node"
        gpus_.append(int(f.read_text().strip()) if f.exists() else -1)
    return nodes, gpus_


def bindings(nodes, gpu_nodes, per_worker):
    """Disjoint core sets per worker: local = the GPU's node, remote = the farthest node (lowest index on ties)."""
    used, out = set(), {"local": [], "remote": [], "unbound": [None] * len(gpu_nodes)}
    for cond in ("local", "remote"):
        used = set()
        for n in gpu_nodes:
            if cond == "local":
                target = n
            else:
                dist = nodes[n]["distance"]
                far = max(dist)
                target = min(i for i, d in enumerate(dist) if d == far)
            free = [c for c in nodes[target]["cpus"] if c not in used][:per_worker]
            used.update(free)
            out[cond].append(dict(node=target, cpus=free))
    return out


def launch_worker(gpu, argv, out, binding):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], FRNN_GPU_UUID=gpu["uuid"], FRNN_CACHE_TAG="node",
               FRNN_TK_ROOT=PLAN["tk_root"], FRNN_TK_CACHE=PLAN["tk_cache"].replace("{job}", os.environ.get(
                   "SLURM_JOB_ID", "dry")), FRNN_TILELANG="1")
    if binding:
        env["FRNN_PREFIX"] = ("/usr/bin/numactl --physcpubind=" + ",".join(map(str, binding["cpus"]))
                              + f" --membind={binding['node']}")
    log = open(out / "task.log", "x")
    proc = subprocess.Popen(["bash", str(RUNTIME), "tools/hc_bench.py"] + argv, cwd=SOURCE, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log


def main():
    rows, apps = gpus()
    RESULT.update(gpus_at_start=rows, compute_apps_at_start=apps, gpfs_free_bytes=shutil.disk_usage(BASE).free)
    save()
    if len(rows) < 4 or apps:
        RESULT["status"] = "ADMISSION_FAIL"
        save()
        return 2
    lanes = rows[:4]

    # A: ablation ceilings, one task per lane.
    work = []
    for n, t in enumerate(PLAN["ablation"]):
        out = SCIENCE / "ablate" / t["name"]
        argv = ["ablate", "--out", str(out), "--config", json.dumps(t["config"]), "--cell", t["cell"],
                "--width", str(t["width"]), "--steps", str(PLAN["steps"]), "--batches", PLAN["ablation_batches"],
                "--variants", PLAN["variants"], "--reps", str(PLAN["ablation_reps"])]
        work.append((n % 4, lambda g=lanes[n % 4], a=argv, o=out: run_task(g, a, o, tilelang=True, timeout=2400,
                                                                            tool="tools/hc_bench.py")))
    RESULT["phases"]["A_ablate"] = dict(records=parallel(lanes, work), finished=time.time())
    save()

    # B: whole-node NUMA plane.
    nodes, gpu_nodes = node_topology(lanes)
    if -1 in gpu_nodes or any(n not in nodes for n in gpu_nodes):
        RESULT["phases"]["B_topology"] = dict(nodes=nodes, gpu_nodes=gpu_nodes, status="TOPOLOGY_UNKNOWN")
        RESULT.update(finished=time.time(), status="NUMA_SKIPPED_TOPOLOGY_UNKNOWN")
        save()
        return 1
    bind = bindings(nodes, gpu_nodes, PLAN["numa_cores_per_worker"])
    RESULT["phases"]["B_topology"] = dict(nodes=nodes, gpu_nodes=gpu_nodes, bindings=bind)
    save()
    numa_dir = SCIENCE / "numa"

    def run_set(tag, case, cond, bursts, go_now=False):
        go = numa_dir / tag / "go"
        procs = []
        for i, g in enumerate(lanes):
            out = numa_dir / tag / f"gpu{i}"
            argv = ["numa_worker", "--out", str(out), "--config", json.dumps(case["config"]), "--cell", case["cell"],
                    "--width", str(case["width"]), "--steps", str(PLAN["steps"]), "--concurrency",
                    str(case["concurrency"]), "--bursts", str(bursts), "--condition", cond, "--go", str(go)]
            procs.append((out,) + launch_worker(g, argv, out, bind[cond][i]))
        rec = dict(tag=tag, case=case["name"], condition=cond, launched=time.time())
        deadline = time.time() + 900
        while not all((o / "ready.json").exists() for o, _, _ in procs):
            if time.time() > deadline or any(p.poll() not in (None, 0) for _, p, _ in procs):
                break
            time.sleep(0.05)
        rec["all_ready"] = all((o / "ready.json").exists() for o, _, _ in procs)
        go.parent.mkdir(parents=True, exist_ok=True)
        go.write_text(json.dumps(dict(go=time.time())))
        rec["go"] = time.time()
        exits = []
        for o, p, log in procs:
            try:
                exits.append(p.wait(timeout=900))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 15)
                exits.append(p.wait())
            log.close()
        rec["exits"] = exits
        workers = []
        for o, _, _ in procs:
            f = o / "worker.json"
            workers.append(json.loads(f.read_text()) if f.exists() else None)
        if all(w and w.get("status") == "PASS" for w in workers):
            rec["node_requests_per_s"] = sum(w["requests"] for w in workers) / max(w["wall_ms"] for w in workers) * 1e3
            rec["worker_requests_per_s"] = [w["requests_per_s"] for w in workers]
            rec["resident_pages_by_node"] = [w.get("resident_pages_by_node") for w in workers]
        rec["finished"] = time.time()
        return rec

    # Warm the shared kernel caches once per case (one GPU at a time is enough; the go file exists up front).
    warm = []
    for case in PLAN["numa"]:
        tag = f"prewarm-{case['name']}"
        (numa_dir / tag).mkdir(parents=True, exist_ok=True)
        (numa_dir / tag / "go").write_text("{}")
        out = numa_dir / tag / "gpu0"
        argv = ["numa_worker", "--out", str(out), "--config", json.dumps(case["config"]), "--cell", case["cell"],
                "--width", str(case["width"]), "--steps", str(PLAN["steps"]), "--concurrency", str(case["concurrency"]),
                "--bursts", "1", "--condition", "prewarm", "--go", str(numa_dir / tag / "go")]
        warm.append(run_task(lanes[0], argv, out, tilelang=True, timeout=1200, tool="tools/hc_bench.py"))
    RESULT["phases"]["B_prewarm"] = warm
    save()
    sets = []
    conds = PLAN["numa_conditions"]
    for r in range(PLAN["numa_rounds"]):
        order = conds[r % len(conds):] + conds[:r % len(conds)]
        for cond in order:
            for case in PLAN["numa"]:
                rec = run_set(f"r{r}-{cond}-{case['name']}", case, cond, case["bursts"])
                rec["round"] = r
                sets.append(rec)
                RESULT["phases"]["B_numa"] = sets
                save()
    ok_a = all(r["exit"] == 0 and r["done"] for r in RESULT["phases"]["A_ablate"]["records"])
    ok_b = all("node_requests_per_s" in s_ for s_ in sets)
    RESULT["gpus_at_end"], RESULT["compute_apps_at_end"] = gpus()
    RESULT.update(finished=time.time(), status="ALL_COMPLETED" if ok_a and ok_b else "PARTIAL",
                  ablation_complete=ok_a, numa_complete=ok_b)
    save()
    return 0 if ok_a and ok_b else 1


if __name__ == "__main__":
    sys.exit(main())
