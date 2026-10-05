"""Fixed-cohort full-model E2E, from queued token IDs to host-visible tokens.

All C requests arrive together. One worker admits static B-sized groups,
prefills each group, then advances them round-robin. Network and tokenization
are excluded; state allocation/reset, input copies and token delivery count.
"""

import argparse
import hashlib
import json
import os
import time
from itertools import pairwise
from pathlib import Path

import torch
from mamba_graph import Decode, GraphDecode
from mamba_native_runtime import NativeMamba
from mamba_script_gate import errors, verified_metadata


def quantiles(values: list[float]) -> dict[str, float]:
    data = torch.tensor(values, dtype=torch.float64)
    return {
        name: torch.quantile(data, q).item()
        for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    }


def qualify_graph(
    prefill: Decode, decode: Decode, cases: list[dict], contract: dict
) -> list[dict]:
    records = []
    for index, case in enumerate(cases):
        states = [torch.zeros_like(x, device="cuda") for x in case["prefill_cache"]]
        output = prefill(case["input_ids"].cuda(), *states)
        for step in range(contract["generated_tokens"]):
            logits = output[0][:, -1]
            comparison = errors((logits,), (case["logits"][step],), contract)
            if step == contract["generated_tokens"] - 1:
                comparison = errors(
                    (logits, output[1], output[2]),
                    (case["logits"][step], *case["final_cache"]),
                    contract,
                )
            ids = logits.argmax(-1, keepdim=True)
            token_match = torch.equal(ids[:, 0].cpu(), case["generated_ids"][step])
            records.append(
                {
                    "case": index,
                    "step": step,
                    "errors": comparison,
                    "token_match": token_match,
                    "pass": token_match and all(c["pass"] for c in comparison),
                }
            )
            if step + 1 < contract["generated_tokens"]:
                output = decode(ids, output[1], output[2])
    return records


def burst(
    prefill: Decode,
    decode: Decode,
    cases: list[dict],
    contract: dict,
    concurrency: int,
    device: str,
    check_states: bool,
) -> dict:
    batch, steps = contract["batch"], contract["generated_tokens"]
    assert concurrency >= batch and concurrency % batch == 0
    groups = concurrency // batch
    selected = [cases[i % len(cases)] for i in range(groups)]
    delivered = [[] for _ in selected]
    stamps = [[] for _ in selected]
    prefill_start = []
    model_batches = {"prefill": [], "decode": []}
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    states = [
        (
            torch.zeros_like(c["prefill_cache"][0], device=device),
            torch.zeros_like(c["prefill_cache"][1], device=device),
        )
        for c in selected
    ]
    ids = [c["input_ids"].to(device) for c in selected]
    for step in range(steps):
        for group in range(groups):
            model_batches["prefill" if step == 0 else "decode"].append(
                ids[group].shape[0]
            )
            if step == 0:
                prefill_start.append((time.perf_counter_ns() - started) / 1e6)
                output = prefill(ids[group], *states[group])
            else:
                output = decode(ids[group], *states[group])
            states[group] = (output[1], output[2])
            ids[group] = output[0][:, -1].argmax(-1, keepdim=True)
            # Blocking delivery makes timestamps host-visible, not just enqueued.
            delivered[group].append(ids[group][:, 0].cpu())
            stamps[group].append((time.perf_counter_ns() - started) / 1e6)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    requests = []
    token_match = True
    cache_checks = []
    for group, case in enumerate(selected):
        tokens = torch.stack(delivered[group])
        token_match &= torch.equal(tokens, case["generated_ids"])
        for row in range(batch):
            requests.append(
                {
                    "request_id": group * batch + row,
                    "case": group % len(cases),
                    "row": row,
                    "arrival_ms": 0,
                    "prefill_start_ms": prefill_start[group],
                    "token_times_ms": stamps[group],
                    "token_ids": tokens[:, row].tolist(),
                }
            )
        if check_states:
            comparisons = []
            for name, actual, expected in zip(
                ("conv_cache", "ssm_cache"), states[group], case["final_cache"]
            ):
                actual = actual.cpu()
                comparisons.append(
                    {
                        "tensor": name,
                        "max_abs": (actual - expected).abs().max().item(),
                        "pass": torch.isclose(
                            actual, expected, **contract["cache_contract"]
                        )
                        .all()
                        .item(),
                    }
                )
            cache_checks.append({"group": group, "errors": comparisons})
    passed = token_match and all(e["pass"] for c in cache_checks for e in c["errors"])
    result = {
        "status": "PASS" if passed else "CORRECTNESS_FAILED",
        "batch": batch,
        "concurrency": concurrency,
        "admission_batches": [case["input_ids"].shape[0] for case in selected],
        "model_batches": model_batches,
        "peak_active_requests": sum(state[0].shape[1] for state in states),
        "request_count": len(requests),
        "output_tokens": concurrency * steps,
        "elapsed_ms": elapsed_ms,
        "tokens_per_second": concurrency * steps * 1000 / elapsed_ms,
        "requests_per_second": concurrency * 1000 / elapsed_ms,
        "ttft_ms": quantiles([r["token_times_ms"][0] for r in requests]),
        "queue_wait_ms": quantiles([r["prefill_start_ms"] for r in requests]),
        "request_latency_ms": quantiles([r["token_times_ms"][-1] for r in requests]),
        "tpot_ms": quantiles(
            [
                (r["token_times_ms"][-1] - r["token_times_ms"][0]) / (steps - 1)
                for r in requests
            ]
        ),
        "itl_ms": quantiles(
            [b - a for r in requests for a, b in pairwise(r["token_times_ms"])]
        ),
        "token_match": token_match,
        "cache_checks": cache_checks,
        "requests": requests,
    }
    if device == "cuda":
        result.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--baseline", choices=("script", "native"), default="script")
    parser.add_argument("--candidate", choices=("script", "graph"), default="graph")
    parser.add_argument("--native-model", type=Path)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--native-qualification", type=Path)
    args = parser.parse_args()
    if args.baseline == args.candidate:
        parser.error("baseline and candidate must differ")
    if args.baseline == "native":
        if (
            args.native_model is None
            or args.native_source is None
            or args.native_config is None
            or args.native_qualification is None
        ):
            parser.error(
                "native baseline requires model, source, config and qualification"
            )
        if args.device == "cpu" and args.candidate == "graph":
            parser.error("CPU native comparison requires --candidate script")
    assert args.verify_only or (args.device == "cuda" and args.blocks >= 20)
    source, contract = verified_metadata(args.model), verified_metadata(args.goldens)
    qualified = json.loads(args.qualification.read_text())
    assert qualified["status"] == "PASS" and qualified["device"] == args.device
    assert qualified["model"]["artifact_sha256"] == source["artifact_sha256"]
    assert qualified["oracles"]["artifact_sha256"] == contract["artifact_sha256"]
    assert contract["generated_tokens"] > 1
    assert all(
        c >= contract["batch"] and c % contract["batch"] == 0 for c in args.concurrency
    )
    assert len(set(args.concurrency)) == len(args.concurrency)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = torch.load(args.goldens, map_location="cpu", weights_only=True)
    started = time.time()
    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    assert sum(p.numel() for p in model.parameters()) == contract["parameters"]
    baseline_name = "native_torch" if args.baseline == "native" else "torchscript"
    candidate_name = "torchscript" if args.candidate == "script" else "decode_graph"
    arms: dict[str, tuple[Decode, Decode]] = {
        "torchscript": (model.prefill, model.decode)
    }
    native_provenance = None
    if args.baseline == "native":
        native = NativeMamba(
            args.native_model, args.native_source, args.native_config, args.device
        )
        native_qualified = json.loads(args.native_qualification.read_text())
        assert native_qualified["status"] == "PASS"
        assert native_qualified["backend"] == "native"
        assert native_qualified["device"] == args.device
        assert (
            native_qualified["model"]["artifact_sha256"]
            == native.metadata["artifact_sha256"]
        )
        assert (
            native_qualified["oracles"]["artifact_sha256"]
            == contract["artifact_sha256"]
        )
        for path, expected in native_qualified["native_source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        native_provenance = {
            "model": native.metadata,
            "qualification_sha256": hashlib.sha256(
                args.native_qualification.read_bytes()
            ).hexdigest(),
            "source_sha256": native_qualified["native_source_sha256"],
            "scope": "Complete native Torch fallback prefill and decode",
        }
        arms = {baseline_name: (native.prefill, native.decode)}
        if args.candidate == "script":
            arms[candidate_name] = (model.prefill, model.decode)
    metadata = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": started,
        "scope": "TOKENIZED_FULL_MODEL_IN_PROCESS_FIXED_COHORT_E2E",
        "device": args.device,
        "torch": torch.__version__,
        "model": source,
        "native_baseline": native_provenance,
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "oracles": contract,
        "verification_only": args.verify_only,
        "concurrency": args.concurrency,
        "paired_blocks": 0 if args.verify_only else args.blocks,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(GraphDecode.__init__.__code__.co_filename),
                Path(errors.__code__.co_filename),
            )
        },
        "qualification_sha256": hashlib.sha256(
            args.qualification.read_bytes()
        ).hexdigest(),
        "timed": "Queue wait, state allocation/reset, prompt H2D, full prefill, G-1 decode, argmax and host-visible token delivery",
        "excluded": "Tokenization, HTTP/network, weight loading, graph setup, correctness comparisons and JSON output",
        "arrival_policy": "All C requests arrive at time zero; single worker, B-sized static groups, all prefill then round-robin decode; cases cycle deterministically",
        "statistics": "Within-process AB/BA pairs only; fresh-start validation remains separate. P99 uses linear interpolation of this finite cohort, not a steady-state SLO estimate.",
        "memory_scope": "All loaded models and any Graph buffers/private pool remain resident across arms; peaks are process totals, not independent per-arm footprints or capacity savings.",
    }
    if args.device == "cuda":
        metadata["device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.inference_mode(), args.output.open("x") as handle:
        if args.device == "cuda" and args.candidate == "graph":
            setup_start = time.perf_counter_ns()
            first = cases[0]
            graph = GraphDecode(
                model.decode,
                (
                    first["generated_ids"][0, :, None].cuda(),
                    *(x.cuda() for x in first["prefill_cache"]),
                ),
            )
            torch.cuda.synchronize()
            metadata["graph_setup_ms"] = (time.perf_counter_ns() - setup_start) / 1e6
            checks = qualify_graph(model.prefill, graph, cases, contract)
            metadata["graph_qualification"] = checks
            metadata["status"] = (
                "QUALIFIED"
                if all(c["pass"] for c in checks)
                else "GRAPH_CORRECTNESS_FAILED"
            )
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            assert metadata["status"] == "QUALIFIED"
            arms["decode_graph"] = (model.prefill, graph)
        metadata["arm_order"] = list(arms)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        for concurrency in args.concurrency:
            for label, (prefill, decode) in arms.items():
                row = burst(
                    prefill, decode, cases, contract, concurrency, args.device, True
                )
                row.update(arm=label, phase="qualification_and_warmup")
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "arm": label,
                            "concurrency": concurrency,
                            "status": row["status"],
                        }
                    ),
                    flush=True,
                )
                assert row["status"] == "PASS"
            if args.verify_only:
                continue
            for block in range(args.blocks):
                order = list(arms) if block % 2 == 0 else list(reversed(arms))
                for label in order:
                    row = burst(
                        *arms[label],
                        cases,
                        contract,
                        concurrency,
                        args.device,
                        False,
                    )
                    row.update(arm=label, phase="measurement", block=block, order=order)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    assert row["status"] == "PASS"
    metadata.update(status="PASS", finished=time.time())
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
