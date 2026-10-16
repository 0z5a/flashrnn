"""Reject private machine identifiers in a prospective release tree.

This checks the current Git tree and, when requested, every commit after a base.
Publish from a clean base when an earlier research branch contained private data.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "private_path": re.compile(rb"(?<![A-Za-z0-9])/(?:gpfs|home|root|Users|scratch)/"),
    "gpu_identity": re.compile(
        rb"\bGPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        rb"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    "machine_binding": re.compile(
        rb'"(?:boot_id|gpu_uuid|hostname|host|node|node_id|job_id|'
        rb'endpoint|account|project|partition|qos|cluster)"'
        rb'\s*:\s*(?:"[^"\r\n]+"|[0-9]{5,})'
    ),
    "ipv4_address": re.compile(
        rb"(?<![A-Za-z0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![A-Za-z0-9.])"
    ),
    "remote_login": re.compile(
        rb"\b(?:ssh|scp|rsync)\s+(?:-[^\s]+\s+)*[\w.-]+@[\w.-]+"
    ),
    "credential": re.compile(
        rb"\b(?:hf_|gh[opru]_)[A-Za-z0-9]{20,}\b|-----BEGIN [^-]*PRIVATE KEY-----"
    ),
    "cluster_context": re.compile(
        rb"\b(?:EuroHPC|JUWELS|Juelich|Tuebingen|Slurm|salloc|sbatch|srun)\b"
        rb"|\behpc[0-9]{3,}\b|\bas[0-9]{2}r[0-9]+b[0-9]+\b",
        re.IGNORECASE,
    ),
    "other_device": re.compile(
        rb"\b(?:H20|B200|Thor|5080|5090|SM90|SM100|SM110|SM120)\b",
        re.IGNORECASE,
    ),
}
OPAQUE_SUFFIXES = (".pt", ".pt.gz", ".tar", ".tar.gz", ".tgz", ".zip")
PRIVATE_PATH_NAME = re.compile(rb"(?:^|/)(?:gpfs|home|root|Users|scratch)/")


def tracked_paths(root: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [root / item.decode() for item in raw.split(b"\0") if item]


def scan_payload(
    relative: str,
    data: bytes,
    counts: dict[str, int],
    findings: list[tuple[str, str]],
) -> None:
    for category, pattern in PATTERNS.items():
        if category in ("other_device", "cluster_context") and not relative.startswith(
            ("docs/", "evidence/", "commit-message/")
        ):
            continue
        path_pattern = PRIVATE_PATH_NAME if category == "private_path" else pattern
        hits = len(path_pattern.findall(relative.encode()))
        if hits:
            counts[category] += hits
            findings.append((relative, category))
    if (
        relative.endswith(OPAQUE_SUFFIXES)
        or ".part-" in Path(relative).name
        or b"\0" in data
    ):
        counts["opaque_file"] += 1
        findings.append((relative, "opaque_file"))
        return
    for category, pattern in PATTERNS.items():
        if category in ("other_device", "cluster_context") and not relative.startswith(
            ("docs/", "evidence/", "commit-message/")
        ):
            continue
        hits = 0
        for match in pattern.finditer(data):
            if category == "ipv4_address" and relative.startswith("environment_"):
                line_start = data.rfind(b"\n", 0, match.start()) + 1
                prefix = data[line_start : match.start()]
                if re.fullmatch(rb"\s*-\s*[\w.-]+\s*(?:#?==|=)", prefix):
                    continue
            hits += 1
        if hits:
            counts[category] += hits
            findings.append((relative, category))


def scan(
    root: Path, history_base: str | None
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    counts = {category: 0 for category in PATTERNS}
    counts["opaque_file"] = 0
    findings = []
    for path in tracked_paths(root):
        if path.is_file():
            scan_payload(
                path.relative_to(root).as_posix(), path.read_bytes(), counts, findings
            )
    if history_base is not None:
        commits = subprocess.check_output(
            ["git", "rev-list", f"{history_base}..HEAD"], cwd=root, text=True
        )
        seen = set()
        for commit in commits.splitlines():
            message = subprocess.check_output(
                ["git", "show", "-s", "--format=%B", commit], cwd=root
            )
            scan_payload(f"commit-message/{commit[:12]}", message, counts, findings)
            tree = subprocess.check_output(
                ["git", "ls-tree", "-r", "-z", commit], cwd=root
            )
            for entry in tree.split(b"\0"):
                header, _, path = entry.partition(b"\t")
                fields = header.split()
                if len(fields) != 3 or fields[1] != b"blob":
                    continue
                oid = fields[2].decode()
                relative = path.decode()
                if (oid, relative) not in seen:
                    seen.add((oid, relative))
                    data = subprocess.check_output(
                        ["git", "cat-file", "blob", oid], cwd=root
                    )
                    scan_payload(relative, data, counts, findings)
    return counts, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details", action="store_true", help="show paths and categories only"
    )
    parser.add_argument("--history-base", help="also scan each commit after this ref")
    args = parser.parse_args()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    counts, findings = scan(root, args.history_base)
    for category, count in counts.items():
        print(f"{category}: {count}")
    print(f"affected_files: {len({path for path, _ in findings})}")
    if args.details:
        for path, category in findings:
            print(f"{category}: {path}")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
