"""Retarget traced device literals; leaves weights and model arithmetic intact."""

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import torch


def nodes(block: torch._C.Graph | torch._C.Block) -> Iterator[torch._C.Node]:
    for node in block.nodes():
        yield node
        for child in node.blocks():
            yield from nodes(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.input.with_suffix(".meta.json").read_text())
    with args.input.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == metadata["artifact_sha256"]
        )
    module = torch.jit.load(str(args.input), map_location="cpu")
    changed = 0
    for part in module.modules():
        for name in part._c._method_names():
            for node in nodes(part._c._get_method(name).graph):
                if (
                    node.kind() == "prim::Constant"
                    and str(node.output().type()) == "Device"
                ):
                    assert str(node.output().toIValue()) == "cpu"
                    node.s_("value", "cuda")
                    changed += 1
    assert changed > 0
    module.save(str(args.output))
    with args.output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata.update(
        parent_artifact_sha256=metadata["artifact_sha256"],
        artifact_sha256=digest,
        artifact_bytes=args.output.stat().st_size,
        retargeted_device_literals=changed,
        retarget_harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="CUDA_RETARGETED_NOT_EXECUTED",
        required_map_location="cuda",
    )
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "changed_device_literals": changed,
                "artifact_sha256": digest,
            }
        )
    )


if __name__ == "__main__":
    main()
