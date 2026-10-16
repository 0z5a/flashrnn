"""Save complete ParaGRU/ParaLSTM mode outputs and gradients before timing."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

SOURCE_REVISION = "513d75dade40843bb27ac4a521969997188b5459"
CELLS = ("gru", "lstm")
MODES = ("sequential", "parallel_CUDA", "parallel_FUSED")
BUDGET = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float().cpu()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-extension-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    manifest = json.loads(args.manifest.read_text())
    if manifest["revision"] != SOURCE_REVISION or manifest["file_count"] != 31:
        raise ValueError("ParaRNN source revision differs from the pinned snapshot")
    for entry in manifest["files"]:
        if sha256(args.source_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"source file differs: {entry['path']}")
    extensions = list(args.source_root.glob("*parallel_reduce_cuda*.so"))
    if len(extensions) != 1 or sha256(extensions[0]) != args.expected_extension_sha:
        raise ValueError("native CUDA extension differs from the admitted build")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one assigned CUDA device must be visible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261009)
    sys.path.insert(0, str(args.source_root))
    from pararnn.rnn_cell.gru_diag_mh import GRUDiagMH, GRUDiagMHConfig
    from pararnn.rnn_cell.lstm_cifg_diag_mh import LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig
    from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode

    classes = {
        "gru": (GRUDiagMH, GRUDiagMHConfig),
        "lstm": (LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig),
    }
    records = {}
    with torch.enable_grad():
        for cell in CELLS:
            cell_class, config_class = classes[cell]
            layers = [
                cell_class(
                    config_class(
                        state_dim=8,
                        input_dim=8,
                        num_heads=2,
                        device="cuda",
                        dtype=torch.float32,
                        mode="sequential",
                    )
                ).eval()
                for _ in range(2)
            ]
            parameters = [
                parameter for layer in layers for parameter in layer.parameters()
            ]
            weights = [parameter.detach().clone() for parameter in parameters]
            fixture = torch.randn(2, 16, 8, device="cuda", dtype=torch.float32)
            arms = {}
            for mode in MODES:
                x = fixture.detach().clone().requires_grad_(True)
                for layer in layers:
                    layer.zero_grad(set_to_none=True)
                    layer.mode = RNNCellApplicationMode(mode)
                first = layers[0](x)
                last = layers[1](first)
                last.square().sum().backward()
                values = {
                    "output_0": saved(first),
                    "output_1": saved(last),
                    "input_grad": saved(x.grad),
                }
                for index, layer in enumerate(layers):
                    for name, parameter in layer.named_parameters():
                        if parameter.grad is None:
                            raise ValueError(
                                f"missing gradient for {cell}/{mode}/{name}"
                            )
                        values[f"layer{index}.{name}"] = saved(parameter.grad)
                if not all(
                    torch.equal(parameter, weight)
                    for parameter, weight in zip(parameters, weights)
                ):
                    raise ValueError("ParaRNN weights changed between paired modes")
                arms[mode] = values
            records[cell] = arms
    args.output.write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": sha256(args.manifest),
                "extension_sha256": sha256(extensions[0]),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "dtype": "float32",
                "batch": 2,
                "steps": 16,
                "layers": 2,
                "budget": BUDGET,
                "records": records,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
