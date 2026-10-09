"""Inspect actual cuDNN LSTM weight packing behind the stacked E2E baseline."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from stacked_e2e import StackedModel


def packing(layer: torch.nn.LSTM) -> dict:
    weights = layer._flat_weights
    return {
        "weight_count": len(weights),
        "storage_count": len({w.untyped_storage().data_ptr() for w in weights}),
        "weight_data_ptrs_unique": len({w.data_ptr() for w in weights}) == len(weights),
        "cudnn_acceptable": [torch.backends.cudnn.is_acceptable(w) for w in weights],
    }


def observed_warnings(forward) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        for _ in range(2):
            forward()
            torch.cuda.synchronize()
    return [str(item.message) for item in caught]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    if str(props.uuid).removeprefix("GPU-") != args.expected_gpu_uuid.removeprefix(
        "GPU-"
    ):
        raise ValueError("GPU UUID differs from the admitted device")

    model = StackedModel("lstm", "cudnn", 4, 64)
    layers = [head for module in model.modules for head in module.layers]
    plain = torch.nn.LSTM(
        64, 64, batch_first=True, device="cuda", dtype=torch.bfloat16
    ).eval()
    before = [packing(layer) for layer in layers]
    plain_before = packing(plain)
    for layer in (*layers, plain):
        layer.flatten_parameters()
    after = [packing(layer) for layer in layers]
    plain_after = packing(plain)
    ids = torch.arange(16 * 128, dtype=torch.long).reshape(16, 128) % 512
    x = torch.zeros(16, 128, 64, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(1, 16, 64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        model_warnings = observed_warnings(lambda: model.forward(ids, "baseline"))
        plain_warnings = observed_warnings(lambda: plain(x, (state, state)))
    result = {
        "status": "PASS",
        "device_uuid": str(props.uuid),
        "device_name": props.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "flatten_weight_enabled": torch._use_cudnn_rnn_flatten_weight(),
        "mapped_before": before,
        "mapped_after": after,
        "plain_before": plain_before,
        "plain_after": plain_after,
        "mapped_warnings": model_warnings,
        "plain_warnings": plain_warnings,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
