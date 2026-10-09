"""Compare original Haste CUDA LSTM/GRU against identical FlashRNN cells."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from flashrnn.flashrnn2.reference import SIZES, recurrence

TOLERANCES = {
    "float16": {"atol": 0.015, "rtol": 0.03},
    "float32": {"atol": 1e-4, "rtol": 1e-4},
    "float64": {"atol": 1e-10, "rtol": 1e-10},
}


def case(module, cell: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(20261005)
    batch, steps, width, input_size = 3, 7, 4, 5
    gates, bias_gates, states = SIZES[cell]
    shapes = (
        (batch, steps, input_size),
        (gates, 1, width, input_size),
        (gates, 1, width, width),
        (bias_gates, 1, width),
        (states, batch, 1, 1, width),
    )
    x, w, r, b, s = (
        (0.1 * torch.randn(shape, dtype=torch.float64)).to("cuda", dtype=dtype)
        for shape in shapes
    )
    wx = torch.einsum("bti,ghdi->btghd", x, w)
    history, final = recurrence(wx, r, b, s, cell)
    mask = x.new_empty(0)
    if cell == "lstm":
        order = [0, 2, 1, 3]
        kernel = w[order, 0].reshape(4 * width, input_size).T.contiguous()
        recurrent = r[order, 0].reshape(4 * width, width).T.contiguous()
        bias = b[order, 0].reshape(-1).contiguous()
        h, c, _ = module.lstm_forward(
            False,
            0.0,
            x.transpose(0, 1).contiguous(),
            s[0, :, 0, 0].contiguous(),
            s[1, :, 0, 0].contiguous(),
            kernel,
            recurrent,
            bias,
            mask,
        )
        native_final = torch.stack((h[-1], c[-1]))[:, :, None, None]
    else:
        kernel = w[[1, 0, 2], 0].reshape(3 * width, input_size).T.contiguous()
        recurrent = r[[2, 1, 0], 0].reshape(3 * width, width).T.contiguous()
        bias = torch.cat((b[2, 0], b[1, 0], b[3, 0]))
        recurrent_bias = torch.cat((b[2, 0] * 0, b[1, 0] * 0, b[0, 0]))
        h, _ = module.gru_forward(
            False,
            0.0,
            x.transpose(0, 1).contiguous(),
            s[0, :, 0, 0].contiguous(),
            kernel,
            recurrent,
            bias,
            recurrent_bias,
            mask,
        )
        native_final = h[-1][None, :, None, None]
    torch.cuda.synchronize()
    return {
        name: value.detach().cpu()
        for name, value in {
            "x": x,
            "w": w,
            "r": r,
            "b": b,
            "s": s,
            "reference_history": history[0, :, :, 0],
            "reference_final": final,
            "native_history": h[1:].transpose(0, 1),
            "native_final": native_final,
        }.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    meta_path = args.raw.with_suffix(".meta.json")
    if args.raw.exists() or meta_path.exists():
        parser.error("raw or metadata output already exists")
    if torch.cuda.device_count() != 1:
        raise ValueError("isolate exactly one assigned GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(args.module_root))
    import haste_lstm_gru_native as native

    extension = Path(native.__file__).resolve()
    if extension.parent != args.module_root.resolve():
        raise ValueError("Haste extension is outside the qualified build root")
    extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()

    with torch.inference_mode():
        rows = {}
        for cell in ("lstm", "gru"):
            dtypes = [("float64", torch.float64), ("float32", torch.float32)]
            if cell == "gru":
                dtypes.append(("float16", torch.float16))
            for dtype_name, dtype in dtypes:
                rows[f"{cell}_{dtype_name}"] = case(native, cell, dtype)
    torch.save(rows, args.raw)
    metadata = {
        "status": "RAW_SAVED_INDEPENDENT_AUDIT_PENDING",
        "cases": len(rows),
        "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "extension_sha256": extension_sha256,
        "tolerances": TOLERANCES,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
