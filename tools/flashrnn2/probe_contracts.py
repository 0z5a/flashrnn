"""Print resolved upstream CUDA dtype fields without initializing a GPU."""

import json

from flashrnn.flashrnn.flashrnn import FlashRNNConfig


def main() -> None:
    rows = []
    for cell in ("lstm", "slstm"):
        for label, pointwise, state in (
            ("native_bf16", None, None),
            ("fp32_pointwise_state", "float32", "float32"),
        ):
            config = FlashRNNConfig(
                function=cell,
                backend="cuda_fused",
                dtype="bfloat16",
                dtype_a=pointwise,
                dtype_s=state,
                batch_size=16,
                hidden_dim=64,
                num_heads=1,
            )
            rows.append(
                {
                    "cell": cell,
                    "variant": label,
                    "backend": config.backend,
                    "dtype_w": config.dtype_w,
                    "dtype_r": config.dtype_r,
                    "dtype_b": config.dtype_b,
                    "dtype_a": config.dtype_a,
                    "dtype_s": config.dtype_s,
                    "dtype_acc": config.dtype_acc,
                    "status": "CONFIG_ONLY_NOT_EXECUTED",
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
