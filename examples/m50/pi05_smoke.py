#!/usr/bin/env python3
"""Run one FlashRT Pi0.5 inference on Houmo M50/XH2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import flash_rt


def _image(path: str | None) -> np.ndarray:
    if path is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required when --image is used") from exc
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compiled-model-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--prompt", default="put the bowl on the stove")
    parser.add_argument(
        "--state",
        nargs="*",
        type=float,
        default=[0.0] * 8,
        help="Raw robot state values; normalized from checkpoint statistics.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--save", default="m50_pi05_actions.npy")
    args = parser.parse_args()

    images = [_image(path) for path in args.image] if args.image else [_image(None)]
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        config="pi05",
        framework="torch",
        hardware="m50_hmm_compat",
        num_views=len(images),
        compiled_model_dir=args.compiled_model_dir,
        tokenizer_path=args.tokenizer,
        device_id=args.device_id,
    )
    actions = model.predict(images=images, prompt=args.prompt, state=args.state)
    output = Path(args.save).resolve()
    np.save(output, actions)
    print(f"actions shape: {actions.shape}")
    print(f"first action: {actions[0].tolist()}")
    print(f"saved: {output}")
    print(f"latency: {model.pipeline.latency_stats()}")


if __name__ == "__main__":
    main()
