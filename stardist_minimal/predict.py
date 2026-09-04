from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import IMAGE_SUFFIXES, read_image
from .runtime import load_model, predict_instances, resolve_device


def _inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise ValueError(f"No supported images found in {path}")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StarDist instance segmentation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in _inputs(Path(args.input)):
        image = read_image(path)
        labels = predict_instances(
            model,
            image,
            device,
            score_thresh=args.score_thresh,
            iou_thresh=args.nms_iou_thresh,
        )
        destination = output_dir / f"{path.stem}.npy"
        np.save(destination, labels.astype(np.int32))
        print(f"{path.name} -> {destination} ({int(labels.max())} instances)")


if __name__ == "__main__":
    main()
