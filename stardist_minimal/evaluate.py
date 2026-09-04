from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import paired_paths, read_image, read_mask
from .metrics import instance_stats, pq_from_stats
from .runtime import load_model, predict_instances, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a StarDist checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.4)
    parser.add_argument("--match-iou-thresh", type=float, default=0.5)
    parser.add_argument("--save-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    pairs = paired_paths(args.images, args.masks)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    total = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
    for image_path, mask_path in pairs:
        image = read_image(image_path)
        truth = read_mask(mask_path)
        if image.shape[:2] != truth.shape:
            raise ValueError(
                f"Shape mismatch for {image_path.name}: image={image.shape[:2]}, mask={truth.shape}"
            )
        prediction = predict_instances(
            model,
            image,
            device,
            score_thresh=args.score_thresh,
            iou_thresh=args.nms_iou_thresh,
        )
        stats = instance_stats(
            truth, prediction, iou_threshold=args.match_iou_thresh
        )
        total["tp"] += int(stats["tp"])
        total["fp"] += int(stats["fp"])
        total["fn"] += int(stats["fn"])
        total["iou_sum"] += float(stats["iou_sum"])
        if save_dir is not None:
            np.save(save_dir / f"{image_path.stem}.npy", prediction.astype(np.int32))

    scores = pq_from_stats(
        total["tp"], total["fp"], total["fn"], total["iou_sum"]
    )
    result = {
        "images": len(pairs),
        **total,
        **scores,
        "match_iou_threshold": args.match_iou_thresh,
        "score_threshold": args.score_thresh,
        "nms_iou_threshold": args.nms_iou_thresh,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
