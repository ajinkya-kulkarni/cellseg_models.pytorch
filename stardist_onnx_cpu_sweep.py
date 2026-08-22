#!/usr/bin/env python3
"""ONNX-only CPU throughput sweep for Loom-style StarDist tile inference.

Requires only numpy, Pillow, and onnxruntime at runtime. It uses the ONNX model
and example image created by stardist_onnx_standalone.py.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX_PATH = Path("stardist_onnx_validation/stardist_hgsc.onnx")
IMAGE_PATH = Path("stardist_onnx_validation/stardist_histo.jpg")

TILE_SIZE = 256
BATCH_SIZES = (1, 2, 4, 8)
THREAD_COUNTS = (1, 2, 4, 8)
WARMUP = 1
REPEATS = 3

# Loom-like estimate: 256 tile with 128 overlap -> 128 stride.
WSI_SIZE = 30_000
STRIDE = 128


def load_tile(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def make_session(threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def time_batch(session: ort.InferenceSession, batch: np.ndarray) -> tuple[float, float]:
    for _ in range(WARMUP):
        session.run(None, {"image": batch})

    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        session.run(None, {"image": batch})
        timings.append(time.perf_counter() - start)

    return float(np.median(timings)), float(np.mean(timings))


def tile_count(size: int, tile_size: int, stride: int) -> int:
    per_axis = max(1, math.ceil((size - tile_size) / stride) + 1)
    return per_axis * per_axis


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ONNX_PATH}. Run `python stardist_onnx_standalone.py` first."
        )
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IMAGE_PATH}. Run `python stardist_onnx_standalone.py` first."
        )

    tile = load_tile(IMAGE_PATH)
    n_tiles = tile_count(WSI_SIZE, TILE_SIZE, STRIDE)

    print("=== ONNX Runtime CPU sweep ===")
    print(f"Model       : {ONNX_PATH}")
    print(f"Tile        : {TILE_SIZE}x{TILE_SIZE}")
    print(f"WSI estimate: {WSI_SIZE}x{WSI_SIZE}, stride={STRIDE}")
    print(f"Tiles/WSI   : {n_tiles:,}")
    print(f"ORT         : {ort.__version__}")
    print(f"Providers   : {ort.get_available_providers()}")
    print()
    print(
        f"{'threads':>7} {'batch':>6} {'batch ms':>11} {'ms/tile':>10} "
        f"{'tiles/s':>9} {'30k hours':>10}"
    )
    print("-" * 61)

    best = None

    for threads in THREAD_COUNTS:
        session = make_session(threads)
        for batch_size in BATCH_SIZES:
            batch = np.repeat(tile, batch_size, axis=0)
            median_s, _ = time_batch(session, batch)
            batch_ms = median_s * 1000.0
            ms_per_tile = batch_ms / batch_size
            tiles_per_s = batch_size / median_s
            hours = n_tiles / tiles_per_s / 3600.0

            print(
                f"{threads:7d} {batch_size:6d} {batch_ms:11.1f} "
                f"{ms_per_tile:10.1f} {tiles_per_s:9.3f} {hours:10.2f}"
            )

            score = tiles_per_s
            if best is None or score > best[0]:
                best = (score, threads, batch_size, hours, ms_per_tile)

    assert best is not None
    tiles_per_s, threads, batch_size, hours, ms_per_tile = best
    print("\n=== Best measured configuration ===")
    print(f"threads     : {threads}")
    print(f"batch       : {batch_size}")
    print(f"throughput  : {tiles_per_s:.3f} tiles/s")
    print(f"per tile    : {ms_per_tile:.1f} ms")
    print(f"30k estimate: {hours:.2f} hours of model-forward time")
    print(
        "\nNote: the WSI estimate is model-forward time only; Loom will also have "
        "tile reads, StarDist post-processing, stitching, and Zarr writes."
    )


if __name__ == "__main__":
    main()
