#!/usr/bin/env python3
"""Measure StarDist ONNX CPU throughput with concurrent ORT sessions."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path("stardist_onnx_validation/stardist_hgsc.onnx")
TILE_SIZE = 256
WSI_SIZE = 30_000
STRIDE = 128
TASKS_PER_CONFIG = 16

# (independent workers/sessions, intra-op threads per session)
CONFIGS = [
    (1, 4),
    (2, 4),
    (4, 2),
    (2, 2),
    (4, 1),
]


def make_session(threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def run_once(session: ort.InferenceSession, x: np.ndarray) -> None:
    session.run(None, {"image": x})


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing {MODEL_PATH}. Run stardist_onnx_standalone.py first."
        )

    tiles_per_axis = math.ceil((WSI_SIZE - TILE_SIZE) / STRIDE) + 1
    tiles_per_wsi = tiles_per_axis**2
    x = np.zeros((1, 3, TILE_SIZE, TILE_SIZE), dtype=np.float32)

    print("=== StarDist ONNX concurrency sweep ===")
    print(f"Model       : {MODEL_PATH}")
    print(f"Tile        : {TILE_SIZE}x{TILE_SIZE}")
    print(f"WSI estimate: {WSI_SIZE}x{WSI_SIZE}, stride={STRIDE}")
    print(f"Tiles/WSI   : {tiles_per_wsi:,}")
    print(f"Tasks/config: {TASKS_PER_CONFIG}")
    print()
    print("workers  threads/worker   tiles/s   ms/tile   30k hours")
    print("---------------------------------------------------------")

    results: list[tuple[float, int, int, float, float]] = []

    for workers, threads in CONFIGS:
        sessions = [make_session(threads) for _ in range(workers)]

        # Warm every session once before measuring.
        for session in sessions:
            run_once(session, x)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(run_once, sessions[i % workers], x)
                for i in range(TASKS_PER_CONFIG)
            ]
            for future in futures:
                future.result()
        elapsed = time.perf_counter() - start

        tiles_per_second = TASKS_PER_CONFIG / elapsed
        ms_per_tile = 1000.0 / tiles_per_second
        hours = tiles_per_wsi / tiles_per_second / 3600.0
        results.append((tiles_per_second, workers, threads, ms_per_tile, hours))

        print(
            f"{workers:7d}  {threads:14d}  "
            f"{tiles_per_second:8.3f}  {ms_per_tile:8.1f}  {hours:10.2f}"
        )

    best = max(results, key=lambda row: row[0])
    throughput, workers, threads, ms_per_tile, hours = best

    print("\n=== Best measured configuration ===")
    print(f"workers       : {workers}")
    print(f"threads/worker: {threads}")
    print(f"throughput    : {throughput:.3f} tiles/s")
    print(f"per tile      : {ms_per_tile:.1f} ms")
    print(f"30k estimate  : {hours:.2f} hours of model-forward time")
    print(
        "\nNote: WSI estimate is model-forward time only. Loom will also include "
        "tile I/O, StarDist post-processing, stitching, and Zarr writes."
    )


if __name__ == "__main__":
    main()
