#!/usr/bin/env python3
"""
Standalone StarDist PyTorch -> ONNX validation + benchmark.

What this script does
---------------------
1. Downloads a public StarDist example image automatically.
2. Loads the real pretrained cellseg_models.pytorch StarDist checkpoint:
       hgsc_v1_efficientnet_b5
3. Exports ONLY the dense neural-network forward pass to ONNX.
4. Loads the ONNX model with ONNX Runtime.
5. Checks PyTorch vs ONNX dense-output numerical parity.
6. Runs the SAME existing StarDist post-processing on both outputs.
7. Checks final instance-mask and type-mask parity.
8. Benchmarks PyTorch CPU vs ONNX Runtime CPU.

This file is intentionally independent of the ONNX helper added on the
feat/stardist-onnx branch. It defines its own ONNX wrapper/export logic.

Run from your cellseg_models.pytorch repo environment:

    uv pip install onnx onnxscript onnxruntime
    python stardist_onnx_standalone.py

By default it downloads StarDist's H&E example image (histo.jpg), because the
available pretrained cellseg_models.pytorch checkpoint is an H&E model.

To instead test the actual example TIFF from the StarDist repository, set:

    USE_REPO_TIFF = True

That TIFF is a fluorescence image. It is still valid for testing ONNX export,
numerical parity, and runtime speed, but segmentation quality is not meaningful
with the H&E-trained checkpoint.
"""

from __future__ import annotations

import platform
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.models.stardist import StarDist

WEIGHTS = "hgsc_v1_efficientnet_b5"
TILE_SIZE = 256
WARMUP = 5
REPEATS = 30
OPSET_VERSION = 18

# False = H&E repo example (recommended for this pretrained checkpoint)
# True  = actual StarDist example TIFF (fluorescence; parity/speed test only)
USE_REPO_TIFF = False

HISTO_URL = (
    "https://raw.githubusercontent.com/stardist/stardist/"
    "main/stardist/data/images/histo.jpg"
)
TIFF_URL = (
    "https://raw.githubusercontent.com/stardist/stardist/"
    "main/stardist/data/images/img2d.tif"
)

WORKDIR = Path("stardist_onnx_validation")
ONNX_PATH = WORKDIR / "stardist_hgsc.onnx"


class StarDistONNXWrapper(nn.Module):
    """Flatten cellseg_models.pytorch StarDist outputs into plain tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nuc = self.model(x)["nuc"]
        if nuc.binary_map is None:
            raise RuntimeError("StarDist model has no binary_map output.")
        return nuc.binary_map, nuc.aux_map, nuc.type_map


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[image] Using cached file: {path}")
        return path
    print(f"[image] Downloading:\n        {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[image] Saved to: {path}")
    return path


def load_image_as_rgb_tensor(path: Path, tile_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def export_onnx(
    network: nn.Module,
    output_path: Path,
    input_shape: tuple[int, int, int, int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StarDistONNXWrapper(network).eval()

    try:
        device = next(network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    example = torch.zeros(input_shape, dtype=torch.float32, device=device)
    batch_dim = torch.export.Dim("batch", min=1)

    print(f"[onnx] Exporting to {output_path} ...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(output_path),
            input_names=["image"],
            output_names=["binary_map", "ray_map", "type_map"],
            dynamic_shapes={"x": {0: batch_dim}},
            opset_version=OPSET_VERSION,
            dynamo=True,
        )

    print("[onnx] Export complete.")
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("[onnx] onnx.checker: PASS")
    return output_path


def to_postproc_input(
    outputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    binary_map, ray_map, type_logits = outputs
    return {
        "nuc": SoftInstanceOutput(
            type_map=torch.from_numpy(type_logits).argmax(1),
            aux_map=torch.from_numpy(ray_map),
            binary_map=torch.from_numpy(binary_map),
        ),
        "cyto": None,
        "tissue": None,
    }


def benchmark(
    fn: Callable[[], object], warmup: int, repeats: int
) -> tuple[float, float, object]:
    result = None
    for _ in range(warmup):
        result = fn()

    times_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(times_ms)), float(np.mean(times_ms)), result


def dense_parity(
    torch_outputs: tuple[np.ndarray, ...],
    ort_outputs: tuple[np.ndarray, ...],
) -> bool:
    names = ("binary_map", "ray_map", "type_map")
    print("\n=== Dense output parity ===")
    passed = True

    for name, expected, actual in zip(names, torch_outputs, ort_outputs):
        diff = np.abs(expected - actual)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        ok = np.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        print(
            f"{name:12s}: {'PASS' if ok else 'FAIL'} | "
            f"shape={actual.shape} | max_abs={max_abs:.3e} | "
            f"mean_abs={mean_abs:.3e}"
        )
        passed &= ok

    return passed


def postproc_parity(
    model: StarDist,
    torch_outputs: tuple[np.ndarray, ...],
    ort_outputs: tuple[np.ndarray, ...],
) -> bool:
    torch_result = model.post_processor.postproc_serial(
        to_postproc_input(torch_outputs)
    )["nuc"][0]
    ort_result = model.post_processor.postproc_serial(
        to_postproc_input(ort_outputs)
    )["nuc"][0]

    torch_instances, torch_types = torch_result
    ort_instances, ort_types = ort_result

    inst_equal = np.array_equal(torch_instances, ort_instances)
    type_equal = np.array_equal(torch_types, ort_types)
    inst_diff = int(np.count_nonzero(torch_instances != ort_instances))
    type_diff = int(np.count_nonzero(torch_types != ort_types))

    print("\n=== Final post-processing parity ===")
    print(
        "instance mask : "
        f"{'PASS' if inst_equal else 'FAIL'} (different pixels={inst_diff})"
    )
    print(
        "type mask     : "
        f"{'PASS' if type_equal else 'FAIL'} (different pixels={type_diff})"
    )
    print(f"PyTorch instances: {len(np.unique(torch_instances)) - 1}")
    print(f"ONNX instances   : {len(np.unique(ort_instances)) - 1}")

    return inst_equal and type_equal


def main() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("=== Environment ===")
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Platform      : {platform.platform()}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"ONNX          : {onnx.__version__}")
    print(f"ONNX Runtime  : {ort.__version__}")
    print(f"ORT providers : {ort.get_available_providers()}")

    if USE_REPO_TIFF:
        image_url = TIFF_URL
        image_path = WORKDIR / "stardist_img2d.tif"
        print(
            "\nNOTE: using StarDist img2d.tif, which is fluorescence. "
            "The checkpoint used below is H&E-trained, so judge parity/speed, "
            "not segmentation quality."
        )
    else:
        image_url = HISTO_URL
        image_path = WORKDIR / "stardist_histo.jpg"

    download(image_url, image_path)
    x = load_image_as_rgb_tensor(image_path, TILE_SIZE)

    print("\n=== Input ===")
    print(f"Source        : {image_url}")
    print(f"Tensor shape  : {tuple(x.shape)}")
    print(f"Tensor dtype  : {x.dtype}")
    print(f"Range         : [{float(x.min()):.4f}, {float(x.max()):.4f}]")

    print("\n=== Loading real pretrained checkpoint ===")
    print(f"Weights       : {WEIGHTS}")

    model = StarDist.from_pretrained(
        WEIGHTS,
        device=torch.device("cpu"),
    )
    model.set_inference_mode(mixed_precision=False)
    model.model.eval()

    wrapper = StarDistONNXWrapper(model.model).eval()

    export_onnx(
        model.model,
        ONNX_PATH,
        input_shape=(1, 3, TILE_SIZE, TILE_SIZE),
    )

    print("\n=== Creating ONNX Runtime session ===")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    print("ORT outputs    :", [out.name for out in session.get_outputs()])

    x_np = x.numpy()

    def run_torch():
        with torch.inference_mode():
            outputs = wrapper(x)
        return tuple(tensor.detach().cpu().numpy() for tensor in outputs)

    def run_ort():
        return tuple(session.run(None, {"image": x_np}))

    torch_outputs = run_torch()
    ort_outputs = run_ort()

    dense_ok = dense_parity(torch_outputs, ort_outputs)
    postproc_ok = postproc_parity(model, torch_outputs, ort_outputs)

    print("\n=== CPU forward benchmark ===")
    torch_median, torch_mean, _ = benchmark(run_torch, WARMUP, REPEATS)
    ort_median, ort_mean, _ = benchmark(run_ort, WARMUP, REPEATS)
    speedup = torch_median / ort_median

    print(f"PyTorch median : {torch_median:8.3f} ms")
    print(f"PyTorch mean   : {torch_mean:8.3f} ms")
    print(f"ORT median     : {ort_median:8.3f} ms")
    print(f"ORT mean       : {ort_mean:8.3f} ms")
    print(f"ORT speedup    : {speedup:8.3f}x")

    print("\n=== Result ===")
    if dense_ok and postproc_ok:
        print(
            "PASS: ONNX matches PyTorch for both dense outputs "
            "and final StarDist segmentation."
        )
    else:
        print("FAIL: parity mismatch detected. See diagnostics above.")
        raise SystemExit(1)

    print(f"\nONNX model saved at: {ONNX_PATH.resolve()}")


if __name__ == "__main__":
    main()
