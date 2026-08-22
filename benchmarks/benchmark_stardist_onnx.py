import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.models.stardist import StarDist, StarDistONNXWrapper


def _load_image(path: Path, tile_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def _to_soft_output(outputs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
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


def _time_call(fn, repeats: int, warmup: int) -> tuple[float, object]:
    result = None
    for _ in range(warmup):
        result = fn()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)

    return float(np.median(timings)), result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a pretrained StarDist checkpoint in PyTorch vs ONNX Runtime."
    )
    parser.add_argument("--image", type=Path, required=True, help="Real RGB image tile or image to resize.")
    parser.add_argument(
        "--weights",
        default="hgsc_v1_efficientnet_b5",
        help="Registered StarDist checkpoint name or local checkpoint path.",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("stardist.onnx"))
    args = parser.parse_args()

    if args.tile_size <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("tile-size and repeats must be positive; warmup must be non-negative.")

    x = _load_image(args.image, args.tile_size)
    model = StarDist.from_pretrained(args.weights, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)
    wrapper = StarDistONNXWrapper(model.model).eval()

    model.export_onnx(
        args.output,
        input_shape=(1, 3, args.tile_size, args.tile_size),
        dynamic_batch=True,
    )

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    x_np = x.numpy()

    def run_torch():
        with torch.inference_mode():
            return tuple(tensor.cpu().numpy() for tensor in wrapper(x))

    def run_ort():
        return tuple(session.run(None, {"image": x_np}))

    torch_time, torch_outputs = _time_call(run_torch, args.repeats, args.warmup)
    ort_time, ort_outputs = _time_call(run_ort, args.repeats, args.warmup)

    output_names = ("binary_map", "ray_map", "type_map")
    print("\nDense-output parity")
    for name, expected, actual in zip(output_names, torch_outputs, ort_outputs):
        max_abs = float(np.max(np.abs(expected - actual)))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        print(f"  {name}: PASS (max_abs={max_abs:.3e})")

    torch_post = model.post_processor.postproc_serial(_to_soft_output(torch_outputs))["nuc"][0]
    ort_post = model.post_processor.postproc_serial(_to_soft_output(ort_outputs))["nuc"][0]

    np.testing.assert_array_equal(ort_post[0], torch_post[0])
    np.testing.assert_array_equal(ort_post[1], torch_post[1])

    print("\nPost-processing parity")
    print("  instance mask: PASS")
    print("  type mask: PASS")

    print("\nCPU forward benchmark")
    print(f"  PyTorch median:      {torch_time * 1000:.2f} ms")
    print(f"  ONNX Runtime median: {ort_time * 1000:.2f} ms")
    print(f"  ORT speedup:         {torch_time / ort_time:.2f}x")


if __name__ == "__main__":
    main()
