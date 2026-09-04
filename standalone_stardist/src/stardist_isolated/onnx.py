from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .model import StarDist


class StarDistONNXWrapper(nn.Module):
    """Flatten the standalone StarDist dictionary output for ONNX."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["objectness"], out["rays"]


def export_stardist_onnx(
    model: StarDist,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export only the dense StarDist neural network; postprocess stays in Python."""
    if len(input_shape) != 4 or any(dim <= 0 for dim in input_shape):
        raise ValueError("input_shape must contain four positive BCHW dimensions")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    wrapper = StarDistONNXWrapper(model).eval()
    example = torch.zeros(input_shape, dtype=torch.float32, device=device)

    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(output_path),
            input_names=["image"],
            output_names=["objectness", "rays"],
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            opset_version=opset_version,
        )

    return output_path
