from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .model import StarDist
from .postprocess import postprocess_stardist


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: StarDist,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def load_model(path: str | Path, device: torch.device) -> tuple[StarDist, dict[str, Any]]:
    checkpoint = load_checkpoint(path, device)
    config = checkpoint["config"]
    model = StarDist(
        n_rays=int(config["n_rays"]),
        encoder_name=str(config["encoder"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_dense(
    model: StarDist, image: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run dense StarDist prediction on one normalized HWC image."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected normalized HWC RGB image, got {image.shape}")
    h, w = image.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32)
    pad_h, pad_w = (-h) % 32, (-w) % 32
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    with torch.inference_mode():
        score, rays = model(x)
    score = score[0, 0, :h, :w].float().cpu().numpy()
    rays = rays[0, :, :h, :w].float().cpu().numpy()
    return score, rays


def predict_instances(
    model: StarDist,
    image: np.ndarray,
    device: torch.device,
    score_thresh: float = 0.4,
    iou_thresh: float = 0.4,
) -> np.ndarray:
    score, rays = predict_dense(model, image, device)
    return postprocess_stardist(
        score,
        rays,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
    )
