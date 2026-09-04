import numpy as np
import torch
from PIL import Image
from tifffile import imwrite

from stardist_minimal import (
    StarDist,
    StarDistLoss,
    gen_dist_map,
    gen_stardist_maps,
    panoptic_quality,
    postprocess_stardist,
)
from stardist_minimal.data import StarDistDataset
from stardist_minimal.runtime import predict_dense


def _mask(size=64):
    mask = np.zeros((size, size), dtype=np.int32)
    mask[20:40, 20:40] = 1
    return mask


def test_model_forward():
    model = StarDist(n_rays=16, encoder_name="resnet18", pretrained=False).eval()
    with torch.inference_mode():
        score, rays = model(torch.randn(1, 3, 64, 64))
    assert score.shape == (1, 1, 64, 64)
    assert rays.shape == (1, 16, 64, 64)
    assert torch.isfinite(score).all()
    assert torch.isfinite(rays).all()


def test_targets():
    mask = _mask()
    score = gen_dist_map(mask)
    rays = gen_stardist_maps(mask, 16)
    assert score.shape == mask.shape
    assert score.dtype == np.float32
    assert score.max() == 1.0
    assert score[30, 30] > score[20, 20]
    assert rays.shape == (16, 64, 64)
    assert rays[:, 30, 30].min() > 0
    assert np.all(rays[:, 0, 0] == 0)


def test_loss_backward():
    mask = _mask(32)
    target_score = torch.from_numpy(gen_dist_map(mask))[None, None]
    target_rays = torch.from_numpy(gen_stardist_maps(mask, 8))[None]
    foreground = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    score = torch.randn(1, 1, 32, 32, requires_grad=True)
    rays = torch.randn(1, 8, 32, 32, requires_grad=True)
    losses = StarDistLoss()(score, rays, target_score, target_rays, foreground)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert score.grad is not None
    assert rays.grad is not None


def test_postprocess_single_instance():
    score = np.zeros((64, 64), dtype=np.float32)
    rays = np.zeros((16, 64, 64), dtype=np.float32)
    score[32, 32] = 1.0
    rays[:, 32, 32] = 10.0
    labels = postprocess_stardist(score, rays)
    assert labels.shape == score.shape
    assert labels.dtype == np.int32
    assert labels.max() == 1


def test_panoptic_quality_perfect():
    mask = _mask()
    metrics = panoptic_quality(mask, mask)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["pq"] == 1.0


def test_dataset(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:40, 20:40] = (180, 100, 50)
    Image.fromarray(image).save(images / "sample.png")
    imwrite(masks / "sample.tif", _mask())

    dataset = StarDistDataset(images, masks, n_rays=8, patch_size=64, training=False)
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["score"].shape == (1, 64, 64)
    assert sample["rays"].shape == (8, 64, 64)
    assert sample["foreground"].shape == (1, 64, 64)


def test_full_image_dense_prediction_padding():
    model = StarDist(n_rays=8, encoder_name="resnet18", pretrained=False).eval()
    image = np.random.default_rng(0).random((65, 70, 3), dtype=np.float32)
    score, rays = predict_dense(model, image, torch.device("cpu"))
    assert score.shape == (65, 70)
    assert rays.shape == (8, 65, 70)
