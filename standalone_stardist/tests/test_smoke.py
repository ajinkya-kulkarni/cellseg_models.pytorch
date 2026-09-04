import numpy as np
import torch

from stardist_isolated import StarDist, gen_stardist_maps, postprocess_stardist


def test_forward_shapes() -> None:
    model = StarDist(
        n_rays=16,
        enc_name="resnet18",
        enc_pretrain=False,
        out_channels=(128, 64, 32, 16),
    )
    x = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        out = model(x)

    assert set(out) == {"objectness", "rays"}
    assert out["objectness"].shape == (2, 1, 64, 64)
    assert out["rays"].shape == (2, 16, 64, 64)


def test_target_shape_and_background() -> None:
    labels = np.zeros((17, 17), dtype=np.int32)
    labels[5:12, 5:12] = 1

    rays = gen_stardist_maps(labels, n_rays=8)

    assert rays.shape == (8, 17, 17)
    assert rays.dtype == np.float32
    assert np.all(rays[:, 0, 0] == 0)
    assert np.all(rays[:, 8, 8] > 0)


def test_empty_postprocess() -> None:
    objectness = np.zeros((32, 32), dtype=np.float32)
    rays = np.ones((8, 32, 32), dtype=np.float32)

    labels = postprocess_stardist(objectness, rays)

    assert labels.shape == (32, 32)
    assert labels.dtype == np.int32
    assert labels.max() == 0
