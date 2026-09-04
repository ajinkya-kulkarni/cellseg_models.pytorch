# Standalone StarDist

This directory isolates the 2D StarDist pieces needed for a single-class
instance-segmentation benchmark without importing the generic
`cellseg_models_pytorch` framework.

Included:

- a timm encoder;
- a dedicated U-Net decoder matching the existing StarDist defaults:
  fixed unpooling, concatenative U-Net skips, and two bias-free 3x3
  Conv+ReLU blocks per decoder stage;
- independent 128-channel excitation heads for objectness and radial distances;
- StarDist radial-distance target generation;
- the repository's Python/Numba/KDTree StarDist post-processing and NMS;
- optional dense-network ONNX export.

Intentionally excluded:

- `BaseModelInst`;
- `Predictor` / `PostProcessor`;
- `MultiTaskDecoder`;
- generic decoder/module registries;
- nuclei type classification;
- WSI, vectorization, file-management, metrics, and other model code.

## Outputs

```python
from stardist_isolated import StarDist

model = StarDist(
    n_rays=32,
    enc_name="resnet18",
    enc_pretrain=False,
)

out = model(image)
objectness = out["objectness"]  # (B, 1, H, W), raw network values
rays = out["rays"]              # (B, n_rays, H, W), raw distances
```

For post-processing:

```python
from stardist_isolated import postprocess_stardist

labels = postprocess_stardist(
    objectness[0, 0].detach().cpu().numpy(),
    rays[0].detach().cpu().numpy(),
)
```

## Install

From this directory:

```bash
pip install -e .
```

For ONNX export:

```bash
pip install -e ".[onnx]"
```

## Scope

This is an isolated benchmark implementation, not a replacement for the
existing public `cellseg_models_pytorch.models.stardist.StarDist` API.

The computation is deliberately single-class: the existing `nuc_type` head is
not included. Existing framework checkpoints therefore do not load by key
without a conversion step, although the default encoder/decoder operations are
kept aligned with the current StarDist implementation.
