# StarDist-only cell segmentation

This branch contains the complete single-class 2D StarDist workflow extracted from
`cellseg_models.pytorch`, without the other model families or the generic multitask
framework.

It includes:

- timm encoder + StarDist U-Net decoder
- normalized per-instance EDT score target
- radial-distance targets
- paired image/instance-mask dataset
- geometric training augmentation
- StarDist loss
- AdamW + ReduceLROnPlateau training
- validation and resumable checkpoints
- full-image inference + NMS/polygon reconstruction
- DQ/SQ/PQ evaluation
- prediction CLI
- tests

## Install

```bash
uv sync --extra test
```

## Data layout

Images and instance masks are paired by filename stem. Background must be `0` and each
object must have a positive integer ID.

```text
data/
├── train/
│   ├── images/
│   │   ├── 001.png
│   │   └── 002.png
│   └── masks/
│       ├── 001.tif
│       └── 002.tif
└── val/
    ├── images/
    └── masks/
```

Images: PNG/JPEG/TIFF/NPY. Masks: PNG/TIFF/NPY.

Images are percentile-normalized to `[0, 1]`. During training, paired image/mask
flips and 90-degree rotations are applied before StarDist targets are generated.

## Train

```bash
uv run stardist-train \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --output-dir runs/stardist \
  --encoder efficientnet_b5 \
  --n-rays 32 \
  --patch-size 256 \
  --batch-size 8 \
  --epochs 100 \
  --lr 3e-4
```

Use `--amp` on CUDA for mixed precision. Use `--no-pretrained` to avoid ImageNet
encoder weights.

Training writes:

```text
runs/stardist/
├── config.json
├── history.json
├── best.pt
└── last.pt
```

Resume with:

```bash
uv run stardist-train \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --output-dir runs/stardist \
  --epochs 150 \
  --resume runs/stardist/last.pt
```

The checkpoint's encoder, ray count, and patch size are reused when resuming.

## Evaluate / test

Evaluation runs full-image inference, reconstructs instances, matches predictions to
GT instances at IoU 0.5, and reports DQ, SQ, and PQ.

```bash
uv run stardist-evaluate \
  --checkpoint runs/stardist/best.pt \
  --images data/test/images \
  --masks data/test/masks \
  --save-dir runs/stardist/test-predictions
```

Post-processing thresholds can be changed independently:

```bash
uv run stardist-evaluate \
  --checkpoint runs/stardist/best.pt \
  --images data/test/images \
  --masks data/test/masks \
  --score-thresh 0.4 \
  --nms-iou-thresh 0.4 \
  --match-iou-thresh 0.5
```

## Predict

```bash
uv run stardist-predict \
  --checkpoint runs/stardist/best.pt \
  --input data/inference/images \
  --output-dir predictions
```

Predictions are saved as `int32` labelled `.npy` instance masks.

## Python API

```python
import torch
from stardist_minimal import StarDist

model = StarDist(n_rays=32, encoder_name="resnet18", pretrained=False).eval()
with torch.inference_mode():
    score_map, ray_maps = model(torch.randn(1, 3, 256, 256))

print(score_map.shape)  # (1, 1, 256, 256)
print(ray_maps.shape)   # (1, 32, 256, 256)
```

## Test the package

```bash
uv run pytest -q
```

This branch is intentionally single-class. It does not include nuclei-type heads,
Cellpose, HoVer-Net, CPP-Net, CellViT, InstanSeg, WSI readers, or the original generic
multitask model/loss/predictor abstractions.
