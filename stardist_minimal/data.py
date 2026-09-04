from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tifffile import imread as tif_imread

from .targets import gen_dist_map, gen_stardist_maps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
MASK_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}


def _files(directory: str | Path, suffixes: set[str]) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in files:
                raise ValueError(f"Duplicate stem {path.stem!r} in {directory}")
            files[path.stem] = path
    return files


def paired_paths(
    images_dir: str | Path, masks_dir: str | Path
) -> list[tuple[Path, Path]]:
    images = _files(images_dir, IMAGE_SUFFIXES)
    masks = _files(masks_dir, MASK_SUFFIXES)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Image/mask stems do not match. Missing masks={missing_masks[:5]}, "
            f"missing images={missing_images[:5]}"
        )
    if not images:
        raise ValueError(f"No supported images found in {images_dir}")
    return [(images[stem], masks[stem]) for stem in sorted(images)]


def _read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".tif", ".tiff"}:
        return np.asarray(tif_imread(path))
    return np.asarray(Image.open(path))


def read_image(path: str | Path) -> np.ndarray:
    image = _read_array(Path(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"Expected 2D/RGB image, got shape {image.shape} from {path}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Expected 1 or >=3 channels, got shape {image.shape} from {path}")

    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, (0.01, 99.99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    mask = np.asarray(_read_array(Path(path))).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask, got shape {mask.shape} from {path}")
    if np.issubdtype(mask.dtype, np.floating) and not np.all(mask == np.floor(mask)):
        raise ValueError(f"Instance mask must contain integer labels: {path}")
    mask = mask.astype(np.int32, copy=False)
    if mask.min(initial=0) < 0:
        raise ValueError(f"Instance mask labels must be >=0: {path}")
    return mask


def _pad_to_patch(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    pad_h = max(0, size - image.shape[0])
    pad_w = max(0, size - image.shape[1])
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, mask


def _crop(
    image: np.ndarray, mask: np.ndarray, size: int, training: bool
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_patch(image, mask, size)
    max_y = image.shape[0] - size
    max_x = image.shape[1] - size
    if training:
        y = random.randint(0, max_y) if max_y else 0
        x = random.randint(0, max_x) if max_x else 0
    else:
        y, x = max_y // 2, max_x // 2
    return image[y : y + size, x : x + size], mask[y : y + size, x : x + size]


def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    if random.random() < 0.5:
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    k = random.randrange(4)
    if k:
        image, mask = np.rot90(image, k), np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class StarDistDataset(Dataset):
    """Paired image/instance-mask dataset that creates StarDist targets on demand."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        n_rays: int = 32,
        patch_size: int = 256,
        training: bool = False,
    ) -> None:
        if patch_size <= 0 or patch_size % 16:
            raise ValueError("patch_size must be a positive multiple of 16")
        self.pairs = paired_paths(images_dir, masks_dir)
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image, mask = read_image(image_path), read_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Shape mismatch for {image_path.name}: image={image.shape[:2]}, mask={mask.shape}"
            )
        if self.training:
            image, mask = _augment(image, mask)
        image, mask = _crop(image, mask, self.patch_size, self.training)

        score = gen_dist_map(mask)
        rays = gen_stardist_maps(mask, self.n_rays)
        foreground = (mask > 0).astype(np.float32)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "score": torch.from_numpy(score[None]),
            "rays": torch.from_numpy(np.ascontiguousarray(rays)),
            "foreground": torch.from_numpy(foreground[None]),
            "instance": torch.from_numpy(mask.astype(np.int64, copy=False)),
        }
