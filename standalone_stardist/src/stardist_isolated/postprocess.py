"""
Standalone StarDist polygon reconstruction and NMS.

Polygon helpers are adapted from the StarDist project:
https://github.com/stardist/stardist

BSD 3-Clause License

Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
All rights reserved.
"""

from __future__ import annotations

import numpy as np
from skimage.draw import polygon
from skimage.morphology import disk, erosion

from .nms import get_bboxes, nms_stardist


def _ray_angles(n_rays: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def _dist_to_coord(dist: np.ndarray, points: np.ndarray) -> np.ndarray:
    n_rays = dist.shape[1]
    angles = _ray_angles(n_rays)
    coord = (
        dist[:, np.newaxis]
        * np.asarray([np.sin(angles), np.cos(angles)], dtype=np.float32)
    ).astype(np.float32)
    coord += points[..., np.newaxis]
    return coord


def _polygons_to_label(
    dist: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    points = points[order]
    dist = dist[order]

    coord = _dist_to_coord(dist, points)
    labels = np.zeros(shape, dtype=np.int32)

    for label, vertices in zip(order, coord):
        rr, cc = polygon(*vertices, shape)
        labels[rr, cc] = int(label) + 1

    return labels


def _threshold_mask(
    prob: np.ndarray,
    threshold: float,
    border: int = 2,
) -> np.ndarray:
    mask = prob > threshold
    if border > 0:
        valid = np.zeros_like(mask)
        valid[border:-border, border:-border] = True
        mask &= valid
    return mask


def postprocess_stardist(
    objectness_map: np.ndarray,
    ray_map: np.ndarray,
    score_thresh: float = 0.4,
    iou_thresh: float = 0.4,
    trim_bboxes: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Convert dense StarDist predictions into a 2D integer label mask."""
    objectness_map = np.asarray(objectness_map)
    ray_map = np.asarray(ray_map)

    if objectness_map.ndim != 2:
        raise ValueError(
            f"objectness_map must have shape (H, W); got {objectness_map.shape}"
        )
    if ray_map.ndim != 3:
        raise ValueError(
            f"ray_map must have shape (n_rays, H, W); got {ray_map.shape}"
        )
    if objectness_map.shape != ray_map.shape[1:]:
        raise ValueError(
            "objectness_map and ray_map spatial shapes must match; "
            f"got {objectness_map.shape} and {ray_map.shape[1:]}"
        )

    shape = objectness_map.shape
    dist = ray_map.transpose(1, 2, 0).astype(np.float32)
    prob = objectness_map.astype(np.float32)

    if normalize:
        lo = float(prob.min())
        hi = float(prob.max())
        prob = (prob - lo) / (hi - lo) if hi > lo else np.zeros_like(prob)

    mask = _threshold_mask(prob, score_thresh)

    if trim_bboxes and mask.any():
        mask = mask.astype(np.int32)
        mask -= erosion(mask, disk(2)).astype(np.int32)
        mask = mask > 0

    points = np.stack(np.where(mask), axis=1)
    if len(points) == 0:
        return np.zeros(shape, dtype=np.int32)

    dist = dist[mask]
    scores = prob[mask]

    order = np.argsort(scores)[::-1]
    dist = dist[order]
    scores = scores[order]
    points = points[order]

    x1, y1, x2, y2, areas, max_dist = get_bboxes(dist, points)
    boxes = np.stack((x1, y1, x2, y2), axis=1)

    keep = nms_stardist(
        boxes,
        points,
        scores,
        areas,
        max_dist,
        score_threshold=score_thresh,
        iou_threshold=iou_thresh,
    )

    if len(keep) == 0:
        return np.zeros(shape, dtype=np.int32)

    return _polygons_to_label(
        dist[keep],
        points[keep],
        scores[keep],
        shape,
    )
