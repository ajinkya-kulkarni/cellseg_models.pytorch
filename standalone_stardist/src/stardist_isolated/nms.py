from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.spatial import KDTree


@njit
def _intersection(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@njit
def get_bboxes(
    dist: np.ndarray,
    points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Get axis-aligned candidate boxes from StarDist rays."""
    n_polys = dist.shape[0]
    n_rays = dist.shape[1]

    bbox_x1 = np.zeros(n_polys)
    bbox_x2 = np.zeros(n_polys)
    bbox_y1 = np.zeros(n_polys)
    bbox_y2 = np.zeros(n_polys)
    areas = np.zeros(n_polys)

    angle_step = 2 * math.pi / n_rays
    max_dist = 0.0

    for i in range(n_polys):
        max_radius_outer = 0.0
        py = points[i, 0]
        px = points[i, 1]

        for k in range(n_rays):
            d = dist[i, k]
            y = py + d * np.sin(angle_step * k)
            x = px + d * np.cos(angle_step * k)

            if k == 0:
                bbox_x1[i] = x
                bbox_x2[i] = x
                bbox_y1[i] = y
                bbox_y2[i] = y
            else:
                bbox_x1[i] = min(x, bbox_x1[i])
                bbox_x2[i] = max(x, bbox_x2[i])
                bbox_y1[i] = min(y, bbox_y1[i])
                bbox_y2[i] = max(y, bbox_y2[i])

            max_radius_outer = max(d, max_radius_outer)

        areas[i] = (bbox_x2[i] - bbox_x1[i]) * (bbox_y2[i] - bbox_y1[i])
        max_dist = max(max_dist, max_radius_outer)

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2, areas, max_dist


@njit
def _suppress_bbox(
    query: np.ndarray,
    current_idx: int,
    boxes: np.ndarray,
    areas: np.ndarray,
    suppressed: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    for i in range(len(query)):
        query_idx = query[i]

        if query_idx == current_idx or suppressed[query_idx]:
            continue

        overlap = _intersection(boxes[current_idx], boxes[query_idx])
        denom = min(areas[current_idx] + 1e-10, areas[query_idx] + 1e-10)
        suppressed[query_idx] = (overlap / denom) > iou_threshold

    return suppressed


def nms_stardist(
    boxes: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    max_dist: float,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """KDTree-accelerated NMS used by the repository's StarDist postprocess."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    keep: list[int] = []
    kdtree = KDTree(points, leafsize=16)
    suppressed = np.zeros(len(boxes), dtype=np.bool_)

    for current_idx in range(len(scores)):
        if suppressed[current_idx]:
            continue
        if scores[current_idx] < score_threshold:
            break

        query = np.asarray(
            kdtree.query_ball_point(points[current_idx], max_dist),
            dtype=np.int64,
        )
        suppressed = _suppress_bbox(
            query,
            current_idx,
            boxes,
            areas,
            suppressed,
            iou_threshold,
        )
        keep.append(current_idx)

    return np.asarray(keep, dtype=np.int64)
