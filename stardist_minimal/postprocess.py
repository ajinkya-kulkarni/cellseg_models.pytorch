import math

import numpy as np
from numba import njit
from scipy.spatial import KDTree
from skimage.draw import polygon
from skimage.morphology import disk, erosion


@njit
def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


@njit
def _bboxes(dist: np.ndarray, points: np.ndarray):
    n_polys, n_rays = dist.shape
    x1 = np.zeros(n_polys)
    y1 = np.zeros(n_polys)
    x2 = np.zeros(n_polys)
    y2 = np.zeros(n_polys)
    areas = np.zeros(n_polys)
    angle_step = 2 * math.pi / n_rays
    max_dist = 0.0

    for i in range(n_polys):
        py, px = points[i]
        for k in range(n_rays):
            d = dist[i, k]
            y = py + d * np.sin(angle_step * k)
            x = px + d * np.cos(angle_step * k)
            if k == 0:
                x1[i] = x2[i] = x
                y1[i] = y2[i] = y
            else:
                x1[i] = min(x1[i], x)
                x2[i] = max(x2[i], x)
                y1[i] = min(y1[i], y)
                y2[i] = max(y2[i], y)
            max_dist = max(max_dist, d)
        areas[i] = (x2[i] - x1[i]) * (y2[i] - y1[i])

    return np.stack((x1, y1, x2, y2), axis=1), areas, max_dist


@njit
def _suppress(query, current, boxes, areas, suppressed, threshold):
    for other in query:
        if suppressed[other]:
            continue
        overlap = _intersection(boxes[current], boxes[other])
        denom = min(areas[current] + 1e-10, areas[other] + 1e-10)
        suppressed[other] = overlap / denom > threshold
    return suppressed


def _nms(boxes, points, scores, areas, max_dist, threshold):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    tree = KDTree(points, leafsize=16)
    suppressed = np.zeros(len(boxes), dtype=np.bool_)
    keep = []
    for current in range(len(scores)):
        if suppressed[current]:
            continue
        query = np.asarray(tree.query_ball_point(points[current], max_dist), dtype=np.int64)
        suppressed = _suppress(query, current, boxes, areas, suppressed, threshold)
        keep.append(current)
    return np.asarray(keep, dtype=np.int64)


def _coordinates(dist: np.ndarray, points: np.ndarray) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, dist.shape[1], endpoint=False)
    coord = dist[:, None] * np.asarray(
        [np.sin(angles), np.cos(angles)], dtype=np.float32
    )
    return coord.astype(np.float32) + points[..., None]


def postprocess_stardist(
    prob_map: np.ndarray,
    dist_map: np.ndarray,
    score_thresh: float = 0.4,
    iou_thresh: float = 0.4,
) -> np.ndarray:
    """Convert StarDist probability and radial-distance maps to instance labels."""
    prob = np.asarray(prob_map, dtype=np.float32)
    dist = np.asarray(dist_map, dtype=np.float32).transpose(1, 2, 0)
    shape = prob.shape

    lo, hi = float(prob.min()), float(prob.max())
    prob = (prob - lo) / (hi - lo) if hi > lo else np.zeros_like(prob)

    mask = prob > score_thresh
    valid = np.zeros_like(mask)
    valid[2:-2, 2:-2] = True
    mask &= valid
    if mask.any():
        edge = mask.astype(np.int32) - erosion(mask, disk(2)).astype(np.int32)
        mask = edge > 0

    points = np.stack(np.where(mask), axis=1)
    if len(points) == 0:
        return np.zeros(shape, dtype=np.int32)

    dist = dist[mask]
    scores = prob[mask]
    order = np.argsort(scores)[::-1]
    dist, scores, points = dist[order], scores[order], points[order]

    boxes, areas, max_dist = _bboxes(dist, points)
    keep = _nms(boxes, points, scores, areas, max_dist, iou_thresh)
    coords = _coordinates(dist[keep], points[keep])

    labels = np.zeros(shape, dtype=np.int32)
    for label, vertices in enumerate(coords, 1):
        rr, cc = polygon(*vertices, shape)
        labels[rr, cc] = label
    return labels
