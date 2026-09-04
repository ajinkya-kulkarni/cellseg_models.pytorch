from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def instance_stats(
    truth: np.ndarray, prediction: np.ndarray, iou_threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute TP/FP/FN and matched-IoU sums for labelled instance masks."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth={truth.shape}, prediction={prediction.shape}")

    gt_ids = np.unique(truth)
    pred_ids = np.unique(prediction)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return {
            "tp": 0,
            "fp": int(len(pred_ids)),
            "fn": int(len(gt_ids)),
            "iou_sum": 0.0,
        }

    gt_positive = truth > 0
    pred_positive = prediction > 0
    gt_area = np.bincount(
        np.searchsorted(gt_ids, truth[gt_positive]), minlength=len(gt_ids)
    )
    pred_area = np.bincount(
        np.searchsorted(pred_ids, prediction[pred_positive]), minlength=len(pred_ids)
    )

    both = gt_positive & pred_positive
    gt_idx = np.searchsorted(gt_ids, truth[both])
    pred_idx = np.searchsorted(pred_ids, prediction[both])
    pair_idx = gt_idx * len(pred_ids) + pred_idx
    intersections = np.bincount(
        pair_idx, minlength=len(gt_ids) * len(pred_ids)
    ).reshape(len(gt_ids), len(pred_ids))

    union = gt_area[:, None] + pred_area[None, :] - intersections
    iou = np.divide(
        intersections,
        union,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=union > 0,
    )

    rows, cols = linear_sum_assignment(-iou)
    matched = iou[rows, cols] >= iou_threshold
    tp = int(matched.sum())
    fp = int(len(pred_ids) - tp)
    fn = int(len(gt_ids) - tp)
    iou_sum = float(iou[rows[matched], cols[matched]].sum())
    return {"tp": tp, "fp": fp, "fn": fn, "iou_sum": iou_sum}


def pq_from_stats(tp: int, fp: int, fn: int, iou_sum: float) -> dict[str, float]:
    denom = 2 * tp + fp + fn
    dq = 1.0 if denom == 0 else (2.0 * tp) / denom
    sq = 1.0 if tp == 0 and denom == 0 else (iou_sum / tp if tp else 0.0)
    return {"dq": float(dq), "sq": float(sq), "pq": float(dq * sq)}


def panoptic_quality(
    truth: np.ndarray, prediction: np.ndarray, iou_threshold: float = 0.5
) -> dict[str, float | int]:
    stats = instance_stats(truth, prediction, iou_threshold=iou_threshold)
    scores = pq_from_stats(
        int(stats["tp"]), int(stats["fp"]), int(stats["fn"]), float(stats["iou_sum"])
    )
    return {**stats, **scores}
