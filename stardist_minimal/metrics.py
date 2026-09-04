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
        tp = 0
        fp = int(len(pred_ids))
        fn = int(len(gt_ids))
        return {"tp": tp, "fp": fp, "fn": fn, "iou_sum": 0.0}

    gt_index = {int(label): i for i, label in enumerate(gt_ids)}
    pred_index = {int(label): i for i, label in enumerate(pred_ids)}
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)

    both = (truth > 0) & (prediction > 0)
    for gt, pred in zip(truth[both].ravel(), prediction[both].ravel()):
        intersections[gt_index[int(gt)], pred_index[int(pred)]] += 1

    gt_area = np.asarray([(truth == label).sum() for label in gt_ids], dtype=np.int64)
    pred_area = np.asarray([(prediction == label).sum() for label in pred_ids], dtype=np.int64)
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
