"""Multi-object tracking metrics: CLEAR MOT (MOTA, MOTP, misses, false positives, ID switches) and identity
metrics (IDF1, IDP, IDR), computed by py-motmetrics.

MOTChallenge protocol: a prediction matches a ground-truth box at IoU >= 0.5, and predictions matched to an
ignore region (an object too small or hidden to score, or one without ground truth, such as a parked car
that is part of the CARLA map) are removed before scoring instead of counting as false positives.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from perception.core.matching import assign, iou_matrix

# motmetrics metric name -> what it means. MOTP here is the mean (1 - IoU) of matches: lower is better.
METRICS = {
    "mota": "MOTA",
    "idf1": "IDF1",
    "num_switches": "ID switches",
    "num_false_positives": "false positives",
    "num_misses": "misses",
    "motp": "MOTP (1 - IoU)",
    "idp": "ID precision",
    "idr": "ID recall",
    "precision": "precision",
    "recall": "recall",
    "num_fragmentations": "fragmentations",
    "mostly_tracked": "mostly tracked",
    "mostly_lost": "mostly lost",
    "num_unique_objects": "objects",
    "num_frames": "frames",
}


def drop_ignored(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, ignore_boxes: np.ndarray, min_iou: float = 0.5
) -> np.ndarray:
    """Mask of predictions to keep: those not matched to an ignore region.

    Scored and ignored ground truth are matched together, one-to-one, so a prediction that overlaps both a
    scored object and an ignore region goes to whichever fits better.
    """
    keep = np.ones(len(pred_boxes), bool)
    if len(pred_boxes) == 0 or len(ignore_boxes) == 0:
        return keep
    targets = np.concatenate([gt_boxes, ignore_boxes])
    matches, _, _ = assign(1.0 - iou_matrix(pred_boxes, targets), 1.0 - min_iou)
    for p, t in matches:
        if t >= len(gt_boxes):
            keep[p] = False
    return keep


class TrackingEvaluator:
    """Accumulates one sequence's frames; summarize() combines sequences."""

    def __init__(self, min_iou: float = 0.5) -> None:
        import motmetrics as mm

        self.min_iou = min_iou
        self.acc = mm.MOTAccumulator(auto_id=False)

    def update(
        self,
        frame_id: int,
        gt_ids: Sequence[int],
        gt_boxes: np.ndarray,
        pred_ids: Sequence[int],
        pred_boxes: np.ndarray,
        ignore_boxes: np.ndarray | None = None,
    ) -> None:
        """Boxes are xyxy; ids identify objects (ground truth) and tracks (predictions) across frames."""
        gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
        pred_boxes = np.asarray(pred_boxes, np.float32).reshape(-1, 4)
        pred_ids = np.asarray(pred_ids)
        if ignore_boxes is not None:
            keep = drop_ignored(pred_boxes, gt_boxes, np.asarray(ignore_boxes, np.float32).reshape(-1, 4))
            pred_boxes, pred_ids = pred_boxes[keep], pred_ids[keep]
        distances = 1.0 - iou_matrix(gt_boxes, pred_boxes)
        distances[distances > 1.0 - self.min_iou] = np.nan  # motmetrics: nan = may not match
        self.acc.update(list(gt_ids), list(pred_ids), distances, frameid=frame_id)

    def summary(self) -> dict[str, float]:
        return summarize([self])


def summarize(evaluators: Sequence[TrackingEvaluator]) -> dict[str, float]:
    """Metrics over all evaluators' sequences together (counts summed, ratios over the pooled counts)."""
    import motmetrics as mm

    names = [f"seq{i}" for i in range(len(evaluators))]
    table = mm.metrics.create().compute_many(
        [e.acc for e in evaluators], metrics=list(METRICS), names=names, generate_overall=True
    )
    return {name: float(table.loc["OVERALL", name]) for name in METRICS}
