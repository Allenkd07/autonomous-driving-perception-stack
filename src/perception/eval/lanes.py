"""Lane detection metrics against BDD100K ground truth (perception.io.bdd100k_lanes).

ego lane       predicted ego_lane_mask vs. the `area/drivable` polygon: IoU per image (mean over images
               that have a drivable area), plus pixel precision and recall pooled over all images
lane markings  predicted marking pixels vs. ground-truth markings, with a distance tolerance: a predicted
               pixel counts as correct within `tolerance_px` of a marking, and a marking pixel as found
               within `tolerance_px` of a prediction (lines a few pixels off are still the right line)
"""

from __future__ import annotations

import cv2
import numpy as np


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


class LaneEvaluator:
    def __init__(self, tolerance_px: int = 5) -> None:
        self.tolerance_px = tolerance_px
        self.ious: list[float] = []
        self.ego = {"tp": 0, "fp": 0, "fn": 0}
        self.marking = {"pred_hit": 0, "pred": 0, "gt_found": 0, "gt": 0}

    def update_ego(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Masks are (H, W) bool at the same size. Images without a ground-truth ego lane are skipped."""
        if not gt.any():
            return
        tp = int((pred & gt).sum())
        fp, fn = int((pred & ~gt).sum()), int((~pred & gt).sum())
        self.ious.append(tp / (tp + fp + fn))
        for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
            self.ego[key] += value

    def update_markings(self, pred: np.ndarray, gt: np.ndarray) -> None:
        self.marking["pred_hit"] += int((pred & _dilate(gt, self.tolerance_px)).sum())
        self.marking["pred"] += int(pred.sum())
        self.marking["gt_found"] += int((gt & _dilate(pred, self.tolerance_px)).sum())
        self.marking["gt"] += int(gt.sum())

    def summary(self) -> dict[str, float]:
        precision = _ratio(self.marking["pred_hit"], self.marking["pred"])
        recall = _ratio(self.marking["gt_found"], self.marking["gt"])
        return {
            "ego_iou": float(np.mean(self.ious)) if self.ious else float("nan"),
            "ego_precision": _ratio(self.ego["tp"], self.ego["tp"] + self.ego["fp"]),
            "ego_recall": _ratio(self.ego["tp"], self.ego["tp"] + self.ego["fn"]),
            "ego_images": len(self.ious),
            "marking_precision": precision,
            "marking_recall": recall,
            "marking_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
