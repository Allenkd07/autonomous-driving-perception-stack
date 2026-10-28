"""Box overlap and one-to-one matching, shared by trackers and evaluation."""

from __future__ import annotations

import lap
import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) IoU between xyxy boxes a (N, 4) and b (M, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost: np.ndarray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost one-to-one matching; pairs above max_cost stay unmatched.

    Returns (row, col) matches, unmatched rows and unmatched cols.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    _, row_to_col, col_to_row = lap.lapjv(cost.astype(np.float64), extend_cost=True, cost_limit=max_cost)
    matches = [(r, int(c)) for r, c in enumerate(row_to_col) if c >= 0]
    unmatched_rows = [r for r in range(rows) if row_to_col[r] < 0]
    return matches, unmatched_rows, [c for c in range(cols) if col_to_row[c] < 0]
