"""Metrics for per-object distance and dense depth estimates against ground truth (metres).

Per-object errors follow the monocular depth literature, applied to one distance per object:
    abs_rel     mean |estimate - truth| / truth    (relative: 1 m off matters more at 5 m than at 50 m)
    mae_m       median |estimate - truth| in metres (robust to the odd wild estimate)
    rmse_m      root mean squared error in metres  (punishes the wild ones)
    delta1      share with max(estimate / truth, truth / estimate) < 1.25
    coverage    share of objects that got an estimate at all; the errors are over those only
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DISTANCE_BINS = (0.0, 10.0, 20.0, 40.0, np.inf)


def distance_errors(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Errors over objects; NaN estimates count as not covered."""
    truth = np.asarray(truth, np.float64)
    estimate = np.asarray(estimate, np.float64)
    ok = np.isfinite(estimate) & (estimate > 0)
    n = len(truth)
    if not ok.any():
        return {"n": n, "coverage": 0.0 if n else float("nan"), **dict.fromkeys(
            ("abs_rel", "mae_m", "rmse_m", "delta1"), float("nan"))}  # fmt: skip
    t, e = truth[ok], estimate[ok]
    err = e - t
    return {
        "n": n,
        "coverage": float(ok.mean()),
        "abs_rel": float(np.mean(np.abs(err) / t)),
        "mae_m": float(np.median(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "delta1": float(np.mean(np.maximum(e / t, t / e) < 1.25)),
    }


def distance_table(objects: pd.DataFrame, methods: list[str], by: str | None = None) -> pd.DataFrame:
    """One row per method (and value of column `by`) with distance_errors of column method vs 'truth'."""
    rows = []
    groups = objects.groupby(by, observed=True) if by else [(None, objects)]
    for key, part in groups:
        for method in methods:
            row = {"method": method} if by is None else {by: key, "method": method}
            rows.append({**row, **distance_errors(part["truth"].to_numpy(), part[method].to_numpy())})
    return pd.DataFrame(rows)


def distance_bin(truth: np.ndarray, bins: tuple[float, ...] = DISTANCE_BINS) -> pd.Categorical:
    edges = zip(bins[:-1], bins[1:], strict=True)
    labels = [f"{lo:.0f}-{hi:.0f} m" if np.isfinite(hi) else f"{lo:.0f}+ m" for lo, hi in edges]
    return pd.cut(np.asarray(truth), bins=list(bins), labels=labels, right=False)


def dense_abs_rel(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    """Mean |pred - truth| / truth over valid pixels with a finite prediction (a calibrated map is inf
    where it puts a pixel at or beyond infinity; those are left out)."""
    ok = valid & np.isfinite(pred)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(pred[ok] - truth[ok]) / truth[ok]))
