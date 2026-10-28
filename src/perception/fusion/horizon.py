"""Camera pitch from the lane lines' vanishing point, for cameras whose mounting is not known (dashcams).

On a straight, flat road the two ego-lane lines meet at the horizon. A camera pitched down by theta sees
the horizon theta above the image centre: v = cy - fy * tan(theta), so each frame with two good lane lines
measures the pitch. The median of recent measurements gives a stable estimate; on a slope it follows the
road ahead relative to the camera, which is the plane ground-plane distances need.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from perception.core.types import CameraInfo, LaneResult


def vanishing_point(left: np.ndarray, right: np.ndarray, rows: np.ndarray) -> tuple[float, float] | None:
    """Where straight lines through the two lane fits (x = f(y), sampled at rows) cross; None if parallel."""
    a1, b1 = np.polyfit(rows, np.polyval(left, rows), 1)
    a2, b2 = np.polyfit(rows, np.polyval(right, rows), 1)
    if abs(a1 - a2) < 1e-6:
        return None
    v = (b2 - b1) / (a1 - a2)
    return float(a1 * v + b1), float(v)


class PitchEstimator:
    """Median of recent per-frame pitch measurements from lanes with both lines found."""

    def __init__(self, window: int = 30, min_samples: int = 5, max_abs_pitch_deg: float = 10.0) -> None:
        self.min_samples = min_samples
        self.max_abs_pitch = np.radians(max_abs_pitch_deg)
        self._samples: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def pitch_rad(self) -> float | None:
        """The current estimate, or None until there are min_samples measurements."""
        return float(np.median(self._samples)) if len(self._samples) >= self.min_samples else None

    def update(self, lanes: LaneResult | None, camera: CameraInfo) -> float | None:
        if lanes is None or lanes.left is None or lanes.right is None or lanes.confidence < 1.0:
            return self.pitch_rad
        rows = self._rows(lanes, camera)
        vp = vanishing_point(lanes.left, lanes.right, rows)
        if vp is not None:
            fy, cy = camera.K[1, 1], camera.K[1, 2]
            pitch = float(np.arctan((cy - vp[1]) / fy))
            if abs(pitch) <= self.max_abs_pitch:
                self._samples.append(pitch)
        return self.pitch_rad

    @staticmethod
    def _rows(lanes: LaneResult, camera: CameraInfo) -> np.ndarray:
        """Rows to take the lines' direction from: the lower half of the detected ego lane (nearest the car,
        where a curved road is closest to straight), else the bottom third of the image."""
        if lanes.ego_lane_mask is not None and lanes.ego_lane_mask.any():
            filled = np.flatnonzero(lanes.ego_lane_mask.any(axis=1))
            top, bottom = filled.min(), filled.max()
            return np.linspace((top + bottom) / 2, bottom, 10)
        return np.linspace(camera.height * 2 / 3, camera.height - 1, 10)
