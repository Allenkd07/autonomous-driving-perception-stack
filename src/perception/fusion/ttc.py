"""Time to collision per track from its recent distance history.

The closing speed is the slope of a straight-line fit to (time, distance) over the last window_s seconds,
which averages out per-frame noise in the distance estimate far better than differencing two frames.
TTC = distance / closing speed, reported only while the object is approaching faster than
min_closing_speed (a stationary or receding object has no time to collision).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class TTCEstimator:
    def __init__(
        self,
        window_s: float = 0.5,
        min_points: int = 4,
        min_closing_speed: float = 0.5,
        max_gap_s: float = 1.0,
    ) -> None:
        self.window_s = window_s
        self.min_points = min_points
        self.min_closing_speed = min_closing_speed
        self.max_gap_s = max_gap_s  # a track unseen for longer starts a fresh history
        self._history: dict[int, deque[tuple[float, float]]] = {}

    def reset(self) -> None:
        self._history.clear()

    def update(self, track_id: int, timestamp: float, distance_m: float) -> float | None:
        """Add a measurement and return the track's TTC in seconds, or None."""
        history = self._history.setdefault(track_id, deque())
        if history and timestamp - history[-1][0] > self.max_gap_s:
            history.clear()
        history.append((timestamp, distance_m))
        while history[-1][0] - history[0][0] > self.window_s:
            history.popleft()
        if len(history) < self.min_points:
            return None
        t, d = np.array(history).T
        closing_speed = -np.polyfit(t - t[-1], d, 1)[0]
        if closing_speed < self.min_closing_speed:
            return None
        return float(distance_m / closing_speed)

    def forget_older_than(self, timestamp: float) -> None:
        """Drop tracks with no measurement in the last max_gap_s, so the table doesn't grow."""
        self._history = {k: h for k, h in self._history.items() if timestamp - h[-1][0] <= self.max_gap_s}
