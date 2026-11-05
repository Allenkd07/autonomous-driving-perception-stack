"""Monocular ground-plane geometry: where a pixel's viewing ray meets a flat road.

Camera coordinates follow perception.core.types: x right, y down, z forward. The camera sits height_m
above the road and is pitched down by pitch_rad (positive = looking down). Results are in a level frame
centred under the camera: x to the right and z forward along the road, in metres.
"""

from __future__ import annotations

import numpy as np

from perception.core.types import CameraInfo

MIN_RAY_DOWN = 1e-3  # rays this close to the horizon (or above it) never meet the road


def ground_point(u: float, v: float, camera: CameraInfo) -> tuple[float, float] | None:
    """(x, z) on the road of the pixel (u, v), or None if its ray points at or above the horizon."""
    ray = np.linalg.solve(camera.K, np.array([u, v, 1.0]))
    c, s = np.cos(camera.pitch_rad), np.sin(camera.pitch_rad)
    down = ray[1] * c + ray[2] * s  # undo the pitch: rotate the ray into the level frame
    forward = -ray[1] * s + ray[2] * c
    if down <= MIN_RAY_DOWN:
        return None
    t = camera.height_m / down
    return float(t * ray[0]), float(t * forward)


def road_depth(rows: np.ndarray, camera: CameraInfo) -> np.ndarray:
    """Depth (along the optical axis, as a depth camera measures it) of the flat road seen at each image row;
    NaN for rows at or above the horizon. The same for every column of a row: the ray through pixel (u, v)
    is K^-1 (u, v, 1), whose z is 1, so its depth is the ray's scale where it meets the road."""
    fy, cy = camera.K[1, 1], camera.K[1, 2]
    down = (np.asarray(rows, np.float64) - cy) / fy * np.cos(camera.pitch_rad) + np.sin(camera.pitch_rad)
    with np.errstate(divide="ignore"):
        return np.where(down > MIN_RAY_DOWN, camera.height_m / down, np.nan)
