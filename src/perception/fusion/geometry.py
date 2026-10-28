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
