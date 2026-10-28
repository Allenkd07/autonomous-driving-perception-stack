import dataclasses

import numpy as np
import pytest

from perception.core.types import CameraInfo, Frame, LaneResult, Track
from perception.fusion.ground_plane import GroundPlaneDistance
from perception.fusion.horizon import PitchEstimator, vanishing_point

LEVEL = CameraInfo.from_fov(800, 600, 90.0, 1.6)
PITCHED = dataclasses.replace(LEVEL, pitch_rad=np.radians(3.0))  # the real camera looks 3 degrees down


def project(x: float, z: float, camera: CameraInfo) -> tuple[float, float]:
    c, s = np.cos(camera.pitch_rad), np.sin(camera.pitch_rad)
    y = camera.height_m
    u, v, w = camera.K @ np.array([x, y * c - z * s, y * s + z * c])
    return u / w, v / w


def lanes_seen_by(camera: CameraInfo, half_width: float = 1.75) -> LaneResult:
    """Image-space lane fits x = f(y) of straight road lines at +-half_width, as a lane detector returns."""
    fits = []
    for x in (-half_width, half_width):
        u, v = np.array([project(x, z, camera) for z in np.linspace(6, 40, 20)]).T
        fits.append(np.polyfit(v, u, 2))
    return LaneResult(left=fits[0], right=fits[1], ego_lane_mask=None, confidence=1.0)


def test_vanishing_point_of_two_lines():
    left = np.array([0.0, -1.0, 700.0])  # x = 700 - y
    right = np.array([0.0, 1.0, 100.0])  # x = 100 + y

    assert vanishing_point(left, right, np.linspace(400, 600, 5)) == pytest.approx((400.0, 300.0))
    assert vanishing_point(left, left, np.linspace(400, 600, 5)) is None


def test_pitch_is_recovered_from_lanes():
    estimator = PitchEstimator(min_samples=3)
    lanes = lanes_seen_by(PITCHED)

    estimates = [estimator.update(lanes, LEVEL) for _ in range(3)]

    assert estimates[:2] == [None, None]  # not enough samples yet
    assert np.degrees(estimates[2]) == pytest.approx(3.0, abs=0.05)


def test_one_line_or_implausible_pitch_is_ignored():
    estimator = PitchEstimator(min_samples=1, max_abs_pitch_deg=10)
    one_line = dataclasses.replace(lanes_seen_by(PITCHED), confidence=0.5)
    steep = lanes_seen_by(dataclasses.replace(LEVEL, pitch_rad=np.radians(20)))

    assert estimator.update(one_line, LEVEL) is None
    assert estimator.update(steep, LEVEL) is None


def test_auto_pitch_corrects_distances():
    # A car 20 m ahead seen by the pitched camera, while the pipeline believes the camera is level
    u, v = project(0.0, 20.0, PITCHED)
    box = np.array([u - 40, v - 50, u + 40, v], np.float32)
    track = Track(box=box, score=0.9, class_id=0, class_name="car", track_id=1)
    lanes = lanes_seen_by(PITCHED)

    def distance_after(estimator: GroundPlaneDistance, frames: int = 6) -> float:
        for i in range(frames):
            frame = Frame(
                image=np.zeros((600, 800, 3), np.uint8), timestamp=i * 0.05, frame_id=i, camera=LEVEL
            )
            (state,) = estimator.estimate([track], None, lanes, frame)
        return state.distance_m

    assert distance_after(GroundPlaneDistance()) > 30  # wrong pitch: far too far
    assert distance_after(GroundPlaneDistance(auto_pitch=True)) == pytest.approx(20.0, rel=0.02)
