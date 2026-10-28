import dataclasses

import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import DistanceEstimator
from perception.core.types import CameraInfo, Frame, LaneResult, Track
from perception.fusion.geometry import ground_point
from perception.fusion.ground_plane import GroundPlaneDistance
from perception.fusion.ttc import TTCEstimator

# 800x600, 90 degree FOV: focal length 400 px, principal point (400, 300); camera 1.6 m above the road
CAMERA = CameraInfo.from_fov(800, 600, 90.0, 1.6)


def project(x: float, z: float, camera: CameraInfo = CAMERA) -> tuple[float, float]:
    """Pixel of the road point (x right, z forward) for a camera pitched down by camera.pitch_rad."""
    p = np.array([x, camera.height_m, z])  # level frame: road is height_m below the camera (y down)
    c, s = np.cos(camera.pitch_rad), np.sin(camera.pitch_rad)
    cam = np.array([p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c])  # into the pitched camera frame
    u, v, w = camera.K @ cam
    return u / w, v / w


def test_ground_point_level_camera():
    # 10 m ahead: 400 * 1.6 / 10 = 64 px below the principal point
    assert ground_point(400, 364, CAMERA) == pytest.approx((0.0, 10.0))
    x, z = ground_point(*project(2.0, 25.0), CAMERA)
    assert (x, z) == pytest.approx((2.0, 25.0))
    assert ground_point(400, 300, CAMERA) is None and ground_point(400, 200, CAMERA) is None


def test_ground_point_pitched_camera():
    pitched = dataclasses.replace(CAMERA, pitch_rad=np.radians(5))

    assert ground_point(*project(-1.5, 30.0, pitched), pitched) == pytest.approx((-1.5, 30.0))


def test_ttc_from_an_approaching_object():
    ttc = TTCEstimator(window_s=0.5)
    # closing at 5 m/s from 20 m, 20 Hz, with +-5 cm noise
    rng = np.random.default_rng(0)
    for i in range(12):
        t = i * 0.05
        result = ttc.update(1, t, 20.0 - 5.0 * t + rng.uniform(-0.05, 0.05))

    assert result == pytest.approx((20.0 - 5.0 * 0.55) / 5.0, rel=0.1)


def test_no_ttc_for_receding_or_new_objects():
    ttc = TTCEstimator()
    for i in range(10):
        receding = ttc.update(1, i * 0.05, 10.0 + i * 0.1)
    assert receding is None
    assert TTCEstimator().update(2, 0.0, 10.0) is None  # one measurement is not a speed


def test_ttc_history_restarts_after_a_gap():
    ttc = TTCEstimator(max_gap_s=1.0)
    for i in range(10):
        ttc.update(1, i * 0.05, 20.0 - i * 0.25)

    assert ttc.update(1, 5.0, 10.0) is None  # seen again much later: no stale speed


def track(box, track_id=1) -> Track:
    return Track(box=np.array(box, np.float32), score=0.9, class_id=0, class_name="car", track_id=track_id)


def frame(t: float = 0.0) -> Frame:
    return Frame(image=np.zeros((600, 800, 3), np.uint8), timestamp=t, frame_id=0, camera=CAMERA)


def test_distance_and_in_lane_from_a_box():
    estimator = GroundPlaneDistance()
    _, v = project(0.0, 10.0)
    u_side, v_side = project(4.0, 20.0)

    ahead, beside = estimator.estimate(
        [track([360, v - 60, 440, v]), track([u_side - 30, v_side - 40, u_side + 30, v_side], 2)],
        None,
        None,
        frame(),
    )

    assert ahead.distance_m == pytest.approx(10.0) and ahead.in_ego_lane
    assert beside.distance_m == pytest.approx(20.0) and not beside.in_ego_lane  # 4 m to the side
    assert ahead.ttc_s is None  # needs a history first
    assert isinstance(estimator, DistanceEstimator)


def test_lane_mask_overrides_the_corridor():
    mask = np.zeros((600, 800), bool)
    mask[:, 500:] = True  # the detected ego lane is off to the right
    lanes = LaneResult(left=None, right=None, ego_lane_mask=mask, confidence=1.0)
    _, v = project(0.0, 10.0)

    (state,) = GroundPlaneDistance().estimate([track([360, v - 60, 440, v])], None, lanes, frame())

    assert not state.in_ego_lane


def test_box_above_the_horizon_has_no_distance():
    (state,) = GroundPlaneDistance().estimate([track([350, 100, 450, 250])], None, None, frame())

    assert state.distance_m is None and state.ttc_s is None


def test_ttc_over_frames_and_reset():
    estimator = GroundPlaneDistance()
    for i in range(10):
        t = i * 0.05
        _, v = project(0.0, 15.0 - 6.0 * t)  # closing at 6 m/s
        (state,) = estimator.estimate([track([360, v - 60, 440, v])], None, None, frame(t))

    assert state.ttc_s == pytest.approx((15.0 - 6.0 * 0.45) / 6.0, rel=0.05)
    estimator.reset()
    _, v = project(0.0, 10.0)
    (state,) = estimator.estimate([track([360, v - 60, 440, v])], None, None, frame(1.0))
    assert state.ttc_s is None


def test_registered_as_ground_plane():
    assert registry.get("distance", "ground_plane") is GroundPlaneDistance
