import dataclasses

import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import DistanceEstimator
from perception.core.types import CameraInfo, Frame, LaneResult, Track
from perception.fusion.base import in_ego_lane, lane_span
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


def test_lane_lines_override_the_corridor():
    mask = np.zeros((600, 800), bool)
    mask[:, 500:700] = True  # the detected ego lane is off to the right
    lanes = LaneResult(
        left=np.array([0.0, 0.0, 500.0]),
        right=np.array([0.0, 0.0, 700.0]),
        ego_lane_mask=mask,
        confidence=1.0,
    )
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


def straight_lane(top: int = 400, bottom: int = 570, fits: bool = True) -> LaneResult:
    """An ego lane whose lines meet at the vanishing point (400, 300): 600 px wide at the image bottom.
    The mask covers rows top..bottom only, as when an obstacle ahead stops the lane scan."""
    left, right = np.array([0.0, -1.0, 700.0]), np.array([0.0, 1.0, 100.0])  # x = 700 - y, x = y + 100
    ys, xs = np.mgrid[0:600, 0:800]
    mask = (ys >= top) & (ys <= bottom) & (xs >= 700 - ys) & (xs < ys + 100)
    return LaneResult(
        left=left if fits else None, right=right if fits else None, ego_lane_mask=mask, confidence=1.0
    )


def test_lane_span_extrapolates_the_lines_above_the_mask():
    lanes = straight_lane(top=400)
    assert lane_span(lanes, 450) == (250.0, 550.0)
    assert lane_span(lanes, 350) == (350.0, 450.0)  # above the mask: the lines, extrapolated
    assert lane_span(lanes, 590) == (130.0, 670.0)  # below the mask: its bottom row
    assert lane_span(lanes, 290) is None  # beyond the vanishing point
    assert lane_span(straight_lane(fits=False), 450) is None  # a mask alone may be the whole road


def test_obstacle_ahead_is_in_lane_although_it_hides_the_lane():
    # A car straight ahead stands on row 392; the lane scan stopped at row 400, below it (seen in CARLA)
    assert in_ego_lane([380, 360, 420, 392], None, straight_lane(top=400), 1.8)


def test_without_lane_lines_the_corridor_decides():
    road = straight_lane(fits=False)  # e.g. the drivable-area fallback: the mask may span several lanes
    assert in_ego_lane([100, 300, 200, 450], 0.5, road, 1.8)
    assert not in_ego_lane([380, 300, 420, 450], 4.0, road, 1.8)


def test_in_lane_needs_half_the_object_or_a_quarter_of_the_lane():
    lanes = straight_lane()  # row 500: lane from 200 to 600, 400 px wide
    assert not in_ego_lane([560, 300, 800, 500], None, lanes, 1.8)  # parked over the line: 40 of 240 px in
    assert in_ego_lane([480, 300, 800, 500], None, lanes, 1.8)  # cutting in: 120 px = 30 % of the lane
    assert in_ego_lane([570, 400, 630, 500], None, lanes, 1.8)  # pedestrian stepping in: half of them in
    assert not in_ego_lane([585, 400, 630, 500], None, lanes, 1.8)  # a third in: not yet
