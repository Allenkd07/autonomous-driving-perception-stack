import dataclasses

import numpy as np
import pytest
from test_distance import CAMERA, project

from perception.core import registry
from perception.core.interfaces import DepthEstimator, DistanceEstimator
from perception.core.registry import register
from perception.core.types import DepthMap, Frame, LaneResult, Track
from perception.depth.network import GroundTruthDepth, NetworkDepth
from perception.fusion.depth_scale import DepthScaleDistance, fit_inverse_depth, to_metric
from perception.fusion.fused import FusedDistance
from perception.fusion.geometry import ground_point, road_depth

A, B = 3.0, 0.2  # the "unknown" per-frame scale and shift of the fake relative depth


def scene(objects: list[tuple[list[float], float]] = ()) -> np.ndarray:
    """Metric depth of an 800x600 frame: flat road below the horizon, sky (1000 m) above, and boxes
    (xyxy, depth) of constant depth on top."""
    z = np.broadcast_to(road_depth(np.arange(600), CAMERA)[:, None], (600, 800)).copy()
    z[np.isnan(z)] = 1000.0
    for (x1, y1, x2, y2), d in objects:
        z[int(y1) : int(y2), int(x1) : int(x2)] = d
    return z


def relative(z: np.ndarray) -> DepthMap:
    return DepthMap(depth=(A / z + B).astype(np.float32), is_metric=False)


def frame(z: np.ndarray | None = None) -> Frame:
    return Frame(np.zeros((600, 800, 3), np.uint8), 0.0, 0, CAMERA, depth_gt=z)


def track(box, track_id: int = 1) -> Track:
    return Track(box=np.array(box, np.float32), score=0.9, class_id=0, class_name="car", track_id=track_id)


def box_at(z: float, width: float = 80, height: float = 120, u: float = 400) -> list[float]:
    """A box standing on the road z metres ahead: its bottom at the road row of depth z."""
    v = project(0.0, z)[1]
    return [u - width / 2, v - height, u + width / 2, v]


def test_road_depth_agrees_with_ground_point():
    for camera in (CAMERA, dataclasses.replace(CAMERA, pitch_rad=np.radians(4))):
        v = project(0.0, 20.0, camera)[1]
        assert road_depth(np.array([v]), camera)[0] == pytest.approx(
            20.0 * np.cos(camera.pitch_rad), rel=0.01
        )
        assert ground_point(400, v, camera)[1] == pytest.approx(20.0)
    assert np.isnan(road_depth(np.array([300.0, 100.0]), CAMERA)).all()  # horizon and sky


def test_fit_recovers_scale_and_shift_despite_outliers():
    rng = np.random.default_rng(0)
    z = rng.uniform(4, 30, 500)
    inverse = A / z + B + rng.normal(0, 0.002, z.size)
    inverse[:50] = rng.uniform(0, 2, 50)  # 10 % junk: a car among the road pixels

    a, b = fit_inverse_depth(inverse, z)

    assert (a, b) == pytest.approx((A, B), rel=0.02)
    assert fit_inverse_depth(inverse[:10], z[:10]) is None  # too few points
    assert fit_inverse_depth(-inverse[50:], z[50:]) is None  # nearer would be smaller


def test_to_metric_inverts_the_fit_and_maps_the_far_side_to_infinity():
    assert to_metric(np.array([A / 10 + B, B, B - 0.1]), A, B) == pytest.approx([10.0, np.inf, np.inf])


@pytest.mark.parametrize("reference", ["ground", "gt"])
def test_depth_scale_distance_from_relative_depth(reference):
    near, far = box_at(10.0), box_at(40.0, u=200)
    z = scene([(near, 10.0), (far, 40.0)])
    estimator = DepthScaleDistance(reference=reference)

    states = estimator.estimate([track(near), track(far, 2)], relative(z), None, frame(z))

    assert [s.distance_m for s in states] == pytest.approx([10.0, 40.0], rel=0.01)
    assert estimator.scale == pytest.approx((A, B), rel=0.01)
    assert states[0].in_ego_lane and not states[1].in_ego_lane  # 200 px left at 40 m is 20 m to the side


def test_ground_reference_uses_the_ego_lane_and_skips_objects():
    near = box_at(10.0)
    mask = np.zeros((600, 800), bool)
    mask[:, 300:500] = True
    estimator = DepthScaleDistance()

    rows, cols, z = estimator.reference_points(
        frame(), LaneResult(None, None, mask, 1.0), [track(near)], CAMERA
    )

    assert ((cols >= 300) & (cols < 500)).all() and len(cols) >= 50
    inside = (cols >= near[0]) & (cols <= near[2]) & (rows >= near[1]) & (rows <= near[3])
    assert not inside.any()
    assert z.min() >= 4.0 and z.max() <= 30.0


def test_metric_depth_is_used_without_calibration():
    near = box_at(12.0)
    z = scene([(near, 12.0)])

    (state,) = DepthScaleDistance().estimate([track(near)], DepthMap(z, is_metric=True), None, frame())

    assert state.distance_m == pytest.approx(12.0)


def test_no_depth_or_failed_fit_gives_no_distance():
    near = box_at(10.0)
    flat = DepthMap(np.ones((600, 800), np.float32), is_metric=False)  # no structure: a = 0
    for depth in (None, flat):
        (state,) = DepthScaleDistance().estimate([track(near)], depth, None, frame())
        assert state.distance_m is None


def test_fused_weights_each_estimate_by_its_error_model():
    near, far = box_at(10.0), box_at(60.0, u=420)
    z = scene([(near, 30.0), (far, 50.0)])  # depth disagrees with the box bottoms on purpose
    tracks = [track(near), track(far, 2)]

    states = FusedDistance().estimate(tracks, relative(z), None, frame())
    assert states[0].distance_m == pytest.approx(10.0, abs=0.5)  # ground plane 0.4 m sigma vs depth 4.5 m
    assert abs(states[1].distance_m - 60.0) < abs(states[1].distance_m - 50.0)  # 6 m vs 12.5 m

    # a less certain ground plane (4 px, no floor): 22.5 m sigma far away, so the depth's 12.5 m wins there
    loose = FusedDistance(pixel_sigma=4.0, ground_rel_sigma=0.0).estimate(tracks, relative(z), None, frame())
    assert abs(loose[1].distance_m - 50.0) < abs(loose[1].distance_m - 60.0)


def test_fused_uses_depth_when_the_box_bottom_is_cut_off():
    cut = [250, 350, 550, 600]  # bottom at the image edge: the contact point is below the image
    z = scene([(cut, 3.0)])

    (state,) = FusedDistance().estimate([track(cut)], relative(z), None, frame())

    assert state.distance_m == pytest.approx(3.0, rel=0.02)


def test_fused_without_depth_is_the_ground_plane():
    (state,) = FusedDistance().estimate([track(box_at(15.0))], None, None, frame())
    assert state.distance_m == pytest.approx(15.0)


@register("backend", "fake_depth")
class FakeDepthBackend:
    def __init__(self, model, inputs, outputs, device=None, half=False):
        self.seen = None

    def run(self, inputs):
        self.seen = inputs["images"]
        _, _, h, w = self.seen.shape
        return {"depth": np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))[None]}


def test_midas_depth_preprocesses_and_resizes_back():
    midas = NetworkDepth(model="unused", backend="fake_depth", input_size=(256, 192))
    image = np.full((600, 800, 3), 128, np.uint8)

    depth = midas.estimate(Frame(image, 0.0, 0, CAMERA))

    assert midas.backend.seen.shape == (1, 3, 192, 256) and midas.backend.seen.dtype == np.float32
    assert depth.depth.shape == (600, 800) and not depth.is_metric
    assert depth.depth[-1, 0] > depth.depth[0, 0]  # the ramp survives the resize


def test_metric_depth_is_rescaled_to_the_camera_focal_length():
    def run(**kw):
        net = NetworkDepth(model="unused", backend="fake_depth", input_size=(686, 518), metric=True, **kw)
        return net.estimate(Frame(np.zeros((600, 800, 3), np.uint8), 0.0, 0, CAMERA)).depth

    as_predicted, rescaled = run(), run(train_focal_px=1001.5)

    expected = CAMERA.K[0, 0] * 686 / 800 / 1001.5  # this camera's focal length at the input / training's
    assert rescaled[100:] / as_predicted[100:] == pytest.approx(expected)
    assert expected < 1  # a wider lens than training's: the raw output reads too far


def test_ground_truth_depth_and_registration():
    z = scene()
    assert GroundTruthDepth().estimate(frame(z)).is_metric
    with pytest.raises(ValueError):
        GroundTruthDepth().estimate(frame())
    for name in ("midas", "depth_anything", "ground_truth"):
        assert name in registry.available("depth")
    for name in ("ground_plane", "depth_scale", "fused"):
        assert name in registry.available("distance")
    assert isinstance(registry.build("distance", {"type": "fused"}), DistanceEstimator)
    assert isinstance(GroundTruthDepth(), DepthEstimator)
