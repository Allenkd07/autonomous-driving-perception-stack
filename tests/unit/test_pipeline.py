"""Pipeline orchestration with fake stages (no models, no GPU)."""

import numpy as np

from perception.core.interfaces import DepthEstimator, Detector, DistanceEstimator, LaneDetector, Tracker
from perception.core.types import DepthMap, Detection, LaneResult, ObjectState, Track
from perception.pipeline import PerceptionPipeline


class FakeDetector:
    def detect(self, frame):
        return [
            Detection(box=np.array([100.0, 200.0, 150.0, 300.0]), score=0.9, class_id=0, class_name="car")
        ]


class FakeTracker:
    def __init__(self):
        self.resets = 0

    def update(self, detections, frame):
        return [Track(d.box, d.score, d.class_id, d.class_name, track_id=i) for i, d in enumerate(detections)]

    def reset(self):
        self.resets += 1


class FakeLanes:
    def detect(self, frame):
        return LaneResult(left=None, right=None, ego_lane_mask=None, confidence=0.0)


class CountingDepth:
    def __init__(self):
        self.calls = 0

    def estimate(self, frame):
        self.calls += 1
        return DepthMap(depth=np.ones(frame.image.shape[:2], np.float32), is_metric=False)


class FakeDistance:
    def estimate(self, tracks, depth, lanes, frame):
        return [ObjectState(track=t, distance_m=10.0, ttc_s=None, in_ego_lane=True) for t in tracks]


def make_pipeline(**kwargs):
    return PerceptionPipeline(
        detector=FakeDetector(), tracker=FakeTracker(), distance=FakeDistance(), **kwargs
    )


def test_fakes_satisfy_interfaces():
    assert isinstance(FakeDetector(), Detector)
    assert isinstance(FakeTracker(), Tracker)
    assert isinstance(FakeLanes(), LaneDetector)
    assert isinstance(CountingDepth(), DepthEstimator)
    assert isinstance(FakeDistance(), DistanceEstimator)


def test_process_runs_all_stages_and_times_them(make_frame):
    pipeline = make_pipeline(lanes=FakeLanes(), depth=CountingDepth())

    out = pipeline.process(make_frame(0))

    assert len(out.objects) == 1
    assert out.objects[0].track.track_id == 0
    assert out.lanes is not None and out.depth is not None
    assert set(out.timings_ms) == {"detect", "track", "lanes", "depth", "distance"}


def test_optional_stages_can_be_disabled(make_frame):
    out = make_pipeline().process(make_frame(0))

    assert out.lanes is None and out.depth is None
    assert "lanes" not in out.timings_ms and "depth" not in out.timings_ms


def test_depth_every_n_reuses_last_map(make_frame):
    depth = CountingDepth()
    pipeline = make_pipeline(depth=depth, depth_every_n=3)

    outputs = [pipeline.process(make_frame(i)) for i in range(7)]

    assert depth.calls == 3  # frames 0, 3, 6
    assert all(o.depth is not None for o in outputs)
    assert "depth" not in outputs[1].timings_ms


def test_reset_clears_state_and_resets_stages(make_frame):
    depth = CountingDepth()
    pipeline = make_pipeline(depth=depth, depth_every_n=5)
    pipeline.process(make_frame(0))
    pipeline.process(make_frame(1))

    pipeline.reset()
    pipeline.process(make_frame(2))

    assert pipeline.tracker.resets == 1
    assert depth.calls == 2  # reset restarts the every-N counter
