import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import Tracker
from perception.core.types import Detection
from perception.tracking.deepsort import (
    DeepSort,
    DeepSortTracker,
    xyah_to_xyxy,
    xyxy_to_xyah,
)

CLASS_IDS = {"car": 0, "person": 3}
DIM = 8


def det(x: float, y: float, score: float = 0.9, name: str = "car", size: float = 50.0) -> Detection:
    box = np.array([x, y, x + size, y + size], np.float32)
    return Detection(box=box, score=score, class_id=CLASS_IDS[name], class_name=name)


def look(i: int) -> np.ndarray:
    """A unit appearance vector; different i are maximally different (cosine distance 1)."""
    return np.eye(DIM, dtype=np.float32)[i]


def step(tracker: DeepSort, objects: list[tuple[Detection, int]]):
    dets = [d for d, _ in objects]
    features = np.stack([look(i) for _, i in objects]) if objects else np.zeros((0, DIM), np.float32)
    return tracker.update(dets, features)


@pytest.fixture
def tracker():
    return DeepSort()


def test_box_conversions_round_trip():
    box = np.array([10.0, 20.0, 50.0, 100.0], np.float32)

    np.testing.assert_allclose(xyxy_to_xyah(box), [30, 60, 0.5, 80])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box)


def test_track_is_reported_once_confirmed_and_keeps_its_id(tracker):
    reported = [step(tracker, [(det(100 + 5 * i, 100), 0)]) for i in range(10)]

    assert reported[0] == [] and reported[1] == []  # tentative until n_init=3 hits
    assert {t.track_id for frame in reported[2:] for t in frame} == {1}
    assert reported[-1][0].age == 9


def test_tentative_track_is_deleted_on_its_first_miss(tracker):
    step(tracker, [(det(100, 100), 0)])
    step(tracker, [])
    for _ in range(3):
        tracks = step(tracker, [(det(100, 100), 0)])

    assert [t.track_id for t in tracks] == [2]


def test_appearance_recovers_the_right_track_after_occlusion(tracker):
    for i in range(5):
        step(tracker, [(det(100 + 5 * i, 100), 0)])
    for _ in range(5):  # occluded: longer than the IoU stage covers, so only appearance can re-match
        step(tracker, [])

    # The car comes back where the Kalman filter expects it, next to a new, different-looking object
    tracks = step(tracker, [(det(178, 100), 1), (det(150, 100), 0)])

    assert [t.track_id for t in tracks] == [1]
    assert tracks[0].box[0] == pytest.approx(150, abs=10)


def test_confirmed_track_is_dropped_after_track_buffer_misses():
    tracker = DeepSort(track_buffer=5)
    for _ in range(3):
        step(tracker, [(det(100, 100), 0)])
    for _ in range(6):
        step(tracker, [])

    assert tracker.tracks == []


def test_reset_restarts_ids(tracker):
    for _ in range(3):
        step(tracker, [(det(100, 100), 0), (det(400, 100), 1)])

    tracker.reset()
    for _ in range(3):
        tracks = step(tracker, [(det(300, 300), 2)])

    assert [t.track_id for t in tracks] == [1]


class SliceBackend:
    """Fake embedding model: the first DIM pixel values of each crop as its features."""

    def __init__(self, model, device=None):
        pass

    def run(self, inputs):
        images = inputs["images"]
        return {"features": images.reshape(len(images), -1)[:, :DIM]}


def test_tracker_ignores_low_score_detections(monkeypatch, make_frame):
    monkeypatch.setitem(registry._registry["backend"], "slice", SliceBackend)
    tracker = DeepSortTracker(reid_model="unused", reid_backend="slice", min_score=0.25)

    for i in range(3):
        tracks = tracker.update([det(100, 100), det(400, 100, score=0.15)], make_frame(i))

    assert isinstance(tracker, Tracker)
    assert [t.track_id for t in tracks] == [1]


def test_registered_as_deepsort():
    assert registry.get("tracker", "deepsort") is DeepSortTracker
