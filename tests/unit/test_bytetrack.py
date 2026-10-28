import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import Tracker
from perception.core.types import Detection
from perception.tracking.bytetrack import ByteTrackTracker, TrackerInput

CLASS_IDS = {"car": 0, "person": 3}


def det(x: float, y: float, score: float = 0.9, name: str = "car", size: float = 50.0) -> Detection:
    box = np.array([x, y, x + size, y + size], np.float32)
    return Detection(box=box, score=score, class_id=CLASS_IDS[name], class_name=name)


@pytest.fixture
def tracker():
    return ByteTrackTracker()


def test_input_converts_xyxy_to_centre_size():
    boxes = TrackerInput.from_detections([det(10, 20, size=40)])

    np.testing.assert_allclose(boxes.xywh, [[30, 40, 40, 40]])
    assert len(boxes[np.array([False])]) == 0
    assert len(TrackerInput.from_detections([])) == 0


def test_moving_object_keeps_its_id(tracker, make_frame):
    ids, ages = [], []
    for i in range(10):
        (track,) = tracker.update([det(100 + 5 * i, 100)], make_frame(i))
        ids.append(track.track_id)
        ages.append(track.age)

    assert set(ids) == {1}
    assert ages == list(range(10))
    assert isinstance(tracker, Tracker)


def test_separate_objects_get_separate_ids_and_keep_their_class(tracker, make_frame):
    for i in range(3):
        tracks = tracker.update([det(100 + 5 * i, 100), det(500, 300, name="person")], make_frame(i))

    assert len({t.track_id for t in tracks}) == 2
    assert {t.class_name for t in tracks} == {"car", "person"}


def test_low_score_detection_keeps_a_track_alive_but_does_not_start_one(tracker, make_frame):
    for i in range(3):
        tracker.update([det(100 + 5 * i, 100)], make_frame(i))

    # The car's score drops (e.g. partly occluded); a weak detection elsewhere appears
    tracks = tracker.update([det(115, 100, score=0.15), det(500, 300, score=0.15)], make_frame(3))

    assert [t.track_id for t in tracks] == [1]


def test_track_comes_back_with_the_same_id_after_a_short_gap(tracker, make_frame):
    for i in range(5):
        tracker.update([det(100 + 5 * i, 100)], make_frame(i))
    for i in range(5, 10):
        assert tracker.update([], make_frame(i)) == []

    (track,) = tracker.update([det(150, 100)], make_frame(10))

    assert track.track_id == 1


def test_reset_restarts_ids(tracker, make_frame):
    tracker.update([det(100, 100)], make_frame(0))
    tracker.update([det(500, 300)], make_frame(1))
    tracker.update([det(500, 300)], make_frame(2))

    tracker.reset()
    (track,) = tracker.update([det(300, 300)], make_frame(0))

    assert track.track_id == 1 and track.age == 0


def test_registered_as_bytetrack():
    assert registry.get("tracker", "bytetrack") is ByteTrackTracker
