from pathlib import Path

import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import Detector
from perception.core.types import CameraInfo, Frame
from perception.detection.yolo import YoloDetector, class_ids, to_detections

NAMES = {0: "car", 1: "truck", 2: "person"}
WEIGHTS = Path("weights/yolo11s_bdd100k.pt")
VAL_IMAGES = Path("data/bdd100k/images/100k/val")


def test_class_ids_by_name():
    assert class_ids(NAMES, ["person", "car"]) == [2, 0]
    assert class_ids(NAMES, None) is None


def test_unknown_class_name_is_an_error():
    with pytest.raises(ValueError, match="bus"):
        class_ids(NAMES, ["car", "bus"])


def test_to_detections():
    xyxy = np.array([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 5.0, 5.0]])

    dets = to_detections(xyxy, np.array([0.9, 0.4]), np.array([2.0, 0.0]), NAMES)

    assert [(d.class_id, d.class_name) for d in dets] == [(2, "person"), (0, "car")]
    assert dets[0].score == pytest.approx(0.9)
    np.testing.assert_array_equal(dets[0].box, xyxy[0])
    assert to_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), NAMES) == []


def test_registered_as_yolo():
    assert registry.get("detector", "yolo") is YoloDetector


@pytest.mark.skipif(
    not (WEIGHTS.exists() and VAL_IMAGES.exists()), reason="needs fine-tuned weights and BDD100K"
)
def test_detects_on_a_bdd100k_image():
    import cv2

    image = cv2.imread(str(sorted(VAL_IMAGES.glob("*.jpg"))[0]))
    h, w = image.shape[:2]
    camera = CameraInfo.from_fov(width=w, height=h, fov_deg=90.0, height_m=1.6)
    frame = Frame(image=image, timestamp=0.0, frame_id=0, camera=camera)
    detector = YoloDetector(WEIGHTS, conf=0.3, classes=["car", "truck", "bus", "person"], device="cpu")

    dets = detector.detect(frame)

    assert isinstance(detector, Detector)
    assert dets, "expected at least one detection on a street scene"
    assert {d.class_name for d in dets} <= {"car", "truck", "bus", "person"}
    for d in dets:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and d.score >= 0.3
