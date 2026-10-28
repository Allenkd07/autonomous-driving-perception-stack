import cv2
import numpy as np
import pytest

from perception.core import registry
from perception.core.types import CameraInfo, Frame
from perception.lanes.classical import ClassicalLaneDetector

W, H = 800, 600
VP = (400, 300)  # the detector's default vanishing point (0.5, 0.5)
WHITE, YELLOW = (230, 230, 230), (0, 200, 220)


def road(road_gray=90, lines=((200, WHITE), (600, WHITE)), thickness=6) -> Frame:
    """Grey road with lane lines running from (x, bottom of image) to the vanishing point."""
    image = np.full((H, W, 3), road_gray, np.uint8)
    for x_bottom, colour in lines:
        top = (int(VP[0] + (x_bottom - VP[0]) * 0.1), int(VP[1] + (H - VP[1]) * 0.1))
        cv2.line(image, (x_bottom, H), top, colour, thickness)
    camera = CameraInfo.from_fov(W, H, 90.0, 1.6)
    return Frame(image=image, timestamp=0.0, frame_id=0, camera=camera)


@pytest.fixture
def detector():
    # the drawn lines cross the detector's bottom row (y = 540) at x = 400 -+ 160: lane width 0.4
    return ClassicalLaneDetector(vanishing_point=(0.5, 0.5), bottom=0.9, lane_width=0.4)


def line_x(fit, y):
    return np.polyval(fit, y)


def test_finds_both_lines_and_the_lane_between(detector):
    result = detector.detect(road())

    assert result.confidence == 1.0
    # at y = 500 the drawn lines are at 400 -+ 160 * (500 - 300) / (540 - 300)
    assert line_x(result.left, 500) == pytest.approx(400 - 160 * 5 / 6, abs=6)
    assert line_x(result.right, 500) == pytest.approx(400 + 160 * 5 / 6, abs=6)
    assert result.ego_lane_mask.shape == (H, W)
    assert result.ego_lane_mask[500, 400] and not result.ego_lane_mask[500, 150]


def test_works_on_a_dark_image_and_yellow_paint(detector):
    dark = detector.detect(road(road_gray=15, lines=((200, (60, 60, 60)), (600, (60, 60, 60)))))
    yellow = detector.detect(road(lines=((200, YELLOW), (600, WHITE))))

    assert dark.confidence == 1.0 and yellow.confidence == 1.0


def test_a_missing_line_is_placed_one_lane_width_away(detector):
    result = detector.detect(road(lines=((200, WHITE),)))

    assert result.confidence == 0.5
    assert line_x(result.right, 500) == pytest.approx(400 + 160 * 5 / 6, abs=10)


def test_no_lines_gives_an_empty_result(detector):
    result = detector.detect(road(lines=()))

    assert result.left is None and result.ego_lane_mask is None and result.confidence == 0.0


def test_video_reuses_and_smooths_the_previous_fit(detector):
    detector.detect(road())
    assert detector._prev is not None

    shifted = detector.detect(road(lines=((212, WHITE), (612, WHITE))))  # 8 px right at y = 500
    x = line_x(shifted.left, 500)
    assert 400 - 160 * 5 / 6 + 1 < x < 408 - 160 * 5 / 6 - 1  # between the old and new positions

    detector.reset()
    assert detector._prev is None


def test_registered_as_classical():
    assert registry.get("lanes", "classical") is ClassicalLaneDetector
