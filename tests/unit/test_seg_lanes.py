import cv2
import numpy as np
import pytest
from test_classical_lanes import WHITE, H, W, road

from perception.core import registry
from perception.core.registry import register
from perception.lanes.segmentation import SegLaneDetector, letterbox, run_centres


@register("backend", "fake_seg")
class PaintBackend:
    """Stands in for the network: bright pixels are lane line; the left road_end of the input is drivable."""

    road_end = 1.0

    def __init__(self, model, inputs, outputs, device=None, half=False) -> None:
        pass

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = inputs["images"]
        n, _, h, w = x.shape
        lane = np.where(x.mean(axis=1) > 0.8, 10.0, -10.0)
        drivable = np.full((n, h, w), -10.0)
        drivable[..., : int(self.road_end * w)] = 10.0
        stack = lambda pos: np.stack([-pos, pos], axis=1).astype(np.float32)  # noqa: E731
        return {"drivable": stack(drivable), "lanes": stack(lane)}


def build(geometry: str, **kwargs) -> SegLaneDetector:
    PaintBackend.road_end = 1.0
    return registry.build(
        "lanes",
        {"type": "segmentation", "model": "unused", "backend": "fake_seg", "geometry": geometry,
         "vanishing_point": (0.5, 0.5), "bottom": 0.9, "lane_width": 0.4, **kwargs},
    )  # fmt: skip


@pytest.fixture(params=["rows", "bev"])
def detector(request):
    return build(request.param)


@pytest.mark.parametrize(
    ("shape", "placed"),
    [((720, 1280), (12, 0, 360, 640)), ((600, 800), (0, 64, 384, 512)), ((384, 640), (0, 0, 384, 640))],
)
def test_letterbox_keeps_aspect_and_centres(shape, placed):
    canvas, box = letterbox(np.zeros((*shape, 3), np.uint8), 640, 384)

    assert canvas.shape == (384, 640, 3)
    assert box == placed
    top, left, h, w = box
    assert (canvas[top : top + h, left : left + w] == 0).all()
    assert canvas.sum() == 114 * 3 * (384 * 640 - h * w)


def test_segment_returns_probabilities_at_image_size(detector):
    drivable, lanes = detector.segment(road().image)

    assert drivable.shape == lanes.shape == (H, W)
    assert drivable[150:].min() > 0.99  # rows above 150 are cropped to the 16:9 training aspect
    assert lanes[H - 5, 200] > 0.5 and lanes[H - 5, 400] < 0.5  # on the left line / mid-lane


def test_finds_the_lane_from_the_network_output(detector):
    result = detector.detect(road())

    assert result.confidence == 1.0
    assert np.polyval(result.left, 500) == pytest.approx(400 - 160 * 5 / 6, abs=6)
    assert np.polyval(result.right, 500) == pytest.approx(400 + 160 * 5 / 6, abs=6)
    assert result.ego_lane_mask[500, 400] and not result.ego_lane_mask[500, 150]


def test_ego_lane_is_clipped_to_the_drivable_area(detector):
    PaintBackend.road_end = 0.55  # drivable up to input x 352 = image x 440
    result = detector.detect(road())

    assert result.ego_lane_mask[500, 300] and not result.ego_lane_mask[500, 470]


def test_no_lane_lines_gives_an_empty_result():
    for detector in (build("bev"), build("rows", fallback="none")):
        result = detector.detect(road(lines=()))

        assert result.ego_lane_mask is None and result.confidence == 0.0


@pytest.mark.parametrize("fallback", ["corridor", "drivable"])
def test_rows_falls_back_to_the_road_without_lane_lines(fallback):
    result = build("rows", fallback=fallback).detect(road(lines=()))

    assert result.confidence == 0.25 and result.left is None
    assert result.ego_lane_mask[550, 400]
    # corridor: one expected lane width (0.4 * 800 * 250 / 240 = 333 px at y = 550); drivable: the whole road
    assert result.ego_lane_mask[550, 100] == (fallback == "drivable")


def test_run_centres_skips_wide_runs():
    cols = np.r_[10:16, 40:200, 300:304]  # a line, a stop line, a line

    assert run_centres(cols, max_width=50).tolist() == [12.5, 301.5]


def test_rows_ignores_a_stop_line_across_the_lane():
    frame = road()
    cv2.line(frame.image, (180, 470), (620, 470), WHITE, 10)

    result = build("rows").detect(frame)

    assert result.confidence == 1.0
    assert np.polyval(result.left, 520) == pytest.approx(400 - 160 * 220 / 240, abs=8)


def test_rows_follows_a_curve():
    image = road(lines=()).image
    ys = np.arange(330, H)
    for x_bottom in (200, 600):
        xs = 400 + (x_bottom - 400) * (ys - 300) / 300 + 1e-4 * (ys - H) ** 2  # bends right further ahead
        cv2.polylines(image, [np.c_[xs, ys].astype(np.int32)], False, WHITE, 6)
    frame = road(lines=())
    frame.image[:] = image

    result = build("rows").detect(frame)

    y = 360
    expected_left = 400 - 200 * (y - 300) / 300 + 1e-4 * (y - H) ** 2
    assert result.confidence == 1.0
    assert np.polyval(result.left, y) == pytest.approx(expected_left, abs=8)


def test_rows_with_one_line_cuts_the_road_region_at_it():
    PaintBackend.road_end = 1.0
    detector = build("rows")
    PaintBackend.road_end = 0.8  # road ends at input x 512 = image x 640 (800x450 crop scaled by 0.8)
    result = detector.detect(road(lines=((200, WHITE),)))

    assert result.confidence == 0.5
    assert result.ego_lane_mask[500, 400] and result.ego_lane_mask[500, 620]  # up to the road edge
    assert not result.ego_lane_mask[500, 230] and not result.ego_lane_mask[500, 660]  # past the line / road


def test_rows_places_a_missing_line_one_lane_width_away():
    result = build("rows", one_line="width").detect(road(lines=((200, WHITE),)))

    assert result.confidence == 0.5
    # expected width at y = 500: 0.4 * 800 * (500 - 300) / (540 - 300)
    assert np.polyval(result.right, 500) - np.polyval(result.left, 500) == pytest.approx(
        320 * 200 / 240, abs=8
    )


def test_geometry_without_a_model_runs_on_given_masks():
    detector = registry.build(
        "lanes",
        {
            "type": "segmentation",
            "model": None,
            "vanishing_point": (0.5, 0.5),
            "bottom": 0.9,
            "lane_width": 0.4,
        },
    )
    frame = road()
    paint = frame.image.mean(axis=2) > 200

    result = detector.lanes_from_masks(paint, np.ones_like(paint))

    assert result.confidence == 1.0 and result.ego_lane_mask[500, 400]
    with pytest.raises(RuntimeError):
        detector.detect(frame)


def test_taller_images_are_cropped_from_the_top_to_the_training_aspect():
    class Recorder(PaintBackend):
        seen = []

        def run(self, inputs):
            Recorder.seen.append(inputs["images"].copy())
            return super().run(inputs)

    register("backend", "recorder")(Recorder)
    detector = registry.build("lanes", {"type": "segmentation", "model": "unused", "backend": "recorder"})
    image = np.zeros((600, 800, 3), np.uint8)
    image[:150] = 255  # the rows a 16:9 crop drops (800 / (16 / 9) = 450 kept)

    drivable, lanes = detector.segment(image)

    x = Recorder.seen[-1][0]
    assert x.shape == (3, 384, 640)
    assert np.allclose(x[:, 12:372], 0) and np.allclose(x[:, :12], 114 / 255)  # training layout, no white
    assert (lanes[:150] == 0).all() and (drivable[:150] == 0).all() and drivable[150:].min() > 0.99


def test_one_line_region_can_be_limited_to_a_number_of_lane_widths():
    result = build("rows", one_line_max_lanes=1.0).detect(road(lines=((200, WHITE),)))

    # at y = 500 the line is at x 267 and one lane width is 0.4 * 800 * 200 / 240 = 267 px
    assert result.ego_lane_mask[500, 400] and not result.ego_lane_mask[500, 560]
