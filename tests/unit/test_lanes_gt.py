import json

import numpy as np
import pandas as pd
import pytest

from perception.eval.lanes import LaneEvaluator
from perception.io.bdd100k import parse_polygons
from perception.io.bdd100k_lanes import ego_lane_mask, marking_mask, sample_poly2d


def polygons(*rows) -> pd.DataFrame:
    """rows: (kind, category, direction, vertices, types)."""
    return pd.DataFrame(
        [{"kind": k, "category": c, "direction": d, "vertices": v, "types": t} for k, c, d, v, t in rows]
    )


def test_parse_polygons(tmp_path):
    label = {
        "name": "img1",
        "frames": [
            {
                "objects": [
                    {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
                    {
                        "category": "lane/single white",
                        "attributes": {"direction": "parallel", "style": "dashed"},
                        "poly2d": [[10, 700, "L"], [600, 400, "L"]],
                    },
                    {"category": "area/drivable", "attributes": {}, "poly2d": [[0, 0, "L"], [5, 0, "C"]]},
                ]
            }
        ],
    }
    path = tmp_path / "img1.json"
    path.write_text(json.dumps(label))

    lane, area = parse_polygons(path)

    assert (lane["kind"], lane["category"], lane["style"]) == ("lane", "single white", "dashed")
    assert lane["vertices"] == [10.0, 700.0, 600.0, 400.0] and lane["types"] == "LL"
    assert (area["kind"], area["category"], area["direction"]) == ("area", "drivable", "none")


def test_sample_straight_and_bezier():
    np.testing.assert_allclose(sample_poly2d([0, 0, 10, 0], "LL", closed=False), [[0, 0], [10, 0]])

    # Cubic Bezier from (0, 0) to (30, 0) with both controls at y=30: peaks at y = 30 * 3/4 halfway
    curve = sample_poly2d([0, 0, 0, 30, 30, 30, 30, 0], "LCCL", closed=False, steps=4)
    assert len(curve) == 5
    np.testing.assert_allclose(curve[[0, 2, 4]], [[0, 0], [15, 22.5], [30, 0]])


def test_closed_polygon_joins_back_to_the_start():
    square = sample_poly2d([0, 0, 10, 0, 10, 10, 0, 10], "LLLL", closed=True)

    np.testing.assert_allclose(square[[0, -1]], [[0, 0], [0, 0]])


def test_ego_lane_mask_uses_only_the_drivable_area():
    rows = polygons(
        ("area", "drivable", "none", [100, 700, 300, 700, 300, 500, 100, 500], "LLLL"),
        ("area", "alternative", "none", [800, 700, 1000, 700, 1000, 500, 800, 500], "LLLL"),
    )

    mask = ego_lane_mask(rows)
    half = ego_lane_mask(rows, shape=(360, 640))

    assert mask.shape == (720, 1280) and mask[600, 200] and not mask[600, 900]
    assert half[300, 100] and abs(int(half.sum()) - int(mask.sum()) // 4) < 400


def test_marking_mask_keeps_only_lane_lines():
    rows = polygons(
        ("lane", "single white", "parallel", [600, 719, 600, 400], "LL"),
        ("lane", "crosswalk", "vertical", [100, 600, 400, 600], "LL"),
        ("lane", "road curb", "parallel", [1000, 719, 1100, 400], "LL"),
    )

    mask = marking_mask(rows, thickness=8)

    assert mask[500, 600] and mask[500, 597] and not mask[500, 610]
    assert not mask[600, 250] and not mask[550, 1050]


def test_ego_lane_scores():
    gt = np.zeros((10, 10), bool)
    gt[5:, :] = True  # 50 px
    pred = np.zeros((10, 10), bool)
    pred[5:, :5] = True  # 25 px, all inside
    ev = LaneEvaluator()

    ev.update_ego(pred, gt)
    ev.update_ego(pred, np.zeros((10, 10), bool))  # no ground truth: skipped
    m = ev.summary()

    assert m["ego_iou"] == pytest.approx(0.5) and m["ego_images"] == 1
    assert m["ego_precision"] == 1.0 and m["ego_recall"] == pytest.approx(0.5)


def test_marking_scores_allow_a_small_offset():
    gt = np.zeros((50, 50), bool)
    gt[:, 20] = True
    near, far = np.zeros_like(gt), np.zeros_like(gt)
    near[:, 23] = True
    far[:, 40] = True

    close = LaneEvaluator(tolerance_px=5)
    close.update_markings(near, gt)
    off = LaneEvaluator(tolerance_px=5)
    off.update_markings(far, gt)

    assert close.summary()["marking_f1"] == 1.0
    assert off.summary()["marking_f1"] == 0.0
