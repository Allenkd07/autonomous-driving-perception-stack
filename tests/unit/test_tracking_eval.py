import numpy as np
import pytest

from perception.eval.tracking import TrackingEvaluator, drop_ignored, summarize

A = np.array([[0, 0, 50, 50]], np.float32)
B = np.array([[200, 0, 250, 50]], np.float32)


def run(frames) -> dict[str, float]:
    """frames: (gt_ids, gt_boxes, pred_ids, pred_boxes[, ignore_boxes]) per frame."""
    ev = TrackingEvaluator()
    for i, frame in enumerate(frames):
        ev.update(i, *frame)
    return ev.summary()


def test_perfect_tracking():
    m = run([([1, 2], np.r_[A, B], [7, 8], np.r_[A, B])] * 5)

    assert m["mota"] == 1.0 and m["idf1"] == 1.0
    assert m["num_switches"] == 0 and m["num_frames"] == 5


def test_id_switch_is_counted():
    frames = [([1], A, [7], A)] * 3 + [([1], A, [9], A)] * 3

    m = run(frames)

    assert m["num_switches"] == 1
    assert m["idf1"] == pytest.approx(0.5)
    assert m["mota"] == pytest.approx(1 - 1 / 6)


def test_misses_and_false_positives():
    m = run([([1], A, [7], B)])  # wrong place: one miss and one false positive

    assert m["num_misses"] == 1 and m["num_false_positives"] == 1
    assert m["mota"] == pytest.approx(-1.0)


def test_low_iou_does_not_match():
    shifted = A + np.array([30, 0, 30, 0], np.float32)  # IoU 20/80 = 0.25

    assert run([([1], A, [7], shifted)])["num_misses"] == 1


def test_prediction_on_an_ignore_region_is_not_a_false_positive():
    parked = np.array([[400, 0, 450, 50]], np.float32)

    m = run([([1], A, [7, 8], np.r_[A, parked], parked)])

    assert m["num_false_positives"] == 0 and m["mota"] == 1.0


def test_drop_ignored_prefers_the_better_match():
    gt = A
    ignore = A + np.array([20, 0, 20, 0], np.float32)

    # Two predictions: one exactly on the scored object, one exactly on the ignore region
    keep = drop_ignored(np.r_[A, ignore], gt, ignore)

    assert keep.tolist() == [True, False]
    assert drop_ignored(A, gt, np.zeros((0, 4))).tolist() == [True]


def test_summarize_pools_sequences():
    perfect, missed = TrackingEvaluator(), TrackingEvaluator()
    for i in range(3):
        perfect.update(i, [1], A, [7], A)
        missed.update(i, [1], A, [], np.zeros((0, 4)))

    m = summarize([perfect, missed])

    assert m["num_frames"] == 6 and m["num_misses"] == 3
    assert m["mota"] == pytest.approx(0.5)
