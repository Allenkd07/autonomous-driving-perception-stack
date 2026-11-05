import numpy as np
import pandas as pd
import pytest

from perception.eval.distance import dense_abs_rel, distance_bin, distance_errors, distance_table


def test_distance_errors():
    m = distance_errors(np.array([10.0, 20.0, 40.0, 5.0]), np.array([11.0, 20.0, 30.0, np.nan]))

    assert m["n"] == 4 and m["coverage"] == pytest.approx(0.75)
    assert m["abs_rel"] == pytest.approx((0.1 + 0.0 + 0.25) / 3)
    assert m["mae_m"] == pytest.approx(1.0)
    assert m["rmse_m"] == pytest.approx(np.sqrt((1 + 0 + 100) / 3))
    assert m["delta1"] == pytest.approx(2 / 3)  # 40 / 30 = 1.33 is outside 1.25


def test_no_estimates_give_zero_coverage():
    m = distance_errors(np.array([10.0]), np.array([np.nan]))
    assert m["coverage"] == 0.0 and np.isnan(m["abs_rel"])


def test_distance_table_by_group_and_bins():
    objects = pd.DataFrame({"truth": [5.0, 15.0, 50.0], "a": [5.0, 15.0, 40.0], "g": ["x", "x", "y"]})
    objects["bin"] = distance_bin(objects["truth"].to_numpy())

    by_bin = distance_table(objects, ["a"], by="bin")
    by_group = distance_table(objects, ["a"], by="g")

    assert list(by_bin["bin"]) == ["0-10 m", "10-20 m", "40+ m"]  # empty bins are left out
    assert by_group.set_index("g").loc["y", "abs_rel"] == pytest.approx(0.2)


def test_dense_abs_rel_skips_invalid_and_infinite_pixels():
    truth = np.array([[10.0, 20.0], [1000.0, 5.0]])
    pred = np.array([[11.0, np.inf], [3.0, 5.0]])
    valid = truth < 80

    assert dense_abs_rel(pred, truth, valid) == pytest.approx((0.1 + 0.0) / 2)
