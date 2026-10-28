import numpy as np

from perception.core.matching import assign, iou_matrix


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)

    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_assign_respects_max_cost():
    cost = np.array([[0.1, 0.9], [0.8, 0.95]])

    matches, rows, cols = assign(cost, max_cost=0.5)

    assert matches == [(0, 0)] and rows == [1] and cols == [1]
    assert assign(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])
