"""Lane ground truth from BDD100K polygons (data/bdd100k_polygons.py): ego-lane and lane-marking masks.

ego lane      the `area/drivable` polygon: the area the car can drive in directly, i.e. its own lane
              (BDD100K labels neighbouring lanes separately as `area/alternative`)
lane markings painted lines running along the road (direction "parallel"): crosswalks and stop lines
              (direction "vertical") and road curbs (not paint) are left out. BDD100K draws both edges
              of each painted line, so drawing every polyline a few pixels thick covers the line.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from perception.io.bdd100k import IMAGE_HEIGHT, IMAGE_WIDTH

MARKING_EXCLUDED = ("road curb",)


def sample_poly2d(vertices, types: str, closed: bool, steps: int = 16) -> np.ndarray:
    """Dense (M, 2) points along a BDD100K poly2d: straight segments between "L" points, cubic Beziers
    where two "C" control points sit between them. A closed polygon also joins its last point to its first.
    """
    points = np.asarray(vertices, np.float64).reshape(-1, 2)
    if closed:
        points, types = np.vstack([points, points[:1]]), types + types[0]
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        if types[i + 1] == "C" and i + 3 < len(points):
            p0, c1, c2, p3 = points[i : i + 4]
            t = np.linspace(0, 1, steps + 1)[1:, None]
            out.extend((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * p3)
            i += 3
        else:
            out.append(points[i + 1])
            i += 1
    return np.array(out)


def _scaled(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    scale = np.array([shape[1] / IMAGE_WIDTH, shape[0] / IMAGE_HEIGHT])
    return np.round(points * scale).astype(np.int32)


def ego_lane_mask(polygons: pd.DataFrame, shape: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)) -> np.ndarray:
    """(H, W) bool mask of one image's `area/drivable` polygons, drawn at `shape` (rows, cols)."""
    mask = np.zeros(shape, np.uint8)
    drivable = polygons[(polygons.kind == "area") & (polygons.category == "drivable")]
    for row in drivable.itertuples(index=False):
        cv2.fillPoly(mask, [_scaled(sample_poly2d(row.vertices, row.types, closed=True), shape)], 1)
    return mask.astype(bool)


def marking_mask(
    polygons: pd.DataFrame, shape: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH), thickness: int = 8
) -> np.ndarray:
    """(H, W) bool mask of one image's lane markings, each polyline `thickness` px wide at 1280x720."""
    mask = np.zeros(shape, np.uint8)
    lanes = polygons[
        (polygons.kind == "lane")
        & (polygons.direction == "parallel")
        & ~polygons.category.isin(MARKING_EXCLUDED)
    ]
    width = max(1, round(thickness * shape[1] / IMAGE_WIDTH))
    for row in lanes.itertuples(index=False):
        points = _scaled(sample_poly2d(row.vertices, row.types, closed=False), shape)
        cv2.polylines(mask, [points], isClosed=False, color=1, thickness=width)
    return mask.astype(bool)
