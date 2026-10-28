"""Fit the classical lane detector's bird's-eye-view geometry to BDD100K from its lane labels.

    python scripts/calibrate_lanes.py                  # writes configs/lanes/classical_bdd100k.yaml

Dashcams differ from car to car, so this finds the dataset's typical geometry (medians over val images):
- vanishing point: where straight lane markings on the left and right of the image meet
- bottom row: above the hood for 90% of cars (10th percentile of where the drivable-area labels end;
  hoods differ a lot between cars, so the median would put half of them in view)
- lane width at that row, from the drivable area (the ego lane)
- extend_to: the ego lane is drawn down to the 30th percentile of where drivable areas end (mostly above
  the hood), extending the searched region towards the car
In CARLA the camera is known, so its geometry comes from the camera model instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from perception.io.bdd100k import IMAGE_HEIGHT, IMAGE_WIDTH
from perception.io.bdd100k_lanes import MARKING_EXCLUDED, ego_lane_mask, sample_poly2d

MIN_LINE_PX = 150  # only long, clearly sloped markings define the vanishing point


def marking_lines(polygons: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """(point, direction) of each long, straight-ish lane marking in one image."""
    lines = []
    lanes = polygons[
        (polygons.kind == "lane")
        & (polygons.direction == "parallel")
        & ~polygons.category.isin(MARKING_EXCLUDED)
    ]
    for row in lanes.itertuples(index=False):
        points = sample_poly2d(row.vertices, row.types, closed=False)
        start, end = points[0], points[-1]
        d = end - start
        if np.linalg.norm(d) < MIN_LINE_PX or abs(d[1]) < 0.25 * np.linalg.norm(d):
            continue
        # Straight enough: every point within 5 px of the start-end chord
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if np.abs((points - start) @ normal).max() < 5:
            lines.append((start, d / np.linalg.norm(d)))
    return lines


def intersect(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    (p, u), (q, v) = a, b
    det = u[0] * v[1] - u[1] * v[0]
    if abs(det) < 1e-3:
        return None
    t = ((q - p)[0] * v[1] - (q - p)[1] * v[0]) / det
    return p + t * u


def image_vanishing_point(lines) -> np.ndarray | None:
    """Median crossing point of the image's left-side markings with its right-side ones."""
    # A left-side marking runs up and to the right towards the vanishing point, so dx/dy < 0
    slopes = [u[0] / u[1] for _, u in lines]
    left = [line for line, s in zip(lines, slopes, strict=True) if s < -0.2]
    right = [line for line, s in zip(lines, slopes, strict=True) if s > 0.2]
    points = [x for a in left for b in right if (x := intersect(a, b)) is not None]
    points = [
        x for x in points if 0 < x[0] < IMAGE_WIDTH and 0.25 * IMAGE_HEIGHT < x[1] < 0.75 * IMAGE_HEIGHT
    ]
    return np.median(points, axis=0) if points else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/bdd100k"))
    parser.add_argument("--images", type=int, default=3000, help="val images sampled")
    parser.add_argument("--out", type=Path, default=Path("configs/lanes/classical_bdd100k.yaml"))
    args = parser.parse_args()

    polygons = pd.read_parquet(args.root / "index" / "polygons.parquet")
    by_image = dict(tuple(polygons.groupby("image_id", observed=True)))
    rng = np.random.default_rng(0)
    ids = rng.choice(sorted(by_image), size=min(args.images, len(by_image)), replace=False)

    vps, bottoms = [], []
    for image_id in ids:
        rows = by_image[image_id]
        vp = image_vanishing_point(marking_lines(rows))
        if vp is not None:
            vps.append(vp)
        mask = ego_lane_mask(rows)
        if mask.any():
            bottoms.append(np.nonzero(mask.any(axis=1))[0].max())
    vx, vy = np.median(vps, axis=0)
    bottom_y = float(np.percentile(bottoms, 10))
    extend_y = float(np.percentile(bottoms, 30))

    widths = []
    row = int(round(bottom_y))
    for image_id in ids:
        mask = ego_lane_mask(by_image[image_id])
        cols = np.nonzero(mask[row])[0]
        if len(cols) and cols.max() - cols.min() + 1 == len(cols):  # one contiguous span
            widths.append(len(cols))
    lane_width = float(np.median(widths))

    geometry = {
        "vanishing_point": [round(float(vx) / IMAGE_WIDTH, 4), round(float(vy) / IMAGE_HEIGHT, 4)],
        "bottom": round(bottom_y / IMAGE_HEIGHT, 4),
        "lane_width": round(lane_width / IMAGE_WIDTH, 4),
        "extend_to": round(extend_y / IMAGE_HEIGHT, 4),
    }
    print(
        f"vanishing point from {len(vps)} images: ({vx:.0f}, {vy:.0f}) px; bottom row {bottom_y:.0f} px "
        f"from {len(bottoms)}; lane width {lane_width:.0f} px from {len(widths)}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Classical lane detector on BDD100K (1280x720 dashcams). Geometry as fractions of image\n"
        f"# width and height, fitted by scripts/calibrate_lanes.py on {len(ids)} val images.\n"
    )
    args.out.write_text(header + yaml.safe_dump({"type": "classical", **geometry}, sort_keys=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
