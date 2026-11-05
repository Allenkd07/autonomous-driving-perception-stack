"""Metric distance from relative depth (MiDaS), calibrated per frame.

MiDaS gives inverse depth up to an unknown scale and shift that change every frame: output = a / z + b.
Each frame, (a, b) is fitted by robust least squares on pixels whose metric depth z is known:

ground  the road in front of the car: on a flat road a pixel's depth follows from its row, the camera's
        height and its pitch (perception.fusion.geometry.road_depth). Uses the ego-lane mask when the lane
        detector gives one (it knows where the road is), else a strip of the road ahead, minus every
        tracked object's box. Needs nothing but the camera, so it works in deployment.
gt      CARLA's ground-truth depth over the whole image: the best any per-frame scale and shift can do, an
        upper bound that isolates MiDaS's own error from the calibration's.

The fit keeps the road's 4-30 m by default: nearer rows show the car's bonnet, farther ones are a few pixels
each and their geometric depth is no better than the ground-plane method's. Objects beyond the road samples
are extrapolated in inverse depth, so their error grows with distance.

An object's distance is a low quantile of its box's depth over the central part of the box: its nearer
surface, which is what braking cares about, rather than background showing through gaps (between a
pedestrian's legs, through windows) or the box's edges, which MiDaS blurs into the background.
"""

from __future__ import annotations

import numpy as np

from perception.core.registry import register
from perception.core.types import CameraInfo, DepthMap, Frame, LaneResult, Track
from perception.fusion.base import Measurement, TrackDistanceEstimator
from perception.fusion.geometry import road_depth

MAD_TO_SIGMA = 1.4826


def fit_inverse_depth(
    inverse: np.ndarray, metric: np.ndarray, k: float = 3.0, iterations: int = 5, min_points: int = 50
) -> tuple[float, float] | None:
    """(a, b) with inverse ~ a / metric + b, by least squares repeated without outliers beyond k robust
    standard deviations (cars or kerbs among road pixels, sky in ground truth). None if there are too few
    points or the map doesn't show the reference's depth change (the fitted change across the samples is not
    clearly above the residual noise, or a <= 0: nearer would be smaller).

    Residuals are relative (divided by 1 / metric): an inverse-depth error e at depth z is a depth error of
    about e * z, so the same error matters z times more far away. Unweighted, the fit is dominated by the
    nearest pixels (largest inverse depth) and far pixels come out badly; weighted, it minimises the
    relative depth error, which is what distances are judged by."""
    x = 1.0 / np.asarray(metric, np.float64)
    y = np.asarray(inverse, np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
    keep = finite
    a = b = None
    noise = 0.0
    for _ in range(iterations):
        if keep.sum() < min_points or np.ptp(x[keep]) == 0:
            return None
        a, b = np.polyfit(x[keep], y[keep], 1, w=1.0 / x[keep])
        residual = np.abs(y - (a * x + b))
        relative = np.divide(residual, x, out=np.full_like(x, np.inf), where=x > 0)
        sigma = MAD_TO_SIGMA * np.median(relative[keep]) + 1e-12
        noise = MAD_TO_SIGMA * np.median(residual[keep])
        new = finite & (relative <= k * sigma)
        if np.array_equal(new, keep):
            break
        keep = new
    if a is None or a * np.ptp(x[keep]) <= k * noise + 1e-12:
        return None
    return float(a), float(b)


def to_metric(inverse: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Inverse depth -> metres for a fit (a, b); inf at or beyond infinity (inverse <= b)."""
    d = np.asarray(inverse, np.float64) - b
    with np.errstate(divide="ignore"):
        return np.where(d > 0, a / np.where(d > 0, d, 1.0), np.inf)


def central_region(values: np.ndarray, box: np.ndarray, shrink: float) -> np.ndarray:
    """The box's pixels within its central `shrink` fraction of width and height (at least one pixel)."""
    h, w = values.shape[:2]
    x1, y1, x2, y2 = (float(c) for c in box)
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    c1 = min(max(int(np.floor(cx - hw)), 0), w - 1)
    r1 = min(max(int(np.floor(cy - hh)), 0), h - 1)
    c2 = min(max(int(np.ceil(cx + hw)), c1 + 1), w)
    r2 = min(max(int(np.ceil(cy + hh)), r1 + 1), h)
    return values[r1:r2, c1:c2]


@register("distance", "depth_scale")
class DepthScaleDistance(TrackDistanceEstimator):
    """DistanceEstimator from a depth map: relative maps are calibrated per frame, metric ones used as is.

    reference          ground (road geometry) or gt (CARLA depth, upper bound); see the module docstring
    min_ref_m, max_ref_m  depth range of the road samples
    ref_half_width_m   without an ego-lane mask: the road strip's half width around the camera
    stride             sample every stride-th pixel in both directions for the fit
    box_shrink         the central fraction of each box (width and height) whose depths are used
    box_quantile       quantile of those depths taken as the distance (0.5 = median; lower = nearer side)
    other parameters: see TrackDistanceEstimator
    """

    def __init__(
        self,
        reference: str = "ground",
        min_ref_m: float = 4.0,
        max_ref_m: float = 30.0,
        ref_half_width_m: float = 3.0,
        stride: int = 4,
        box_shrink: float = 0.5,
        box_quantile: float = 0.3,
        **base,
    ) -> None:
        super().__init__(**base)
        if reference not in ("ground", "gt"):
            raise ValueError(f"reference must be ground or gt, not {reference!r}")
        self.reference = reference
        self.min_ref_m, self.max_ref_m = min_ref_m, max_ref_m
        self.ref_half_width_m = ref_half_width_m
        self.stride = stride
        self.box_shrink, self.box_quantile = box_shrink, box_quantile
        self.scale: tuple[float, float] | None = None  # the last frame's (a, b), for inspection

    def reference_points(
        self, frame: Frame, lanes: LaneResult | None, tracks: list[Track], camera: CameraInfo
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, cols, metric depth) of the pixels to fit the scale on."""
        h, w = frame.image.shape[:2]
        rows = np.arange(self.stride // 2, h, self.stride)
        cols = np.arange(self.stride // 2, w, self.stride)
        if self.reference == "gt":
            if frame.depth_gt is None:
                raise ValueError("reference=gt needs frames with depth_gt (CARLA recordings)")
            z = frame.depth_gt[np.ix_(rows, cols)].astype(np.float64)
            sel = (z > 0.5) & (z < 1000.0 * 0.999)  # CARLA's depth camera saturates at 1000 m (sky)
        else:
            z_row = road_depth(rows, camera)
            z = np.broadcast_to(z_row[:, None], (len(rows), len(cols)))
            sel = (z >= self.min_ref_m) & (z <= self.max_ref_m)  # NaN above the horizon: False
            mask = lanes.ego_lane_mask if lanes is not None else None
            if mask is not None and (sel & mask[np.ix_(rows, cols)]).sum() >= 50:
                sel = sel & mask[np.ix_(rows, cols)]
            else:
                lateral = z * (cols[None, :] - camera.K[0, 2]) / camera.K[0, 0]
                sel = sel & (np.abs(lateral) <= self.ref_half_width_m)
            for track in tracks:  # objects stand on the road but aren't part of it
                x1, y1, x2, y2 = np.asarray(track.box, np.float64)
                in_rows = (rows >= y1) & (rows <= y2)
                in_cols = (cols >= x1) & (cols <= x2)
                sel = sel & ~(in_rows[:, None] & in_cols[None, :])
        r, c = np.nonzero(sel)
        return rows[r], cols[c], z[r, c]

    def measure(
        self,
        tracks: list[Track],
        depth: DepthMap | None,
        lanes: LaneResult | None,
        frame: Frame,
        camera: CameraInfo,
    ) -> list[Measurement]:
        self.scale = None
        if depth is None:
            return [(None, None)] * len(tracks)
        if not depth.is_metric:
            rows, cols, z = self.reference_points(frame, lanes, tracks, camera)
            self.scale = fit_inverse_depth(depth.depth[rows, cols], z)
            if self.scale is None:
                return [(None, None)] * len(tracks)
        fx, cx = camera.K[0, 0], camera.K[0, 2]
        out = []
        for track in tracks:
            box = np.asarray(track.box, np.float64)
            region = central_region(depth.depth, box, self.box_shrink)
            if depth.is_metric:
                distance = float(np.quantile(region, self.box_quantile))
            else:  # inverse depth falls as depth rises: the depth quantile q is the inverse's 1 - q
                distance = float(to_metric(np.quantile(region, 1.0 - self.box_quantile), *self.scale))
            if not np.isfinite(distance):
                out.append((None, None))
                continue
            u = (box[0] + box[2]) / 2
            out.append((distance, distance * (u - cx) / fx))
        return out
