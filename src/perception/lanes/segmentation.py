"""Segmentation lane detector: a network finds lane-line and drivable-area pixels, geometry turns them into
the ego lane.

The network (TwinLiteNetPlus, exported by scripts/export_twinlitenet.py) predicts, per pixel, lane-line paint
and drivable area. Its drivable area covers every lane the car could drive in (BDD100K's direct and
alternative areas merged in training), so the ego lane comes from the lane lines, and is then clipped to the
drivable area so it doesn't spill onto kerbs or cars. Two ways to find the ego lane's lines:

rows  (default) scan image rows upward from under the car; on each row take the lane-line run nearest the
      line found on the rows below (at the start: nearest the car's column, within half a lane), and fit a
      polynomial x = f(y) to each side. Needs no camera calibration besides the car's column, so it works
      on every BDD100K camera mount and on CARLA. Paint running across the lane (stop lines, crosswalks)
      crosses a row as one wide run instead of a narrow one per line, and is skipped.
bev   the classical detector's pipeline (bird's-eye view, sliding windows) with the network's lane-line
      probabilities in place of the hand-made paint filter: isolates what the learned paint detection adds,
      but inherits the fixed calibration (one vanishing point for all cameras).
"""

from __future__ import annotations

import cv2
import numpy as np

from perception.core import registry
from perception.core.registry import register
from perception.core.types import Frame, LaneResult
from perception.lanes.classical import ClassicalLaneDetector

PAD_VALUE = 114  # grey border used in training


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize keeping the aspect ratio and pad to (height, width) with grey, centred as in training.

    Returns the padded image and the (top, left, h, w) of the image inside it.
    """
    h0, w0 = image.shape[:2]
    scale = min(height / h0, width / w0)
    w, h = round(w0 * scale), round(h0 * scale)
    top, left = (height - h) // 2, (width - w) // 2
    canvas = np.full((height, width, 3), PAD_VALUE, np.uint8)
    canvas[top : top + h, left : left + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas, (top, left, h, w)


def run_centres(cols: np.ndarray, max_width: float = np.inf) -> np.ndarray:
    """Centres of the runs of consecutive column indices (each crossing of a painted line), leaving out
    runs wider than max_width (a stop line or crosswalk band crossing the row, not a lane line)."""
    if not len(cols):
        return cols.astype(np.float64)
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    return np.array([run.mean() for run in runs if len(run) <= max_width], np.float64)


@register("lanes", "segmentation")
class SegLaneDetector(ClassicalLaneDetector):
    """Parameters besides the classical geometry (see ClassicalLaneDetector; with geometry=rows only
    vanishing_point, bottom and lane_width are used, to place a missing line one lane width away):

    model             exported network (.ts for the torch backend, .onnx / .engine later); None builds only
                      the geometry, for lanes_from_masks (e.g. an oracle ego lane from ground-truth masks)
    backend           InferenceBackend type: torch, onnx, tensorrt
    device, half      passed to the backend
    input_size        (width, height) the network was exported at
    train_aspect      width / height of the training images (BDD100K: 16:9). Taller images are cropped from
                      the top to this aspect before the network (the road is at the bottom); the cropped rows
                      get probability 0. Fed whole, a 4:3 CARLA frame fills the input height that training
                      always padded, and the network marks lane lines along the top and bottom edges.
                      None feeds the whole image.
    lane_threshold    lane-line probability counted as paint
    drivable_threshold  drivable-area probability counted as road
    clip_to_drivable  intersect the ego lane with the drivable area
    geometry          rows | bev (see the module docstring)
    ego_x             the car's column, as a fraction of image width
    row_step          rows scanned, every row_step pixels (at image resolution)
    max_jump          largest sideways step of a line between scanned rows, as a fraction of image width
    max_gap           rows without road under the lane centre before the scan stops, as a fraction of height
    max_run           widest paint run on a row still taken as a lane line, as a fraction of image width
    min_points        scanned rows needed to fit a line
    extend_down       draw the ego lane down to where the road under the car starts (above the hood), not
                      only to the lowest line point (lines near the car are often off-frame)
    fallback          when no line is found: corridor (the road within one expected lane width of the car),
                      drivable (the connected road region under the car: unmarked roads are one lane) or none
    one_line          when only one line is found: cut (the road region under the car, cut at that line: the
                      road edge bounds the other side) or width (the other line one expected lane width away)
    one_line_max_lanes  with one_line=cut: the region reaches at most this many expected lane widths from the
                      line (a missed line in an intersection would otherwise give the whole junction); None:
                      no limit
    """

    def __init__(
        self,
        model: str | None,
        backend: str = "torch",
        device: str | None = None,
        half: bool = False,
        input_size: tuple[int, int] = (640, 384),
        train_aspect: float | None = 16 / 9,
        lane_threshold: float = 0.5,
        drivable_threshold: float = 0.5,
        clip_to_drivable: bool = True,
        geometry: str = "rows",
        ego_x: float = 0.5,
        row_step: int = 4,
        max_jump: float = 0.03,
        max_gap: float = 0.05,
        max_run: float = 0.06,
        min_points: int = 8,
        extend_down: bool = True,
        fallback: str = "drivable",
        one_line: str = "cut",
        one_line_max_lanes: float | None = None,
        **classical,
    ) -> None:
        super().__init__(**classical)
        if geometry not in ("rows", "bev"):
            raise ValueError(f"geometry must be rows or bev, not {geometry!r}")
        if fallback not in ("corridor", "drivable", "none"):
            raise ValueError(f"fallback must be corridor, drivable or none, not {fallback!r}")
        if one_line not in ("cut", "width"):
            raise ValueError(f"one_line must be cut or width, not {one_line!r}")
        self.backend = None
        if model is not None:
            self.backend = registry.build(
                "backend",
                {
                    "type": backend,
                    "model": model,
                    "inputs": ["images"],
                    "outputs": ["drivable", "lanes"],
                    "device": device,
                    "half": half,
                },
            )
        self.input_w, self.input_h = int(input_size[0]), int(input_size[1])
        self.train_aspect = train_aspect
        self.lane_threshold = lane_threshold
        self.drivable_threshold = drivable_threshold
        self.clip_to_drivable = clip_to_drivable
        self.geometry = geometry
        self.ego_x = ego_x
        self.row_step = row_step
        self.max_jump = max_jump
        self.max_gap = max_gap
        self.max_run = max_run
        self.min_points = min_points
        self.extend_down = extend_down
        self.fallback = fallback
        self.one_line = one_line
        self.one_line_max_lanes = one_line_max_lanes
        self.drivable: np.ndarray | None = None  # last frame's (H, W) probabilities, for overlays and eval
        self.lanes: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._prev_image_fits: tuple[np.ndarray, np.ndarray] | None = None

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(drivable, lane line) probabilities, each (H, W) float32 at the image's size."""
        if self.backend is None:
            raise RuntimeError("SegLaneDetector built without a model: use lanes_from_masks")
        height, width = image.shape[:2]
        crop = 0
        if self.train_aspect is not None and width / height < self.train_aspect - 1e-3:
            crop = height - round(width / self.train_aspect)
        canvas, (top, left, h, w) = letterbox(image[crop:], self.input_w, self.input_h)
        x = np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)[None], np.float32) / 255.0
        out = self.backend.run({"images": x})
        probs = []
        for name in ("drivable", "lanes"):
            logits = out[name][0, :, top : top + h, left : left + w]
            p = 1.0 / (1.0 + np.exp(logits[0] - logits[1]))  # softmax over the 2 classes, class 1
            full = np.zeros((height, width), np.float32)
            full[crop:] = cv2.resize(
                p.astype(np.float32), (width, height - crop), interpolation=cv2.INTER_LINEAR
            )
            probs.append(full)
        return probs[0], probs[1]

    def bev_paint(self, frame: Frame, to_bev: np.ndarray) -> np.ndarray:
        self.drivable, self.lanes = self.segment(frame.image)
        bev = cv2.warpPerspective(self.lanes, to_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        return bev >= self.lane_threshold

    def detect(self, frame: Frame) -> LaneResult:
        if self.geometry == "bev":
            result = super().detect(frame)
        else:
            self.drivable, self.lanes = self.segment(frame.image)
            result = self._detect_rows(
                self.lanes >= self.lane_threshold, self.drivable >= self.drivable_threshold
            )
        if self.clip_to_drivable and result.ego_lane_mask is not None:
            mask = result.ego_lane_mask
            np.logical_and(mask, self.drivable >= self.drivable_threshold, out=mask)
        return result

    def lanes_from_masks(self, paint: np.ndarray, road: np.ndarray) -> LaneResult:
        """The rows geometry on given (H, W) bool lane-line and drivable masks, e.g. ground truth."""
        result = self._detect_rows(paint, road)
        if self.clip_to_drivable and result.ego_lane_mask is not None:
            np.logical_and(result.ego_lane_mask, road, out=result.ego_lane_mask)
        return result

    # --- geometry=rows -------------------------------------------------------------------------------

    def _lane_width(self, y: np.ndarray, width: int, height: int) -> np.ndarray:
        """Expected ego-lane width in pixels at image rows y, from the configured geometry."""
        vy, yb = self.vanishing_point[1] * height, self.bottom * height
        return self.lane_width * width * np.clip((y - vy) / (yb - vy), 0.0, None)

    def _scan(self, paint: np.ndarray, road: np.ndarray) -> tuple[tuple[list, list], tuple[int, int] | None]:
        """(y, x) points of the ego lane's left and right lines, scanned from under the car upward, and the
        (bottom, top) rows of the road along the lane centre (None if there is no road under the car)."""
        height, width = paint.shape
        x_ref = self.ego_x * width
        below = np.flatnonzero(road[height // 2 :, int(x_ref)])
        if not len(below):
            return ([], []), None
        y = y_start = y_road = height // 2 + int(below.max())
        y_stop = int(self.vanishing_point[1] * height)
        tolerance, gap_limit = self.max_jump * width, self.max_gap * height
        points: tuple[list, list] = ([], [])
        prev: list[float | None] = [None, None]
        off_road = 0
        while y > y_stop:
            centres = run_centres(np.flatnonzero(paint[y]), self.max_run * width)
            # the first point of a line: within one expected lane width of the car's column
            reach = max(self._lane_width(np.array([y]), width, height)[0], 1.0)
            for side, cands in enumerate((centres[centres < x_ref], centres[centres > x_ref])):
                if prev[side] is not None:
                    cands = cands[np.abs(cands - prev[side]) < tolerance]
                    pick = cands[np.argmin(np.abs(cands - prev[side]))] if len(cands) else None
                else:
                    cands = cands[np.abs(cands - x_ref) < reach]
                    pick = (cands.max() if side == 0 else cands.min()) if len(cands) else None
                if pick is not None:
                    points[side].append((y, float(pick)))
                    prev[side] = float(pick)
            if prev[0] is not None and prev[1] is not None:
                x_ref = (prev[0] + prev[1]) / 2
            if road[y, int(np.clip(x_ref, 0, width - 1))]:
                off_road, y_road = 0, y
            else:
                off_road += self.row_step
            if off_road > gap_limit:
                break
            y -= self.row_step
        return points, (y_start, y_road)

    def _fit_points(self, points: list, height: int) -> np.ndarray | None:
        if len(points) < self.min_points:
            return None
        y, x = np.array(points, np.float64).T
        if np.ptp(y) < 0.05 * height:
            return None
        degree = 2 if np.ptp(y) > 0.15 * height else 1
        fit = np.polyfit(y, x, degree)
        residual = np.abs(x - np.polyval(fit, y))
        keep = residual < max(2.0, 2.5 * np.median(residual))
        if keep.sum() >= self.min_points:
            fit = np.polyfit(y[keep], x[keep], degree)
        return np.pad(fit, (3 - len(fit), 0))

    def _detect_rows(self, paint: np.ndarray, road: np.ndarray) -> LaneResult:
        height, width = paint.shape
        (left_pts, right_pts), span = self._scan(paint, road)
        left, right = self._fit_points(left_pts, height), self._fit_points(right_pts, height)
        rows_seen = [y for y, _ in left_pts + right_pts]
        if not rows_seen or (left is None and right is None):
            self._prev_image_fits = None
            return self._fallback(road, span)
        y_top, y_bottom = min(rows_seen), max(rows_seen)

        confidence = 1.0
        if left is not None and right is not None:
            w_bottom = np.polyval(right, y_bottom) - np.polyval(left, y_bottom)
            w_top = np.polyval(right, y_top) - np.polyval(left, y_top)
            expected = self._lane_width(np.array([y_bottom]), width, height)[0]
            if not (0.4 * expected < w_bottom < 2.5 * expected and w_top < w_bottom):
                # not one lane: keep the line with more support
                if len(left_pts) >= len(right_pts):
                    right = None
                else:
                    left = None
        known_side = None
        if left is None or right is None:
            confidence = 0.5
            known_side = "left" if left is not None else "right"
            ys = np.linspace(y_top, y_bottom, 10)
            known = left if left is not None else right
            sign = 1.0 if left is not None else -1.0
            other = np.polyfit(ys, np.polyval(known, ys) + sign * self._lane_width(ys, width, height), 2)
            left, right = (known, other) if left is not None else (other, known)

        if self._prev_image_fits is not None and self.smoothing < 1:
            a = self.smoothing
            left = a * left + (1 - a) * self._prev_image_fits[0]
            right = a * right + (1 - a) * self._prev_image_fits[1]
        self._prev_image_fits = (left, right)

        if known_side is not None and self.one_line == "cut":
            mask = self._road_region(road, span)
            line = left if known_side == "left" else right
            xs = np.polyval(line, np.arange(height))[:, None]
            cols = np.arange(width)[None, :]
            mask &= cols > xs if known_side == "left" else cols < xs
            if self.one_line_max_lanes is not None:
                reach = self.one_line_max_lanes * self._lane_width(np.arange(height), width, height)[:, None]
                mask &= np.abs(cols - xs) < reach
            if mask.any():
                return LaneResult(left=left, right=right, ego_lane_mask=mask, confidence=confidence)
        if self.extend_down:
            y_bottom = max(y_bottom, span[0])
        rows = np.linspace(y_top, y_bottom, 30)
        polygon = np.vstack([np.c_[np.polyval(left, rows), rows], np.c_[np.polyval(right, rows), rows][::-1]])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        return LaneResult(left=left, right=right, ego_lane_mask=mask.astype(bool), confidence=confidence)

    def _fallback(self, road: np.ndarray, span: tuple[int, int] | None) -> LaneResult:
        """Ego lane without lane lines (confidence 0.25), or an empty result."""
        if self.fallback == "none" or span is None:
            return LaneResult(left=None, right=None, ego_lane_mask=None, confidence=0.0)
        height, width = road.shape
        y_bottom, y_top = span
        x = self.ego_x * width
        if self.fallback == "drivable":
            mask = self._road_region(road, span)
        else:
            rows = np.linspace(y_top, y_bottom, 20)
            half = self._lane_width(rows, width, height) / 2
            polygon = np.vstack([np.c_[x - half, rows], np.c_[x + half, rows][::-1]])
            canvas = np.zeros((height, width), np.uint8)
            cv2.fillPoly(canvas, [np.round(polygon).astype(np.int32)], 1)
            mask = canvas.astype(bool) & road
        if not mask.any():
            return LaneResult(left=None, right=None, ego_lane_mask=None, confidence=0.0)
        return LaneResult(left=None, right=None, ego_lane_mask=mask, confidence=0.25)

    def _road_region(self, road: np.ndarray, span: tuple[int, int]) -> np.ndarray:
        """The connected drivable region under the car, up to where the scan left the road."""
        y_bottom, y_top = span
        _, labels = cv2.connectedComponents(road.astype(np.uint8), connectivity=4)
        mask = labels == labels[y_bottom, int(self.ego_x * road.shape[1])]
        mask[:y_top] = False
        return mask
