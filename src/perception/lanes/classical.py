"""Classical lane detector (OpenCV): bird's-eye view, ridge filter, sliding windows, polynomial fit.

1. Warp a trapezoid bounded by two lines through the vanishing point to a rectangle, the bird's-eye view
   (BEV): lines that are parallel on the road become parallel and vertical.
2. Lane markings are thin stripes brighter than the road right next to them. A horizontal top-hat filter
   on lightness (white paint) and saturation (yellow paint) picks them out whatever the overall brightness
   (day, night, shade), and a threshold relative to the frame's own response keeps the strongest.
3. A column histogram of the lower BEV has a peak per line; the peaks nearest the centre on each side
   start sliding windows that follow each line up, and a 2nd-order polynomial x = f(y) is fitted to its
   pixels. The whole BEV width is searched, since lane width in the BEV varies between cameras.
4. The two lines must be roughly parallel and a plausible lane width apart; when only one is usable, the
   other is placed a lane width away. On video, the last fit narrows the next frame's search and fits
   are smoothed over time.
5. The lines are mapped back to the image and the ego lane is the area between them, from the top of the
   BEV down to extend_to: straight BEV lines stay straight in the image, so they can be extended below the
   searched region (e.g. down to the hood of cars whose hood sits lower).

Geometry is given as fractions of image width and height, so one config works at any resolution: the
vanishing point, the bottom row (above the hood) and the lane width at that row. For BDD100K they are
fitted by scripts/calibrate_lanes.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from perception.core.registry import register
from perception.core.types import Frame, LaneResult


@register("lanes", "classical")
class ClassicalLaneDetector:
    """Parameters (sizes in BEV pixels; the BEV is bev_size wide and spans lanes_across lane widths):

    vanishing_point  (x, y) as fractions of image width and height
    bottom           lowest image row used, as a fraction of image height (above the hood)
    lane_width       ego-lane width at the bottom row, as a fraction of image width
    top              where the BEV starts, as a fraction of the way from the vanishing point down to bottom
                     (smaller looks further ahead, with fewer pixels per metre)
    tophat_px        top-hat kernel width; wider than a marking, narrower than a lane
    min_response     smallest top-hat response counted as paint (0-255 scale)
    paint_percentile only the strongest responses count as paint (above this percentile of the frame)
    min_peak         histogram height (smoothed paint pixels per column) that can start a line
    windows, window_margin, min_window_pixels  sliding-window search
    search_margin    on video: search this far around the previous frame's line
    min_line_pixels  pixels needed to accept a line
    degree           1 fits straight lines, 2 allows curves (always returned as 3 coefficients)
    smoothing        on video: weight of the new fit (1 = no smoothing)
    extend_to        ego lane drawn down to this row (fraction of image height); None = bottom
    """

    def __init__(
        self,
        vanishing_point: Sequence[float] = (0.5, 0.5),
        bottom: float = 0.9,
        lane_width: float = 0.4,
        top: float = 0.2,
        lanes_across: float = 3.0,
        bev_size: Sequence[int] = (300, 400),
        tophat_px: int = 15,
        min_response: float = 25.0,
        paint_percentile: float = 97.0,
        min_peak: float = 15.0,
        windows: int = 10,
        window_margin: int = 25,
        min_window_pixels: int = 15,
        search_margin: int = 20,
        min_line_pixels: int = 150,
        degree: int = 1,
        smoothing: float = 0.5,
        extend_to: float | None = None,
    ) -> None:
        self.vanishing_point = (float(vanishing_point[0]), float(vanishing_point[1]))
        self.bottom = bottom
        self.lane_width = lane_width
        self.top = top
        self.lanes_across = lanes_across
        self.bev_w, self.bev_h = int(bev_size[0]), int(bev_size[1])
        self.lane_px = self.bev_w / lanes_across
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_px, 1))
        self.min_response = min_response
        self.paint_percentile = paint_percentile
        self.min_peak = min_peak
        self.windows = windows
        self.window_margin = window_margin
        self.min_window_pixels = min_window_pixels
        self.search_margin = search_margin
        self.min_line_pixels = min_line_pixels
        self.degree = degree
        self.smoothing = smoothing
        self.extend_to = extend_to
        self._warps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[np.ndarray, np.ndarray] | None = None  # last (left, right) BEV fits

    def trapezoid(self, width: int, height: int) -> np.ndarray:
        """Image corners of the BEV region: bottom-left, bottom-right, top-right, top-left."""
        vx, vy = self.vanishing_point[0] * width, self.vanishing_point[1] * height
        yb = self.bottom * height
        yt = vy + self.top * (yb - vy)
        half = self.lanes_across / 2 * self.lane_width * width
        return np.float32(
            [[vx - half, yb], [vx + half, yb], [vx + half * self.top, yt], [vx - half * self.top, yt]]
        )

    def _warp(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if (width, height) not in self._warps:
            bev = np.float32([[0, self.bev_h], [self.bev_w, self.bev_h], [self.bev_w, 0], [0, 0]])
            src = self.trapezoid(width, height)
            self._warps[width, height] = (
                cv2.getPerspectiveTransform(src, bev),
                cv2.getPerspectiveTransform(bev, src),
            )
        return self._warps[width, height]

    def paint_mask(self, bev_bgr: np.ndarray) -> np.ndarray:
        """(H, W) bool: BEV pixels that look like lane paint."""
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        lightness = cv2.morphologyEx(hls[..., 1], cv2.MORPH_TOPHAT, self.kernel)
        saturation = cv2.morphologyEx(hls[..., 2], cv2.MORPH_TOPHAT, self.kernel)
        response = np.maximum(lightness, saturation).astype(np.float32)
        valid = bev_bgr.any(axis=2)  # parts of the trapezoid outside the image are black
        if not valid.any():
            return np.zeros(response.shape, bool)
        threshold = max(self.min_response, float(np.percentile(response[valid], self.paint_percentile)))
        return (response >= threshold) & valid

    def _window_search(self, ys: np.ndarray, xs: np.ndarray, start_x: float) -> np.ndarray:
        """Indices of paint pixels found by sliding windows starting at column start_x."""
        found = []
        x = start_x
        window_h = self.bev_h / self.windows
        for w in range(self.windows):
            y_hi = self.bev_h - w * window_h
            inside = np.flatnonzero(
                (ys >= y_hi - window_h) & (ys < y_hi) & (np.abs(xs - x) < self.window_margin)
            )
            found.append(inside)
            if len(inside) >= self.min_window_pixels:
                x = xs[inside].mean()
        return np.concatenate(found)

    def _fit(self, ys: np.ndarray, xs: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
        """x = f(y) through the pixels idx, refitted without outliers; None if there is too little paint."""
        if len(idx) < self.min_line_pixels or np.ptp(ys[idx]) < self.bev_h / 3:
            return None
        y, x = ys[idx], xs[idx]
        fit = np.polyfit(y, x, self.degree)
        residual = np.abs(x - np.polyval(fit, y))
        keep = residual < max(3.0, 2.5 * np.median(residual))
        if keep.sum() >= self.min_line_pixels:
            fit = np.polyfit(y[keep], x[keep], self.degree)
        return np.pad(fit, (3 - len(fit), 0))  # always (a, b, c) of a y^2 + b y + c

    def _find_lines(self, mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        ys, xs = np.nonzero(mask)
        ys, xs = ys.astype(np.float64), xs.astype(np.float64)
        if self._prev is not None:
            fits = []
            for prev in self._prev:
                idx = np.flatnonzero(np.abs(xs - np.polyval(prev, ys)) < self.search_margin)
                fits.append(self._fit(ys, xs, idx))
            if all(f is not None for f in fits):
                return fits[0], fits[1]

        centre = self.bev_w / 2
        histogram = np.bincount(xs[ys > self.bev_h / 2].astype(int), minlength=self.bev_w).astype(float)
        histogram = np.convolve(histogram, np.ones(9) / 9, mode="same")
        local_max = np.lib.stride_tricks.sliding_window_view(np.pad(histogram, 7, mode="edge"), 15).max(
            axis=1
        )
        peaks = np.flatnonzero((histogram >= self.min_peak) & (histogram == local_max))
        left_peaks, right_peaks = peaks[peaks < centre], peaks[peaks > centre]
        starts = (
            left_peaks.max() if len(left_peaks) else None,
            right_peaks.min() if len(right_peaks) else None,
        )
        fits = [None if s is None else self._fit(ys, xs, self._window_search(ys, xs, s)) for s in starts]
        return fits[0], fits[1]

    def _check(self, left, right) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Lines must be roughly parallel and a plausible lane width apart; returns (left, right, confidence).

        The width range is loose because lane width in the BEV differs between cameras.
        """
        shift = np.array([0.0, 0.0, self.lane_px])
        if left is not None and right is not None:
            ys = np.linspace(0, self.bev_h, 5)
            width = np.polyval(right, ys) - np.polyval(left, ys)
            mean = width.mean()
            if 0.5 * self.lane_px < mean < 2.5 * self.lane_px and np.ptp(width) < 0.3 * mean:
                return left, right, 1.0
            # Not a lane: keep the line nearer to where an ego-lane line is expected
            expected = self.bev_w / 2 - self.lane_px / 2, self.bev_w / 2 + self.lane_px / 2
            if abs(np.polyval(left, self.bev_h) - expected[0]) <= abs(
                np.polyval(right, self.bev_h) - expected[1]
            ):
                right = None
            else:
                left = None
        if left is not None:
            return left, left + shift, 0.5
        if right is not None:
            return right - shift, right, 0.5
        return None, None, 0.0

    def detect(self, frame: Frame) -> LaneResult:
        height, width = frame.image.shape[:2]
        to_bev, to_image = self._warp(width, height)
        bev = cv2.warpPerspective(frame.image, to_bev, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        left, right, confidence = self._check(*self._find_lines(self.paint_mask(bev)))

        if left is None:
            self._prev = None
            return LaneResult(left=None, right=None, ego_lane_mask=None, confidence=0.0)
        if self._prev is not None and self.smoothing < 1:
            a = self.smoothing
            left = a * left + (1 - a) * self._prev[0]
            right = a * right + (1 - a) * self._prev[1]
        self._prev = (left, right)

        ys = np.linspace(0, self.bev_h, 20)
        image_fits = []
        for fit in (left, right):
            pts = cv2.perspectiveTransform(np.c_[np.polyval(fit, ys), ys][None].astype(np.float32), to_image)[
                0
            ]
            image_fits.append(np.polyfit(pts[:, 1], pts[:, 0], 2))

        trapezoid = self.trapezoid(width, height)
        y_top, y_bottom = trapezoid[2, 1], (self.extend_to if self.extend_to else self.bottom) * height
        rows = np.linspace(y_top, y_bottom, 30)
        polygon = np.vstack(
            [np.c_[np.polyval(image_fits[0], rows), rows], np.c_[np.polyval(image_fits[1], rows), rows][::-1]]
        )
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        return LaneResult(
            left=image_fits[0], right=image_fits[1], ego_lane_mask=mask.astype(bool), confidence=confidence
        )
