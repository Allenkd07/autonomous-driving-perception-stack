"""What every DistanceEstimator does once it has a distance per track: TTC and the in-lane check.

Subclasses implement measure() (one distance and lateral offset per track) and may override camera() to
correct the camera model per frame (auto pitch).
"""

from __future__ import annotations

import numpy as np

from perception.core.types import CameraInfo, DepthMap, Frame, LaneResult, ObjectState, Track
from perception.fusion.ttc import TTCEstimator

Measurement = tuple[float | None, float | None]  # (forward distance, lateral offset) in metres


def lane_span(lanes: LaneResult, row: float) -> tuple[float, float] | None:
    """The ego lane's left and right edge (pixel columns) on an image row, from its two lines; None without
    both lines (a mask alone may be the whole road, see SegLaneDetector's fallback) or beyond the vanishing
    point.

    An obstacle straight ahead hides the road behind it, and the lane scan stops where the road does: the
    ego-lane mask (and the lines fitted to it) ends a few rows below the obstacle's bottom, exactly where the
    in-lane question matters most. The lines are therefore extrapolated upwards, where they converge towards
    the vanishing point. Rows below the mask (a box cut off by the image's bottom edge) use its bottom row.
    """
    if lanes.left is None or lanes.right is None or lanes.ego_lane_mask is None:
        return None
    rows = np.flatnonzero(lanes.ego_lane_mask.any(axis=1))
    if not len(rows):
        return None
    r = min(row, float(rows[-1]))
    left, right = float(np.polyval(lanes.left, r)), float(np.polyval(lanes.right, r))
    return (left, right) if right > left else None  # beyond the vanishing point the lines cross


def in_ego_lane(
    box,
    lateral_m: float | None,
    lanes: LaneResult | None,
    corridor_half_width_m: float,
    min_lane_overlap: float = 0.25,
) -> bool:
    """Does the object stand in the ego lane?

    With the ego lane's two lines: the box's bottom edge (where the object meets the road) is compared with
    the lane's span on that row (lane_span). The object is in the lane when at least half its width, or
    min_lane_overlap of the lane's width, lies inside: a pedestrian stepping in counts once half of them is
    in, a wide vehicle cutting in once it takes a quarter of the lane, while a car parked over the line does
    not. Otherwise: a straight corridor of +-corridor_half_width_m around the camera on the object's lateral
    offset, so an obstacle ahead is still caught when no lane is found.
    """
    x1, _, x2, y2 = (float(c) for c in np.asarray(box))
    span = lane_span(lanes, y2) if lanes is not None else None
    if span is None:
        return lateral_m is not None and abs(lateral_m) <= corridor_half_width_m
    left, right = span
    overlap = min(x2, right) - max(x1, left)
    return overlap > 0 and overlap >= min(0.5 * (x2 - x1), min_lane_overlap * (right - left))


class TrackDistanceEstimator:
    """Base for DistanceEstimators.

    max_distance_m          estimates beyond this are reported as unknown
    corridor_half_width_m   in-lane fallback when no ego-lane mask is available (half a lane is ~1.8 m)
    ttc_window_s            history used for the closing speed (see perception.fusion.ttc)
    """

    def __init__(
        self,
        max_distance_m: float = 100.0,
        corridor_half_width_m: float = 1.8,
        ttc_window_s: float = 0.5,
        min_closing_speed: float = 0.5,
    ) -> None:
        self.max_distance_m = max_distance_m
        self.corridor_half_width_m = corridor_half_width_m
        self.ttc = TTCEstimator(window_s=ttc_window_s, min_closing_speed=min_closing_speed)

    def reset(self) -> None:
        self.ttc.reset()

    def camera(self, lanes: LaneResult | None, frame: Frame) -> CameraInfo:
        """The camera model for this frame (called once per frame, before measure)."""
        return frame.camera

    def measure(
        self,
        tracks: list[Track],
        depth: DepthMap | None,
        lanes: LaneResult | None,
        frame: Frame,
        camera: CameraInfo,
    ) -> list[Measurement]:
        raise NotImplementedError

    def estimate(
        self, tracks: list[Track], depth: DepthMap | None, lanes: LaneResult | None, frame: Frame
    ) -> list[ObjectState]:
        camera = self.camera(lanes, frame)
        states = []
        for track, (distance, lateral) in zip(
            tracks, self.measure(tracks, depth, lanes, frame, camera), strict=True
        ):
            if distance is not None and not 0 < distance <= self.max_distance_m:
                distance = lateral = None
            ttc = self.ttc.update(track.track_id, frame.timestamp, distance) if distance is not None else None
            states.append(
                ObjectState(
                    track=track,
                    distance_m=distance,
                    ttc_s=ttc,
                    in_ego_lane=in_ego_lane(track.box, lateral, lanes, self.corridor_half_width_m),
                )
            )
        self.ttc.forget_older_than(frame.timestamp)
        return states
