"""What every DistanceEstimator does once it has a distance per track: TTC and the in-lane check.

Subclasses implement measure() (one distance and lateral offset per track) and may override camera() to
correct the camera model per frame (auto pitch).
"""

from __future__ import annotations

import numpy as np

from perception.core.types import CameraInfo, DepthMap, Frame, LaneResult, ObjectState, Track
from perception.fusion.ttc import TTCEstimator

Measurement = tuple[float | None, float | None]  # (forward distance, lateral offset) in metres


def in_ego_lane(
    u: float, v: float, lateral_m: float | None, lanes: LaneResult | None, corridor_half_width_m: float
) -> bool:
    """Is the road point under pixel (u, v) in the ego lane?

    Uses the lane detector's ego-lane mask when there is one; otherwise a straight corridor of
    +-corridor_half_width_m around the camera, so an obstacle ahead is still caught when no lane is found.
    """
    if lanes is not None and lanes.ego_lane_mask is not None:
        mask = lanes.ego_lane_mask
        row = min(max(int(v), 0), mask.shape[0] - 1)
        col = min(max(int(u), 0), mask.shape[1] - 1)
        return bool(mask[row, col])
    return lateral_m is not None and abs(lateral_m) <= corridor_half_width_m


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
            x1, _, x2, y2 = (float(c) for c in np.asarray(track.box))
            u = (x1 + x2) / 2
            ttc = self.ttc.update(track.track_id, frame.timestamp, distance) if distance is not None else None
            states.append(
                ObjectState(
                    track=track,
                    distance_m=distance,
                    ttc_s=ttc,
                    in_ego_lane=in_ego_lane(u, y2, lateral, lanes, self.corridor_half_width_m),
                )
            )
        self.ttc.forget_older_than(frame.timestamp)
        return states
