"""Distance from the ground plane: where a box's bottom centre (the object's contact with the road) lies.

Needs only the camera's intrinsics, height and pitch, so it works without a depth model and gives metric
distances directly; it assumes a flat road between the camera and the object. With auto_pitch, the camera
pitch is estimated from the lane lines' vanishing point (perception.fusion.horizon) instead of taken
from the camera, for footage from cameras whose mounting is not known.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from perception.core.registry import register
from perception.core.types import DepthMap, Frame, LaneResult, ObjectState, Track
from perception.fusion.geometry import ground_point
from perception.fusion.horizon import PitchEstimator
from perception.fusion.ttc import TTCEstimator


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


@register("distance", "ground_plane")
class GroundPlaneDistance:
    """DistanceEstimator from box bottoms on a flat road, with TTC and the in-lane check.

    max_distance_m          estimates beyond this (boxes near the horizon) are reported as unknown
    corridor_half_width_m   in-lane fallback when no ego-lane mask is available (half a lane is ~1.8 m)
    ttc_window_s            history used for the closing speed (see perception.fusion.ttc)
    auto_pitch              estimate the camera pitch from lanes (the camera's own pitch is used until
                            enough frames with two lane lines have been seen)
    """

    def __init__(
        self,
        max_distance_m: float = 100.0,
        corridor_half_width_m: float = 1.8,
        ttc_window_s: float = 0.5,
        min_closing_speed: float = 0.5,
        auto_pitch: bool = False,
    ) -> None:
        self.max_distance_m = max_distance_m
        self.corridor_half_width_m = corridor_half_width_m
        self.ttc = TTCEstimator(window_s=ttc_window_s, min_closing_speed=min_closing_speed)
        self.pitch = PitchEstimator() if auto_pitch else None

    def reset(self) -> None:
        self.ttc.reset()
        if self.pitch is not None:
            self.pitch.reset()

    def estimate(
        self, tracks: list[Track], depth: DepthMap | None, lanes: LaneResult | None, frame: Frame
    ) -> list[ObjectState]:
        camera = frame.camera
        if self.pitch is not None:
            pitch = self.pitch.update(lanes, camera)
            if pitch is not None:
                camera = dataclasses.replace(camera, pitch_rad=pitch)
        states = []
        for track in tracks:
            x1, _, x2, y2 = (float(c) for c in np.asarray(track.box))
            u = (x1 + x2) / 2
            point = ground_point(u, y2, camera)
            distance = lateral = None
            if point is not None and point[1] <= self.max_distance_m:
                lateral, distance = point
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
