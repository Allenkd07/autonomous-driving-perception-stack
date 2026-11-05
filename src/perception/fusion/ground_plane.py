"""Distance from the ground plane: where a box's bottom centre (the object's contact with the road) lies.

Needs only the camera's intrinsics, height and pitch, so it works without a depth model and gives metric
distances directly; it assumes a flat road between the camera and the object. With auto_pitch, the camera
pitch is estimated from the lane lines' vanishing point (perception.fusion.horizon) instead of taken
from the camera, for footage from cameras whose mounting is not known.

Its error grows with the square of the distance: one pixel of box bottom is dz = z^2 / (f * height) metres
(0.16 m at 10 m, 1.4 m at 30 m for CARLA's 800 px wide, 90 degree camera 1.6 m up), and the same holds for
pitch errors. A box whose bottom is hidden (by another car, or cut off by the image edge) gives a wrong
contact point.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from perception.core.registry import register
from perception.core.types import CameraInfo, DepthMap, Frame, LaneResult, Track
from perception.fusion.base import Measurement, TrackDistanceEstimator, in_ego_lane
from perception.fusion.geometry import ground_point
from perception.fusion.horizon import PitchEstimator

__all__ = ["GroundPlaneDistance", "in_ego_lane"]


@register("distance", "ground_plane")
class GroundPlaneDistance(TrackDistanceEstimator):
    """DistanceEstimator from box bottoms on a flat road, with TTC and the in-lane check.

    auto_pitch   estimate the camera pitch from lanes (the camera's own pitch is used until enough frames
                 with two lane lines have been seen)
    other parameters: see TrackDistanceEstimator
    """

    def __init__(self, auto_pitch: bool = False, **base) -> None:
        super().__init__(**base)
        self.pitch = PitchEstimator() if auto_pitch else None

    def reset(self) -> None:
        super().reset()
        if self.pitch is not None:
            self.pitch.reset()

    def camera(self, lanes: LaneResult | None, frame: Frame) -> CameraInfo:
        if self.pitch is not None:
            pitch = self.pitch.update(lanes, frame.camera)
            if pitch is not None:
                return dataclasses.replace(frame.camera, pitch_rad=pitch)
        return frame.camera

    def measure(
        self,
        tracks: list[Track],
        depth: DepthMap | None,
        lanes: LaneResult | None,
        frame: Frame,
        camera: CameraInfo,
    ) -> list[Measurement]:
        out = []
        for track in tracks:
            x1, _, x2, y2 = (float(c) for c in np.asarray(track.box))
            point = ground_point((x1 + x2) / 2, y2, camera)
            out.append((point[1], point[0]) if point is not None else (None, None))
        return out
