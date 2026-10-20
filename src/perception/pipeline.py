"""PerceptionPipeline: runs the stages for one frame and returns a PerceptionOutput.

The same object is used offline (video/dataset), in CARLA and inside the ROS2 node.
The decision policy is not part of it; the caller (or decision_node) applies it.
"""

from __future__ import annotations

from pathlib import Path

from perception.core import registry
from perception.core.config import PipelineConfig, load_config
from perception.core.interfaces import (
    DepthEstimator,
    Detector,
    DistanceEstimator,
    LaneDetector,
    Tracker,
)
from perception.core.timing import StageTimer
from perception.core.types import DepthMap, Frame, PerceptionOutput


class PerceptionPipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        distance: DistanceEstimator,
        lanes: LaneDetector | None = None,
        depth: DepthEstimator | None = None,
        depth_every_n: int = 1,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.distance = distance
        self.lanes = lanes
        self.depth = depth
        self.depth_every_n = depth_every_n
        self._frames_seen = 0
        self._last_depth: DepthMap | None = None

    @classmethod
    def from_config(cls, cfg: PipelineConfig | str | Path) -> PerceptionPipeline:
        if not isinstance(cfg, PipelineConfig):
            cfg = load_config(cfg)
        return cls(
            detector=registry.build("detector", cfg.detector),
            tracker=registry.build("tracker", cfg.tracker),
            distance=registry.build("distance", cfg.distance),
            lanes=registry.build("lanes", cfg.lanes) if cfg.lanes else None,
            depth=registry.build("depth", cfg.depth) if cfg.depth else None,
            depth_every_n=cfg.depth_every_n,
        )

    def process(self, frame: Frame) -> PerceptionOutput:
        timer = StageTimer()

        with timer.measure("detect"):
            detections = self.detector.detect(frame)
        with timer.measure("track"):
            tracks = self.tracker.update(detections, frame)

        lanes = None
        if self.lanes is not None:
            with timer.measure("lanes"):
                lanes = self.lanes.detect(frame)

        if self.depth is not None and self._frames_seen % self.depth_every_n == 0:
            with timer.measure("depth"):
                self._last_depth = self.depth.estimate(frame)
        depth = self._last_depth

        with timer.measure("distance"):
            objects = self.distance.estimate(tracks, depth, lanes, frame)

        self._frames_seen += 1
        return PerceptionOutput(
            frame=frame, objects=objects, lanes=lanes, depth=depth, timings_ms=timer.timings_ms
        )

    def reset(self) -> None:
        """Clear all temporal state, e.g. between scenarios or MOT sequences."""
        self._frames_seen = 0
        self._last_depth = None
        for stage in (self.detector, self.tracker, self.lanes, self.depth, self.distance):
            reset = getattr(stage, "reset", None)
            if callable(reset):
                reset()
