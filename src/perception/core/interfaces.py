"""Stage interfaces. Every implementation satisfies one of these Protocols.

Stateful stages (tracker, lane smoothing, distance/TTC history) also implement
`reset()`, which the pipeline calls between sequences or scenarios.

Implementations must import heavy dependencies (torch, ultralytics, tensorrt)
inside their constructors, not at module level, so that registering them stays cheap.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from perception.core.types import (
    ControlCommand,
    DepthMap,
    Detection,
    Frame,
    LaneResult,
    ObjectState,
    PerceptionOutput,
    Track,
)


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    def update(self, detections: list[Detection], frame: Frame) -> list[Track]: ...

    def reset(self) -> None: ...


@runtime_checkable
class LaneDetector(Protocol):
    def detect(self, frame: Frame) -> LaneResult: ...


@runtime_checkable
class DepthEstimator(Protocol):
    def estimate(self, frame: Frame) -> DepthMap: ...


@runtime_checkable
class DistanceEstimator(Protocol):
    """Fusion: tracks + depth + lanes -> distance, TTC and in-lane flag per track."""

    def estimate(
        self,
        tracks: list[Track],
        depth: DepthMap | None,
        lanes: LaneResult | None,
        frame: Frame,
    ) -> list[ObjectState]: ...


@runtime_checkable
class DecisionPolicy(Protocol):
    def decide(self, output: PerceptionOutput, ego_speed_mps: float) -> ControlCommand: ...


@runtime_checkable
class InferenceBackend(Protocol):
    """Runs a model's forward pass only. Pre/post-processing stays in the model class."""

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


@runtime_checkable
class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


@runtime_checkable
class Sink(Protocol):
    def write(self, output: PerceptionOutput) -> None: ...

    def close(self) -> None: ...
