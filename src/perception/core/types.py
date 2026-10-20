"""Data contracts shared by every stage.

Conventions:
    images    BGR uint8, shape (H, W, 3)  (OpenCV native)
    boxes     xyxy in pixels, float
    distance  metres
    time      seconds (simulation time when running in CARLA)
    camera    x right, y down, z forward
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraInfo:
    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsics
    height_m: float  # mounting height above ground, used for ground-plane distance
    pitch_rad: float = 0.0

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float, height_m: float) -> CameraInfo:
        """Pinhole intrinsics from a horizontal FOV, as CARLA cameras are configured."""
        f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
        return cls(width=width, height=height, K=K, height_m=height_m)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8 (H, W, 3)
    timestamp: float
    frame_id: int
    camera: CameraInfo
    depth_gt: np.ndarray | None = None  # metres; CARLA only, for evaluation


@dataclass
class Detection:
    box: np.ndarray  # xyxy
    score: float
    class_id: int
    class_name: str


@dataclass
class Track(Detection):
    track_id: int = -1
    age: int = 0  # frames since first seen


@dataclass
class LaneResult:
    left: np.ndarray | None  # polynomial coeffs, x = f(y) in image space
    right: np.ndarray | None
    ego_lane_mask: np.ndarray | None  # (H, W) bool
    confidence: float


@dataclass
class DepthMap:
    depth: np.ndarray  # (H, W) float32
    is_metric: bool  # False = relative (raw MiDaS), True = metres


@dataclass
class ObjectState:
    track: Track
    distance_m: float | None
    ttc_s: float | None  # None if not approaching
    in_ego_lane: bool


@dataclass
class PerceptionOutput:
    frame: Frame
    objects: list[ObjectState]
    lanes: LaneResult | None = None
    depth: DepthMap | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ControlCommand:
    throttle: float
    brake: float
    steer: float
    reason: str = ""  # e.g. "EMERGENCY_BRAKE track 12 ttc=1.3s"
