"""The interface between a simulator and the closed loop (carla_bridge_node), so the loop can be tested with a
fake simulator and no CARLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from perception.core.types import CameraInfo, ControlCommand


@dataclass
class SimStep:
    image: np.ndarray  # BGR uint8 (H, W, 3), the ego camera
    timestamp: float  # simulated seconds since the scenario started
    camera: CameraInfo
    ego_speed: float  # m/s


class Simulator(Protocol):
    def start(self) -> SimStep:
        """Set up the scenario and return its first frame."""

    def step(self, override: ControlCommand | None) -> SimStep | None:
        """Apply the stack's command for the last frame (None: it didn't answer in time) on top of the ego's
        normal driving, advance one tick, and return the next frame; None when the scenario is over."""

    def close(self) -> dict:
        """Remove what the scenario spawned and return its summary (perception.sim.metrics.ScenarioLog)."""
