"""What happened in a closed-loop scenario, measured from the simulator's ground truth.

The perception stack's own distance estimates are not used here: the gap is the true bumper-to-object
distance from the actors' 3D boxes, so the numbers judge the stack instead of repeating it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def forward_gap(
    world_to_ego: np.ndarray, ego_front_m: float, ego_half_width_m: float, corners: np.ndarray
) -> tuple[float, bool]:
    """(gap, in_path): how far ahead of the ego's front bumper the object's nearest point is, and whether
    the object overlaps the ego's width (i.e. the ego would hit it by driving straight on).

    world_to_ego    4x4 matrix from world to the ego's frame (x forward, y right; CARLA's
                    Transform.get_inverse_matrix())
    ego_front_m     front bumper's x in the ego frame (bounding box centre x + half length)
    corners         (8, 3) world corners of the object's 3D box
    The gap is negative once the object reaches the bumper (a collision, or the object is beside us).
    """
    points = np.c_[np.asarray(corners, np.float64).reshape(-1, 3), np.ones(8)] @ np.asarray(world_to_ego).T
    x, y = points[:, 0], points[:, 1]
    gap = float(x.min() - ego_front_m)
    in_path = bool(y.min() <= ego_half_width_m and y.max() >= -ego_half_width_m)
    return gap, in_path


@dataclass
class Tick:
    t: float  # simulated seconds since the scenario started
    ego_speed: float  # m/s
    gap_m: float  # true gap to the scenario's target (nan when there is none)
    in_path: bool
    throttle: float
    brake: float
    steer: float
    reason: str  # the policy's reason when it brakes, else ""
    collided: bool  # the ego has collided at or before this tick
    answered: bool  # the stack's command for this frame arrived in time
    # true distance from the camera to the target's nearest point (the stack's distance_m measures this;
    # gap_m is from the front bumper)
    camera_gap_m: float = math.nan


class ScenarioLog:
    """Per-tick record of a scenario, and its summary."""

    def __init__(self) -> None:
        self.ticks: list[Tick] = []
        self.collision_with: str | None = None

    def add(self, tick: Tick) -> None:
        self.ticks.append(tick)

    def rows(self) -> list[dict]:
        return [asdict(t) for t in self.ticks]

    def summary(self) -> dict:
        """Collision or not, the closest in-path gap, and the first intervention: when, at what speed and
        gap, and why. Stopping gap = the gap once the ego stood still after braking."""
        ticks = self.ticks
        out = {
            "ticks": len(ticks),
            "duration_s": round(ticks[-1].t, 2) if ticks else 0.0,
            "collision": any(t.collided for t in ticks),
            "collision_with": self.collision_with,
            "max_speed_mps": round(max((t.ego_speed for t in ticks), default=0.0), 2),
            "unanswered_frames": sum(not t.answered for t in ticks),
        }
        in_path = [t.gap_m for t in ticks if t.in_path and not math.isnan(t.gap_m)]
        out["min_gap_in_path_m"] = round(min(in_path), 2) if in_path else None
        first = next((t for t in ticks if t.reason and t.brake > 0), None)
        out["braked"] = first is not None
        if first is not None:
            out["first_brake"] = {
                "t": round(first.t, 2),
                "ego_speed_mps": round(first.ego_speed, 2),
                "gap_m": None if math.isnan(first.gap_m) else round(first.gap_m, 2),
                "reason": first.reason,
            }
            stopped = next((t for t in ticks if t.t > first.t and t.ego_speed < 0.1), None)
            out["stopped_gap_m"] = (
                None if stopped is None or math.isnan(stopped.gap_m) else round(stopped.gap_m, 2)
            )
        return out
