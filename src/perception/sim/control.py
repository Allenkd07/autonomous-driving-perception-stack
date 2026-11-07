"""The ego car's normal driving: follow a lane's centre line at a set speed, and the policy's override on top.

In the closed loop the route follower plays the human (or planner) driving the car, and the perception
stack's policy only intervenes (perception.planning): a command with brake > 0 replaces throttle and brake,
while the follower keeps steering, so braking in a curve stays in the lane.

Coordinates are CARLA's (Unreal): x forward, y right, z up, yaw in degrees, positive turning right. Positive
steer turns right.
"""

from __future__ import annotations

import math

import numpy as np

from perception.core.types import ControlCommand


class LaneFollower:
    """Pure pursuit on a centre-line path plus a PI speed controller.

    Pure pursuit steers the car along the circle arc through a point lookahead metres ahead on the path:
    curvature = 2 * lateral offset of that point / distance to it^2,
    steer angle = atan(wheelbase * curvature).

    path            (N, 2) world x, y of the lane centre, in driving order (e.g. waypoints 2 m apart)
    target_speed    m/s
    lookahead_m     minimum lookahead; grows with speed (lookahead_s) so steering stays smooth when fast
    wheelbase_m, max_steer_deg   the car's geometry (defaults: a mid-size car)
    kp, ki          throttle per m/s of speed error, and per m of accumulated error
    max_throttle    caps acceleration
    """

    def __init__(
        self,
        path: np.ndarray,
        target_speed: float,
        lookahead_m: float = 6.0,
        lookahead_s: float = 0.8,
        wheelbase_m: float = 2.9,
        max_steer_deg: float = 70.0,
        kp: float = 0.35,
        ki: float = 0.05,
        max_throttle: float = 0.75,
    ) -> None:
        self.path = np.asarray(path, np.float64)
        if self.path.ndim != 2 or self.path.shape[1] != 2 or len(self.path) < 2:
            raise ValueError("path must be (N >= 2, 2) world x, y")
        self.target_speed = target_speed
        self.lookahead_m = lookahead_m
        self.lookahead_s = lookahead_s
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = math.radians(max_steer_deg)
        self.kp, self.ki = kp, ki
        self.max_throttle = max_throttle
        self.integral = 0.0
        self.progress = 0  # index of the path point nearest the car; only moves forward

    def _lookahead_point(self, x: float, y: float, speed: float) -> np.ndarray:
        # Search near the last position only, so a path that loops back past the car isn't matched early
        window = self.path[self.progress : self.progress + 50]
        self.progress += int(np.argmin(np.hypot(window[:, 0] - x, window[:, 1] - y)))
        distance = max(self.lookahead_m, self.lookahead_s * speed)
        ahead = self.path[self.progress :]
        far = np.hypot(ahead[:, 0] - x, ahead[:, 1] - y) >= distance
        return ahead[int(np.argmax(far))] if far.any() else ahead[-1]

    def steer(self, x: float, y: float, yaw_deg: float, speed: float) -> float:
        tx, ty = self._lookahead_point(x, y, speed) - (x, y)
        yaw = math.radians(yaw_deg)
        forward = math.cos(yaw) * tx + math.sin(yaw) * ty
        lateral = -math.sin(yaw) * tx + math.cos(yaw) * ty  # > 0: to the right
        distance_sq = forward**2 + lateral**2
        if distance_sq < 1e-6:
            return 0.0
        angle = math.atan(self.wheelbase_m * 2.0 * lateral / distance_sq)
        return float(np.clip(angle / self.max_steer_rad, -1.0, 1.0))

    def speed_control(self, speed: float, dt: float) -> tuple[float, float]:
        """(throttle, brake) towards target_speed."""
        error = self.target_speed - speed
        self.integral = float(np.clip(self.integral + error * dt, -5.0, 5.0))
        command = self.kp * error + self.ki * self.integral
        if command >= 0:
            return min(command, self.max_throttle), 0.0
        # Too fast by more than ~1 m/s: brake gently; smaller errors just coast
        return 0.0, float(np.clip(-command - 0.35, 0.0, 0.5))

    def control(self, x: float, y: float, yaw_deg: float, speed: float, dt: float) -> ControlCommand:
        throttle, brake = self.speed_control(speed, dt)
        return ControlCommand(throttle=throttle, brake=brake, steer=self.steer(x, y, yaw_deg, speed))

    @property
    def finished(self) -> bool:
        """The car has reached the end of the path."""
        return self.progress >= len(self.path) - 2


def apply_override(follow: ControlCommand, override: ControlCommand | None) -> ControlCommand:
    """The follower's control, unless the policy brakes: then its brake, no throttle, the follower's steer."""
    if override is None or override.brake <= 0:
        return follow
    return ControlCommand(throttle=0.0, brake=override.brake, steer=follow.steer, reason=override.reason)
