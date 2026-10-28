"""Decision policies on top of a route follower (e.g. CARLA's BasicAgent in the decision node).

A policy only overrides: a command with brake > 0 means "brake now, for this reason"; brake 0 with an empty
reason means no intervention, and the caller keeps its normal control.
"""

from __future__ import annotations

from perception.core.registry import register
from perception.core.types import ControlCommand, ObjectState, PerceptionOutput

NO_INTERVENTION = ControlCommand(throttle=0.0, brake=0.0, steer=0.0, reason="")


@register("policy", "noop")
class NoOpPolicy:
    def decide(self, output: PerceptionOutput, ego_speed_mps: float) -> ControlCommand:
        return NO_INTERVENTION

    def reset(self) -> None:
        pass


@register("policy", "emergency_brake")
class EmergencyBrakePolicy:
    """Full braking for an object in the ego lane that we are about to hit.

    An in-lane object with a known distance triggers braking when any of these holds:
    - its time to collision is below ttc_threshold_s
    - we could no longer stop in time: distance <= v * reaction_s + v^2 / (2 * max_decel_mps2), with v its
      closing speed (distance / TTC); at high closing speeds this fires before the TTC threshold does
    - it is closer than min_distance_m, whatever its TTC (e.g. a pedestrian that just appeared, with no
      speed estimate yet)
    An object we are not closing on never triggers by speed, so following a car is not braking for it.
    Braking is held until nothing has triggered for hold_s, so one missed detection doesn't release it.
    """

    def __init__(
        self,
        ttc_threshold_s: float = 2.0,
        min_distance_m: float = 6.0,
        max_decel_mps2: float = 6.0,
        reaction_s: float = 0.3,
        hold_s: float = 1.0,
    ) -> None:
        self.ttc_threshold_s = ttc_threshold_s
        self.min_distance_m = min_distance_m
        self.max_decel_mps2 = max_decel_mps2
        self.reaction_s = reaction_s
        self.hold_s = hold_s
        self.reset()

    def reset(self) -> None:
        self._last_trigger: tuple[float, str] | None = None  # (timestamp, reason)

    def trigger(self, state: ObjectState) -> str | None:
        """Why this object needs braking, or None."""
        d, ttc = state.distance_m, state.ttc_s
        if not state.in_ego_lane or d is None:
            return None
        name = f"track {state.track.track_id} ({state.track.class_name})"
        if d < self.min_distance_m:
            return f"EMERGENCY_BRAKE {name} d={d:.1f}m"
        if ttc is None:
            return None
        if ttc < self.ttc_threshold_s:
            return f"EMERGENCY_BRAKE {name} ttc={ttc:.1f}s"
        closing = d / ttc
        stopping = closing * self.reaction_s + closing**2 / (2 * self.max_decel_mps2)
        if d <= stopping:
            return f"EMERGENCY_BRAKE {name} d={d:.1f}m < stopping {stopping:.1f}m"
        return None

    def decide(self, output: PerceptionOutput, ego_speed_mps: float) -> ControlCommand:
        now = output.frame.timestamp
        # The nearest triggering object names the reason
        for state in sorted(output.objects, key=lambda s: s.distance_m if s.distance_m is not None else 1e9):
            reason = self.trigger(state)
            if reason is not None:
                self._last_trigger = (now, reason)
                return ControlCommand(throttle=0.0, brake=1.0, steer=0.0, reason=reason)
        if self._last_trigger is not None and now - self._last_trigger[0] <= self.hold_s:
            return ControlCommand(
                throttle=0.0, brake=1.0, steer=0.0, reason=f"{self._last_trigger[1]} (held)"
            )
        return NO_INTERVENTION
