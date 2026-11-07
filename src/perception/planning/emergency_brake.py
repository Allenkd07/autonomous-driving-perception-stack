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
    """Never intervenes: the baseline without the stack's braking.

    Accepts and ignores parameters, so `policy.type=noop` works as an override on a config whose policy
    section holds the emergency-brake parameters (the override keeps them)."""

    def __init__(self, **_ignored) -> None:
        pass

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

    The closing speed measured from monocular distances is noisy, worst far away: a few metres of jitter
    at 40 m reads as 20-30 m/s over a short window. An object ahead in our lane can only close at our own
    speed plus whatever it does on its own (a pedestrian walking at us, a car reversing), so the measured
    closing speed is capped at ego speed + max_object_approach_mps before TTC and stopping distance use it.
    That also keeps a stopped ego from braking for distance noise.

    Braking is held until nothing has triggered for hold_s, so one missed detection doesn't release it.
    Once stopped, the ego stays stopped while an in-lane object is within standstill_clearance_m: without
    this the route follower pulls away again, and the policy brakes hard a few metres later.
    """

    def __init__(
        self,
        ttc_threshold_s: float = 2.0,
        min_distance_m: float = 6.0,
        max_decel_mps2: float = 6.0,
        reaction_s: float = 0.3,
        hold_s: float = 1.0,
        max_object_approach_mps: float = 2.0,
        standstill_clearance_m: float = 15.0,
        standstill_speed_mps: float = 0.5,
    ) -> None:
        self.ttc_threshold_s = ttc_threshold_s
        self.min_distance_m = min_distance_m
        self.max_decel_mps2 = max_decel_mps2
        self.reaction_s = reaction_s
        self.hold_s = hold_s
        self.max_object_approach_mps = max_object_approach_mps
        self.standstill_clearance_m = standstill_clearance_m
        self.standstill_speed_mps = standstill_speed_mps
        self.reset()

    def reset(self) -> None:
        self._last_trigger: tuple[float, str] | None = None  # (timestamp, reason)

    @staticmethod
    def name(state: ObjectState) -> str:
        return f"track {state.track.track_id} ({state.track.class_name})"

    def trigger(self, state: ObjectState, ego_speed_mps: float) -> str | None:
        """Why this object needs braking, or None."""
        d, ttc = state.distance_m, state.ttc_s
        if not state.in_ego_lane or d is None:
            return None
        if d < self.min_distance_m:
            return f"EMERGENCY_BRAKE {self.name(state)} d={d:.1f}m"
        if ttc is None:
            return None
        closing = min(d / ttc, max(ego_speed_mps, 0.0) + self.max_object_approach_mps)
        ttc = d / closing
        if ttc < self.ttc_threshold_s:
            return f"EMERGENCY_BRAKE {self.name(state)} ttc={ttc:.1f}s"
        stopping = closing * self.reaction_s + closing**2 / (2 * self.max_decel_mps2)
        if d <= stopping:
            return f"EMERGENCY_BRAKE {self.name(state)} d={d:.1f}m < stopping {stopping:.1f}m"
        return None

    def decide(self, output: PerceptionOutput, ego_speed_mps: float) -> ControlCommand:
        now = output.frame.timestamp
        # The nearest triggering object names the reason
        nearest_first = sorted(
            output.objects, key=lambda s: s.distance_m if s.distance_m is not None else float("inf")
        )
        for state in nearest_first:
            reason = self.trigger(state, ego_speed_mps)
            if reason is not None:
                self._last_trigger = (now, reason)
                return ControlCommand(throttle=0.0, brake=1.0, steer=0.0, reason=reason)
        if self._last_trigger is not None and now - self._last_trigger[0] <= self.hold_s:
            return ControlCommand(
                throttle=0.0, brake=1.0, steer=0.0, reason=f"{self._last_trigger[1]} (held)"
            )
        if ego_speed_mps < self.standstill_speed_mps:
            for state in nearest_first:
                d = state.distance_m
                if state.in_ego_lane and d is not None and d < self.standstill_clearance_m:
                    reason = f"STANDSTILL {self.name(state)} d={d:.1f}m"
                    return ControlCommand(throttle=0.0, brake=1.0, steer=0.0, reason=reason)
        return NO_INTERVENTION
