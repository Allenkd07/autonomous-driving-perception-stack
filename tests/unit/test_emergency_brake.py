import dataclasses

import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import DecisionPolicy
from perception.core.types import ObjectState, PerceptionOutput, Track
from perception.planning.emergency_brake import NO_INTERVENTION, EmergencyBrakePolicy, NoOpPolicy


def state(distance, ttc=None, in_lane=True, track_id=1, name="person") -> ObjectState:
    track = Track(box=np.zeros(4, np.float32), score=0.9, class_id=3, class_name=name, track_id=track_id)
    return ObjectState(track=track, distance_m=distance, ttc_s=ttc, in_ego_lane=in_lane)


def output(make_frame, *states, t=0.0) -> PerceptionOutput:
    frame = make_frame()
    frame = dataclasses.replace(frame, timestamp=t)
    return PerceptionOutput(frame=frame, objects=list(states))


@pytest.fixture
def policy():
    return EmergencyBrakePolicy(ttc_threshold_s=2.0, min_distance_m=6.0, max_decel_mps2=6.0, reaction_s=0.3)


def test_brakes_for_low_ttc_in_lane(policy, make_frame):
    cmd = policy.decide(output(make_frame, state(20.0, ttc=1.5, track_id=12)), ego_speed_mps=10.0)

    assert cmd.brake == 1.0 and cmd.throttle == 0.0
    assert "track 12" in cmd.reason and "ttc=1.5s" in cmd.reason
    assert isinstance(policy, DecisionPolicy)


def test_ignores_objects_outside_the_lane_or_without_distance(policy, make_frame):
    cmd = policy.decide(output(make_frame, state(5.0, ttc=0.5, in_lane=False), state(None)), 10.0)

    assert cmd == NO_INTERVENTION


def test_brakes_when_too_close_even_without_ttc(policy, make_frame):
    assert policy.decide(output(make_frame, state(4.0)), 0.0).brake == 1.0


def test_stopping_distance_fires_before_the_ttc_threshold_at_speed(policy, make_frame):
    # Closing at 25 m/s from 60 m: TTC 2.4 s (above 2 s), but stopping takes 25*0.3 + 625/12 = 59.6 m
    assert policy.decide(output(make_frame, state(59.0, ttc=59.0 / 25)), 25.0).brake == 1.0
    policy.reset()
    # Same TTC at a low closing speed: 12 m at 5 m/s needs only 1.5 + 2.1 m
    assert policy.decide(output(make_frame, state(12.0, ttc=2.4)), 5.0) == NO_INTERVENTION


def test_a_car_we_are_not_closing_on_is_not_braked_for(policy, make_frame):
    assert policy.decide(output(make_frame, state(10.0, ttc=None, name="car")), 15.0) == NO_INTERVENTION


def test_reason_names_the_nearest_triggering_object(policy, make_frame):
    cmd = policy.decide(
        output(make_frame, state(30.0, ttc=1.0, track_id=1), state(15.0, ttc=1.8, track_id=2)), 10.0
    )

    assert "track 2" in cmd.reason


def test_braking_is_held_through_a_missed_detection(policy, make_frame):
    policy.decide(output(make_frame, state(10.0, ttc=1.0), t=0.0), 10.0)

    held = policy.decide(output(make_frame, t=0.5), 10.0)
    released = policy.decide(output(make_frame, t=1.6), 10.0)

    assert held.brake == 1.0 and held.reason.endswith("(held)")
    assert released == NO_INTERVENTION


def test_noop_never_intervenes(make_frame):
    assert NoOpPolicy().decide(output(make_frame, state(1.0, ttc=0.1)), 10.0) == NO_INTERVENTION


def test_registered():
    assert registry.get("policy", "emergency_brake") is EmergencyBrakePolicy
    assert registry.get("policy", "noop") is NoOpPolicy
