import math
from types import SimpleNamespace

import carla
import numpy as np
import pytest

from perception.core.types import ControlCommand
from perception.sim import LaneFollower, ScenarioLog, apply_override, forward_gap
from perception.sim.metrics import Tick
from perception.sim.scenarios import straight_lane


def drive(follower: LaneFollower, x, y, yaw_deg, speed, seconds, dt=0.05, max_steer_deg=70.0, wheelbase=2.9):
    """Kinematic bicycle driven by the follower; returns the (x, y, yaw, speed) history."""
    history = []
    for _ in range(round(seconds / dt)):
        c = follower.control(x, y, yaw_deg, speed, dt)
        accel = 4.0 * c.throttle - 8.0 * c.brake - 0.3  # throttle, brake, rolling resistance (m/s^2)
        speed = max(0.0, speed + accel * dt)
        delta = c.steer * math.radians(max_steer_deg)
        yaw = math.radians(yaw_deg)
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        yaw_deg = math.degrees(yaw + speed / wheelbase * math.tan(delta) * dt)
        history.append((x, y, yaw_deg, speed))
    return np.array(history)


def test_follower_converges_to_the_lane_and_the_speed():
    path = np.c_[np.arange(0, 300, 2.0), np.full(150, 3.0)]  # straight lane along x, 3 m to the right
    follower = LaneFollower(path, target_speed=8.0)

    history = drive(follower, x=0.0, y=0.0, yaw_deg=0.0, speed=0.0, seconds=15)

    late = history[-60:]  # the last 3 s
    assert np.abs(late[:, 1] - 3.0).max() < 0.1  # on the lane centre
    assert np.abs(late[:, 3] - 8.0).max() < 0.3  # at the target speed
    assert np.all(np.diff(history[:, 0]) >= 0)  # never turned around


def test_follower_follows_a_curve_and_steers_right_for_a_right_turn():
    # quarter circle of radius 40 m turning right (CARLA: y to the right, yaw increasing clockwise)
    angles = np.linspace(0, math.pi / 2, 60)
    path = np.c_[40 * np.sin(angles), 40 * (1 - np.cos(angles))]
    follower = LaneFollower(path, target_speed=6.0)

    first = follower.control(0.0, 0.0, 0.0, 6.0, 0.05)
    history = drive(LaneFollower(path, target_speed=6.0), 0.0, 0.0, 0.0, 6.0, seconds=9)

    assert first.steer > 0
    radius = np.hypot(history[:, 0], history[:, 1] - 40.0)
    assert np.abs(radius - 40.0).max() < 1.0


def test_override_brakes_but_keeps_the_followers_steering():
    follow = ControlCommand(throttle=0.6, brake=0.0, steer=0.2)

    assert apply_override(follow, None) is follow
    assert apply_override(follow, ControlCommand(0.0, 0.0, 0.0, "")) is follow
    braked = apply_override(follow, ControlCommand(0.0, 1.0, 0.0, "EMERGENCY_BRAKE track 3"))
    assert (braked.throttle, braked.brake, braked.steer) == (0.0, 1.0, 0.2)
    assert braked.reason == "EMERGENCY_BRAKE track 3"


def box(x0, x1, y0, y1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, 1.5)])


def test_forward_gap_in_the_egos_frame():
    # ego at (10, 5) heading +y (yaw 90): world -> ego is a rotation by -90 and a shift
    yaw = math.radians(90)
    ego_to_world = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0, 10],
            [math.sin(yaw), math.cos(yaw), 0, 5],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    world_to_ego = np.linalg.inv(ego_to_world)

    ahead = box(9.0, 11.0, 25.0, 29.5)  # 20 m ahead of the ego origin, across its path
    beside = box(14.0, 16.0, 25.0, 29.5)  # same distance, 4-6 m to the left (x smaller = left here)

    gap, in_path = forward_gap(world_to_ego, ego_front_m=2.4, ego_half_width_m=1.0, corners=ahead)
    assert gap == pytest.approx(20.0 - 2.4) and in_path
    _, in_path = forward_gap(world_to_ego, 2.4, 1.0, beside)
    assert not in_path


def tick(t, speed, gap, brake=0.0, reason="", collided=False, answered=True):
    return Tick(t, speed, gap, True, 0.0, brake, 0.0, reason, collided, answered)


def test_scenario_summary_reports_the_first_intervention_and_the_stop():
    log = ScenarioLog()
    for i, (speed, gap) in enumerate(
        [(8.3, 30.0), (8.3, 20.0), (8.3, 17.0), (4.0, 12.0), (0.0, 9.5), (0.0, 9.5)]
    ):
        braking = i >= 2
        log.add(
            tick(
                i * 0.5,
                speed,
                gap,
                1.0 if braking else 0.0,
                "EMERGENCY_BRAKE track 1 ttc=1.9s" if braking else "",
            )
        )
    log.add(tick(3.0, 0.0, 9.5, answered=False))

    s = log.summary()

    assert not s["collision"] and s["braked"]
    assert s["first_brake"] == {
        "t": 1.0,
        "ego_speed_mps": 8.3,
        "gap_m": 17.0,
        "reason": "EMERGENCY_BRAKE track 1 ttc=1.9s",
    }
    assert s["stopped_gap_m"] == 9.5 and s["min_gap_in_path_m"] == 9.5
    assert s["unanswered_frames"] == 1


def test_scenario_summary_of_a_collision_without_braking():
    log = ScenarioLog()
    log.add(tick(0.0, 8.3, 5.0))
    log.add(tick(0.5, 8.3, -0.2, collided=True))
    log.collision_with = "vehicle.audi.tt"

    s = log.summary()

    assert s["collision"] and s["collision_with"] == "vehicle.audi.tt"
    assert not s["braked"] and "first_brake" not in s
    assert s["min_gap_in_path_m"] == -0.2


class FakeWaypoint:
    """Waypoint on a fake road given as (yaw_deg, is_junction) every 2 m."""

    lane_type = carla.LaneType.Driving

    def __init__(self, road: list, index: int) -> None:
        self.road, self.index = road, index
        self.is_junction = road[index][1]
        self.transform = SimpleNamespace(rotation=SimpleNamespace(yaw=road[index][0]))

    def next(self, distance_m: float) -> list:
        j = self.index + round(distance_m / 2.0)
        return [FakeWaypoint(self.road, j)] if j < len(self.road) else []


class FakeMap:
    name = "fake"

    def __init__(self, roads: list) -> None:
        self.roads = roads

    def generate_waypoints(self, distance_m: float) -> list:
        step = round(distance_m / 2.0)
        return [FakeWaypoint(road, i) for road in self.roads for i in range(0, len(road), step)]


def road(straight_m: float, then: tuple, yaw: float = 179.0) -> list:
    """straight_m of straight road (yaw near the +-180 wrap), then 20 m of `then` (yaw, is_junction)."""
    return [(yaw, False)] * round(straight_m / 2.0) + [then] * 10


@pytest.mark.parametrize("seed", range(5))
def test_straight_lane_finds_the_only_long_enough_stretch(seed):
    short = road(60, then=(179.0, True))  # 60 m, then a junction
    bend = road(70, then=(-170.0, False))  # 70 m, then 11 degrees off (across the +-180 wrap)
    long = road(100, then=(179.0, True))  # 100 m, then a junction
    path = straight_lane(FakeMap([short, bend, long]), 80.0, seed)
    assert len(path) == 40 and all(wp.road is long for wp in path)
    assert not any(wp.is_junction for wp in path)


def test_straight_lane_fails_when_no_stretch_is_long_enough():
    with pytest.raises(RuntimeError, match="no straight lane of 80"):
        straight_lane(FakeMap([road(60, then=(179.0, True)), road(70, then=(0.0, False))]), 80.0, 0)
