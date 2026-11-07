"""Scenario actors for the closed loop: what the ego meets on a straight stretch of road.

Each scenario places its actor on the ego's lane path (a list of carla.Waypoint 2 m apart, the same path
the ego follows), updates it every tick, and names its target: the actor whose true gap to the ego is
measured. Configured from configs/scenarios/*.yaml (`type` + the class's parameters).

    stopped_car          a parked car in the ego lane, distance_m ahead
    pedestrian_crossing  a pedestrian at the kerb walks across the ego lane once the ego is trigger_gap_m away
    lead_brake           a car ahead drives at the ego's speed, then brakes hard at brake_at_s
"""

from __future__ import annotations

import math

import carla
import numpy as np

from perception.sim.control import LaneFollower

WAYPOINT_STEP_M = 2.0


def path_xy(path: list) -> np.ndarray:
    return np.array([[w.transform.location.x, w.transform.location.y] for w in path], np.float64)


def lifted(transform: carla.Transform, z: float = 0.5) -> carla.Transform:
    """A spawn transform slightly above the road, so the actor doesn't collide with it on spawn."""
    loc = transform.location
    return carla.Transform(carla.Location(loc.x, loc.y, loc.z + z), transform.rotation)


class Scenario:
    target: carla.Actor | None = None

    def __init__(self, world: carla.World, path: list, ego_speed: float) -> None:
        self.world = world
        self.path = path
        self.ego_speed = ego_speed
        self.actors: list[carla.Actor] = []

    def waypoint_at(self, distance_m: float):
        return self.path[min(round(distance_m / WAYPOINT_STEP_M), len(self.path) - 1)]

    def spawn(self, blueprint: str, transform: carla.Transform) -> carla.Actor:
        actor = self.world.spawn_actor(self.world.get_blueprint_library().find(blueprint), transform)
        self.actors.append(actor)
        return actor

    def start(self) -> None:
        """Called after the first tick, once physics runs (initial velocities)."""

    def update(self, t: float, gap_m: float) -> None:
        """Every tick before it runs: t = simulated seconds, gap_m = the ego's true gap to the target."""


class StoppedCar(Scenario):
    def __init__(self, world, path, ego_speed, distance_m: float = 50.0, blueprint: str = "vehicle.audi.tt"):
        super().__init__(world, path, ego_speed)
        self.target = self.spawn(blueprint, lifted(self.waypoint_at(distance_m).transform))

    def start(self) -> None:
        self.target.apply_control(carla.VehicleControl(hand_brake=True))


class PedestrianCrossing(Scenario):
    """side_offset_m: start this far right of the lane centre (on the kerb); walks left across the lane."""

    def __init__(
        self,
        world,
        path,
        ego_speed,
        distance_m: float = 45.0,
        trigger_gap_m: float = 35.0,
        side_offset_m: float = 4.5,
        walk_speed: float = 1.4,
        blueprint: str = "walker.pedestrian.0001",
    ):
        super().__init__(world, path, ego_speed)
        wp = self.waypoint_at(distance_m).transform
        right = wp.get_right_vector()
        start = wp.location + carla.Location(right.x * side_offset_m, right.y * side_offset_m, 1.0)
        self.direction = carla.Vector3D(-right.x, -right.y, 0.0)
        self.trigger_gap_m = trigger_gap_m
        self.walk_speed = walk_speed
        self.end_offset_m = side_offset_m  # stops as far left of the centre as it started right of it
        self.lane_centre, self.right = wp.location, right
        yaw = math.degrees(math.atan2(-right.y, -right.x))
        self.target = self.spawn(blueprint, carla.Transform(start, carla.Rotation(yaw=yaw)))
        self.walking = False

    def update(self, t: float, gap_m: float) -> None:
        if not self.walking and gap_m <= self.trigger_gap_m:
            self.walking = True
        if not self.walking:
            return
        loc = self.target.get_location() - self.lane_centre
        offset = loc.x * self.right.x + loc.y * self.right.y  # metres right of the lane centre
        speed = self.walk_speed if offset > -self.end_offset_m else 0.0
        self.target.apply_control(carla.WalkerControl(direction=self.direction, speed=speed))


class LeadBrake(Scenario):
    def __init__(
        self,
        world,
        path,
        ego_speed,
        distance_m: float = 25.0,
        speed_kmh: float | None = None,
        brake_at_s: float = 5.0,
        blueprint: str = "vehicle.lincoln.mkz_2020",
    ):
        super().__init__(world, path, ego_speed)
        start = round(distance_m / WAYPOINT_STEP_M)
        self.speed = ego_speed if speed_kmh is None else speed_kmh / 3.6
        self.target = self.spawn(blueprint, lifted(self.path[start].transform))
        self.follower = LaneFollower(path_xy(self.path[start:]), self.speed)
        self.brake_at_s = brake_at_s

    def start(self) -> None:
        forward = self.target.get_transform().get_forward_vector()
        self.target.set_target_velocity(forward * self.speed)

    def update(self, t: float, gap_m: float) -> None:
        if t >= self.brake_at_s:
            self.target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return
        tf, v = self.target.get_transform(), self.target.get_velocity()
        speed = math.hypot(v.x, v.y)
        c = self.follower.control(tf.location.x, tf.location.y, tf.rotation.yaw, speed, 0.05)
        self.target.apply_control(carla.VehicleControl(throttle=c.throttle, brake=c.brake, steer=c.steer))


SCENARIOS = {"stopped_car": StoppedCar, "pedestrian_crossing": PedestrianCrossing, "lead_brake": LeadBrake}


def build_scenario(world, path, ego_speed, cfg: dict) -> Scenario:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in SCENARIOS:
        raise ValueError(f"unknown scenario {kind!r}; expected one of {sorted(SCENARIOS)}")
    return SCENARIOS[kind](world, path, ego_speed, **cfg)


def straight_lane(carla_map: carla.Map, length_m: float, seed: int, start_every_m: float = 10.0) -> list:
    """Waypoints 2 m apart along a straight lane with no junction for length_m. A straight road keeps the
    scenario about braking, not steering, and no junction keeps the lane markings visible.

    Starts are tried in a seeded random order among waypoints every start_every_m on all driving lanes, not
    only spawn points: those rarely begin a long enough stretch (Town10HD's longest is ~98 m)."""
    rng = np.random.default_rng(seed)
    starts = [
        w
        for w in carla_map.generate_waypoints(start_every_m)
        if w.lane_type == carla.LaneType.Driving and not w.is_junction
    ]
    for i in rng.permutation(len(starts)):
        wp = starts[i]
        path = [wp]
        while len(path) * WAYPOINT_STEP_M < length_m:
            nxt = path[-1].next(WAYPOINT_STEP_M)
            if not nxt or nxt[0].is_junction:
                break
            yaw_change = (nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180
            if abs(yaw_change) > 5.0:
                break
            path.append(nxt[0])
        if len(path) * WAYPOINT_STEP_M >= length_m:
            return path
    raise RuntimeError(f"no straight lane of {length_m} m in {carla_map.name}")
