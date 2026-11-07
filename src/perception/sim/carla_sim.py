"""CARLA as the closed loop's Simulator: one ego car with a camera on a straight lane, one scenario,
synchronous mode at 20 Hz.

The server must already be running in the scenario's town: switching maps at runtime crashes CARLA under CPU
rendering (lavapipe, Kaggle) and needs more RAM than the target laptop has, so it is refused here instead of
attempted. No background traffic: the scenario's actor is the only other road user, which keeps runs
reproducible and CPU rendering fast. The map's parked cars (a layer of the _Opt maps) stay unless the scenario
sets parked_vehicles: false: Town10HD's stand partly over the lines of the lanes beside the kerb, so with
them a test of braking for one obstacle also becomes a test of not braking for those.

The ego starts already at its target speed, and its normal driving is perception.sim.control.LaneFollower
(lane centre at ego_speed_kmh). The scenario ends at the first of: max_seconds, 1 s after a collision, 2 s
standing still after having driven, or the end of the lane.
"""

from __future__ import annotations

import dataclasses
import math
import queue

import carla
import numpy as np

from perception.core.types import CameraInfo, ControlCommand
from perception.sim.base import SimStep
from perception.sim.control import LaneFollower, apply_override
from perception.sim.metrics import ScenarioLog, Tick, forward_gap
from perception.sim.scenarios import build_scenario, lifted, path_xy, straight_lane

DT = 0.05  # 20 Hz, as the recordings
CAMERA_MOUNT = {"x": 1.5, "y": 0.0, "z": 1.6, "pitch": 0.0}  # as carla_scripts/record_sequence.py
STANDSTILL_MPS = 0.1


def to_bgr(image: carla.Image) -> np.ndarray:
    return np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, :3].copy()


class CarlaSimulator:
    """scenario: the scenario YAML as a dict:
        town, weather (carla.WeatherParameters preset), seed (which straight lane), ego_speed_kmh,
        road_length_m (straight lane needed), max_seconds, ego_blueprint, actor: {type, ...} (scenarios.py)
    camera: width, height, fov of the ego camera (CARLA's default recording camera)."""

    def __init__(
        self,
        scenario: dict,
        host: str = "localhost",
        port: int = 2000,
        timeout_s: float = 60.0,
        camera: dict | None = None,
    ) -> None:
        self.cfg = scenario
        self.timeout_s = timeout_s
        self.camera_cfg = {"width": 800, "height": 600, "fov": 90.0, **(camera or {})}
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)
        self.world = self.client.get_world()
        town = scenario.get("town", "Town10HD_Opt")
        if not self.world.get_map().name.endswith(town):
            raise RuntimeError(
                f"server runs {self.world.get_map().name}, scenario needs {town}: start CARLA in that map"
            )
        self.original_settings = self.world.get_settings()
        self.actors: list[carla.Actor] = []
        self.log = ScenarioLog()

    # --- setup -------------------------------------------------------------------------------------------

    def start(self) -> SimStep:
        cfg, world = self.cfg, self.world
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)
        world.set_weather(getattr(carla.WeatherParameters, cfg.get("weather", "ClearNoon")))
        # The layer stays as set on the server: set it either way, so the previous scenario doesn't decide
        if cfg.get("parked_vehicles", True):
            world.load_map_layer(carla.MapLayer.ParkedVehicles)
        else:
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        self.ego_speed = cfg.get("ego_speed_kmh", 30.0) / 3.6
        path = straight_lane(world.get_map(), cfg.get("road_length_m", 80.0), cfg.get("seed", 0))
        lib = world.get_blueprint_library()
        self.ego = world.spawn_actor(
            lib.find(cfg.get("ego_blueprint", "vehicle.tesla.model3")), lifted(path[0].transform)
        )
        self.actors.append(self.ego)
        self.scenario = build_scenario(world, path, self.ego_speed, cfg["actor"])
        self.actors += self.scenario.actors
        self.follower = LaneFollower(path_xy(path), self.ego_speed)

        c = self.camera_cfg
        bp = lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(c["width"]))
        bp.set_attribute("image_size_y", str(c["height"]))
        bp.set_attribute("fov", str(c["fov"]))
        mount = carla.Transform(
            carla.Location(x=CAMERA_MOUNT["x"], y=CAMERA_MOUNT["y"], z=CAMERA_MOUNT["z"]),
            carla.Rotation(pitch=CAMERA_MOUNT["pitch"]),
        )
        self.camera = world.spawn_actor(bp, mount, attach_to=self.ego)
        self.images: queue.Queue = queue.Queue()
        self.camera.listen(self.images.put)
        self.collision = world.spawn_actor(
            lib.find("sensor.other.collision"), carla.Transform(), attach_to=self.ego
        )
        self.collision.listen(self._on_collision)
        self.actors += [self.camera, self.collision]
        self.camera_info = dataclasses.replace(
            CameraInfo.from_fov(c["width"], c["height"], c["fov"], CAMERA_MOUNT["z"]),
            pitch_rad=math.radians(CAMERA_MOUNT["pitch"]),
        )

        bbox = self.ego.bounding_box
        self.ego_front_m = bbox.location.x + bbox.extent.x
        self.ego_half_width_m = bbox.extent.y
        self.collided = False

        # One tick so physics runs, then start everyone at speed (skips the slow part of CPU rendering)
        world.tick()
        self.ego.set_target_velocity(self.ego.get_transform().get_forward_vector() * self.ego_speed)
        self.scenario.start()
        world.tick()
        self.t0 = self._snapshot_time()
        self.t = 0.0
        self.standstill_s = 0.0
        self.end_s: float | None = None
        return self._frame()

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        if not self.collided:
            self.collided = True
            self.log.collision_with = event.other_actor.type_id

    def _snapshot_time(self) -> float:
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def _speed(self) -> float:
        v = self.ego.get_velocity()
        return math.hypot(v.x, v.y)

    def _frame(self) -> SimStep:
        """The camera image of the current tick; older ones (images arrive asynchronously) are skipped."""
        current = self.world.get_snapshot().frame
        image = self.images.get(timeout=self.timeout_s)
        while image.frame < current:
            image = self.images.get(timeout=self.timeout_s)
        return SimStep(
            image=to_bgr(image), timestamp=self.t, camera=self.camera_info, ego_speed=self._speed()
        )

    def _gap(self) -> tuple[float, bool]:
        target = self.scenario.target
        if target is None:
            return float("nan"), False
        corners = target.bounding_box.get_world_vertices(target.get_transform())
        corners = np.array([[c.x, c.y, c.z] for c in corners])
        world_to_ego = np.array(self.ego.get_transform().get_inverse_matrix())
        return forward_gap(world_to_ego, self.ego_front_m, self.ego_half_width_m, corners)

    # --- loop --------------------------------------------------------------------------------------------

    def step(self, override: ControlCommand | None) -> SimStep | None:
        tf, speed = self.ego.get_transform(), self._speed()
        follow = self.follower.control(tf.location.x, tf.location.y, tf.rotation.yaw, speed, DT)
        command = apply_override(follow, override)
        self.ego.apply_control(
            carla.VehicleControl(throttle=command.throttle, brake=command.brake, steer=command.steer)
        )
        gap, in_path = self._gap()
        self.scenario.update(self.t, gap)
        self.log.add(
            Tick(
                t=self.t,
                ego_speed=speed,
                gap_m=gap,
                in_path=in_path,
                throttle=command.throttle,
                brake=command.brake,
                steer=command.steer,
                reason=command.reason,
                collided=self.collided,
                answered=override is not None,
                camera_gap_m=gap + self.ego_front_m - CAMERA_MOUNT["x"],
            )
        )
        if self._finished(speed, command.brake):
            return None
        self.world.tick()
        self.t = self._snapshot_time() - self.t0
        return self._frame()

    def _finished(self, speed: float, brake: float) -> bool:
        if self.t >= self.cfg.get("max_seconds", 20.0) or self.follower.finished:
            return True
        if self.collided and self.end_s is None:
            self.end_s = self.t + 1.0
        # Only a held stop ends the run: once the brake is released the ego must be seen pulling away
        held = speed < STANDSTILL_MPS and brake > 0.0
        self.standstill_s = self.standstill_s + DT if held else 0.0
        return (self.end_s is not None and self.t >= self.end_s) or self.standstill_s >= 2.0

    def close(self) -> dict:
        try:
            for sensor in (getattr(self, "camera", None), getattr(self, "collision", None)):
                if sensor is not None:
                    sensor.stop()
            self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in self.actors])
            self.world.apply_settings(self.original_settings)
        except RuntimeError as exc:  # the server may be gone; the summary still matters
            print(f"cleanup failed: {exc}")
        return {"scenario": self.cfg, **self.log.summary()}

    def rows(self) -> list[dict]:
        return self.log.rows()
