"""Record a CARLA sequence for offline evaluation (tracking, detection, depth, distance).

One ego camera records RGB, instance segmentation and depth at the same pose, plus every vehicle's and
pedestrian's 3D box and velocity per frame. 2D ground truth is derived offline by
perception.io.carla_recording, which also defines the layout written here.

Run from WSL against the CARLA server on Windows (`CarlaUE4.exe -quality-level=Low`):
    python carla_scripts/record_sequence.py --out data/carla/town10_noon_s0
    python carla_scripts/record_sequence.py --town Town03 --weather WetCloudySunset --seed 1 \\
        --vehicles 60 --walkers 80 --frames 1200 --out data/carla/town03_wet_s1
"""

from __future__ import annotations

import argparse
import json
import queue
import random
from pathlib import Path

import carla
import cv2
import numpy as np
import pandas as pd

from perception.io.carla_recording import BASE_TYPE_CLASS, DEPTH_DIR, FRAME_NAME, INSTANCE_DIR, RGB_DIR

CAMERA_MOUNT = {"x": 1.5, "y": 0.0, "z": 1.6, "pitch": 0.0}  # metres / degrees from the ego origin


def class_of(actor: carla.Actor) -> str | None:
    if actor.type_id.startswith("walker.pedestrian"):
        return "person"
    if actor.type_id.startswith("vehicle."):
        base_type = actor.attributes.get("base_type", "").lower()
        if base_type in BASE_TYPE_CLASS:
            return BASE_TYPE_CLASS[base_type]
        return "motorcycle" if actor.attributes.get("number_of_wheels") == "2" else "car"
    return None


def to_bgr(image: carla.Image) -> np.ndarray:
    return np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, :3].copy()


def spawn_vehicles(
    world, tm_port: int, n: int, spawn_points: list, rng: random.Random, two_wheeler_share: float
) -> list:
    """n autopilot vehicles; each is a bicycle/motorcycle with probability two_wheeler_share."""
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    two_wheelers = [bp for bp in blueprints if bp.get_attribute("number_of_wheels").as_int() == 2]
    four_wheelers = [bp for bp in blueprints if bp.get_attribute("number_of_wheels").as_int() != 2]
    vehicles = []
    for transform in spawn_points:
        if len(vehicles) == n:
            break
        bp = rng.choice(two_wheelers if rng.random() < two_wheeler_share else four_wheelers)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        vehicle = world.try_spawn_actor(bp, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True, tm_port)
            vehicles.append(vehicle)
    return vehicles


def spawn_walkers(
    world, n: int, rng: random.Random, centre: carla.Location | None = None, radius: float = 0.0
) -> tuple[list, list]:
    """n walkers with AI controllers; with a radius, only on sidewalks within radius metres of centre."""
    lib = world.get_blueprint_library()
    blueprints = lib.filter("walker.pedestrian.*")
    walkers = []
    # Some random locations collide, so try more often than n; far more with a radius, since most random
    # locations of a large map fall outside it (the calls are cheap).
    for _ in range(n * (50 if radius > 0 else 2)):
        if len(walkers) == n:
            break
        location = world.get_random_location_from_navigation()
        if location is None or (radius > 0 and location.distance(centre) > radius):
            continue
        bp = rng.choice(blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(bp, carla.Transform(location + carla.Location(z=1.0)))
        if walker is not None:
            walkers.append(walker)
    controller_bp = lib.find("controller.ai.walker")
    controllers = [world.spawn_actor(controller_bp, carla.Transform(), attach_to=w) for w in walkers]
    world.tick()  # controllers need one tick before they can start
    for controller in controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(rng.uniform(1.0, 1.8))
    return walkers, controllers


def spawn_cameras(world, ego, args) -> dict:
    lib = world.get_blueprint_library()
    mount = carla.Transform(
        carla.Location(x=CAMERA_MOUNT["x"], y=CAMERA_MOUNT["y"], z=CAMERA_MOUNT["z"]),
        carla.Rotation(pitch=CAMERA_MOUNT["pitch"]),
    )
    cameras = {}
    for name, blueprint in (
        (RGB_DIR, "sensor.camera.rgb"),
        (INSTANCE_DIR, "sensor.camera.instance_segmentation"),
        (DEPTH_DIR, "sensor.camera.depth"),
    ):
        bp = lib.find(blueprint)
        bp.set_attribute("image_size_x", str(args.width))
        bp.set_attribute("image_size_y", str(args.height))
        bp.set_attribute("fov", str(args.fov))
        cameras[name] = world.spawn_actor(bp, mount, attach_to=ego)
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--weather", default="ClearNoon", help="a carla.WeatherParameters preset")
    parser.add_argument("--vehicles", type=int, default=40)
    parser.add_argument("--walkers", type=int, default=60)
    parser.add_argument("--frames", type=int, default=600, help="recorded frames (20 per simulated second)")
    parser.add_argument(
        "--warmup", type=int, default=40, help="ticks before recording, to get traffic moving"
    )
    parser.add_argument(
        "--spawn-radius",
        type=float,
        default=0.0,
        help="spawn vehicles and walkers within this many metres of the ego (0: anywhere); keeps traffic in "
        "view on large maps like Town05",
    )
    parser.add_argument(
        "--two-wheeler-share",
        type=float,
        default=0.3,
        help="fraction of spawned vehicles that are bicycles/motorcycles (uniform blueprint choice gives "
        "~15%%, and they rarely end up in view)",
    )
    parser.add_argument(
        "--ego-ignore-lights",
        type=float,
        default=100.0,
        help="percentage of red lights the ego drives through, so it keeps moving instead of waiting "
        "half the sequence at a light; collision avoidance stays on",
    )
    parser.add_argument("--max-range", type=float, default=100.0, help="actors further away are not stored")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds to wait for the server and for each frame; raise it for CPU rendering (lavapipe)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    world.set_pedestrians_seed(args.seed)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    rng = random.Random(args.seed)

    for folder in (RGB_DIR, INSTANCE_DIR, DEPTH_DIR):
        (args.out / folder).mkdir(parents=True, exist_ok=False)

    # Stored before anything can fail, so a partial recording can still be saved if the server dies.
    meta = {
        "map": world.get_map().name,
        "weather": args.weather,
        "seed": args.seed,
        "fps": round(1 / settings.fixed_delta_seconds),
        "camera": {"width": args.width, "height": args.height, "fov": args.fov, "mount": CAMERA_MOUNT},
        "max_range_m": args.max_range,
        "spawn_radius_m": args.spawn_radius,
        "two_wheeler_share": args.two_wheeler_share,
        "ego_ignore_lights_pct": args.ego_ignore_lights,
        "carla_version": client.get_server_version(),
    }
    vehicles, walkers, controllers, cameras = [], [], [], {}
    frame_rows, actor_rows, error = [], [], None
    try:
        spawn_points = world.get_map().get_spawn_points()
        rng.shuffle(spawn_points)
        ego = world.spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"), spawn_points[0])
        vehicles = [ego]
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, args.ego_ignore_lights)
        others = spawn_points[1:]
        if args.spawn_radius > 0:
            centre = spawn_points[0].location
            others = [p for p in others if p.location.distance(centre) <= args.spawn_radius]
        vehicles += spawn_vehicles(world, tm.get_port(), args.vehicles, others, rng, args.two_wheeler_share)
        walkers, controllers = spawn_walkers(
            world, args.walkers, rng, spawn_points[0].location, args.spawn_radius
        )
        cameras = spawn_cameras(world, ego, args)
        queues = {name: queue.Queue() for name in cameras}
        for name, camera in cameras.items():
            camera.listen(queues[name].put)
        tracked = {a.id: a for a in [*vehicles[1:], *walkers] if class_of(a) is not None}
        n_two = sum(class_of(v) in ("bicycle", "motorcycle") for v in vehicles[1:])
        print(
            f"Map {world.get_map().name}: {len(vehicles) - 1} vehicles ({n_two} two-wheelers), "
            f"{len(walkers)} walkers"
        )

        meta.update(vehicles=len(vehicles) - 1, walkers=len(walkers))
        for tick in range(args.warmup + args.frames):
            world.tick()
            images = {name: q.get(timeout=args.timeout) for name, q in queues.items()}
            if len({image.frame for image in images.values()}) != 1:
                raise RuntimeError(f"sensors out of sync: {[image.frame for image in images.values()]}")
            if tick < args.warmup:
                continue

            index = tick - args.warmup
            name = FRAME_NAME.format(index)
            jpeg = [cv2.IMWRITE_JPEG_QUALITY, 95]
            cv2.imwrite(str(args.out / RGB_DIR / f"{name}.jpg"), to_bgr(images[RGB_DIR]), jpeg)
            cv2.imwrite(str(args.out / INSTANCE_DIR / f"{name}.png"), to_bgr(images[INSTANCE_DIR]))
            cv2.imwrite(str(args.out / DEPTH_DIR / f"{name}.png"), to_bgr(images[DEPTH_DIR]))

            snapshot = world.get_snapshot()
            rgb = images[RGB_DIR]
            ego_velocity = snapshot.find(ego.id).get_velocity()
            frame_rows.append(
                {
                    "index": index,
                    "carla_frame": rgb.frame,
                    "timestamp": rgb.timestamp,
                    "cam_to_world": np.array(rgb.transform.get_matrix(), np.float64).ravel(),
                    "ego_velocity": np.array([ego_velocity.x, ego_velocity.y, ego_velocity.z], np.float32),
                }
            )
            camera_location = rgb.transform.location
            for actor_id, actor in tracked.items():
                state = snapshot.find(actor_id)
                if state is None:
                    continue
                transform = state.get_transform()
                if transform.location.distance(camera_location) > args.max_range:
                    continue
                corners = actor.bounding_box.get_world_vertices(transform)
                velocity = state.get_velocity()
                actor_rows.append(
                    {
                        "index": index,
                        "actor_id": actor_id,
                        "blueprint": actor.type_id,
                        "class_name": class_of(actor),
                        "corners": np.array([[c.x, c.y, c.z] for c in corners], np.float64).ravel(),
                        "velocity": np.array([velocity.x, velocity.y, velocity.z], np.float32),
                    }
                )
            if index % 100 == 0:
                print(f"frame {index}/{args.frames}")

    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        client.set_timeout(30.0)  # the server may be gone: don't wait args.timeout on each cleanup call
        raise
    finally:
        if frame_rows:
            save(args.out, frame_rows, actor_rows, {**meta, "frames": len(frame_rows), "error": error})
        cleanup(client, world, tm, original_settings, cameras, controllers, walkers, vehicles)


def save(out: Path, frame_rows: list, actor_rows: list, meta: dict) -> None:
    """Write the recorded frames; a recording cut short by an error is kept, with "complete": false."""
    pd.DataFrame(frame_rows).to_parquet(out / "frames.parquet", index=False)
    pd.DataFrame(actor_rows).to_parquet(out / "actors.parquet", index=False)
    meta = {**meta, "complete": meta["error"] is None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    status = "" if meta["complete"] else f" (INCOMPLETE: {meta['error']})"
    print(f"wrote {len(frame_rows)} frames, {len(actor_rows)} actor rows to {out}{status}")


def cleanup(client, world, tm, original_settings, cameras, controllers, walkers, vehicles) -> None:
    """Same order as test_connection.py: release actors from the Traffic Manager and stop sensors and walker
    controllers, tick, then destroy, or the TM thread aborts on deleted actors. Best effort: after a server
    crash every call fails, which must not hide the original error."""
    try:
        for camera in cameras.values():
            camera.stop()
        for controller in controllers:
            controller.stop()
        for vehicle in vehicles:
            vehicle.set_autopilot(False, tm.get_port())
        world.tick()
        client.apply_batch_sync(
            [carla.command.DestroyActor(a) for a in [*cameras.values(), *controllers, *walkers, *vehicles]]
        )
        world.tick()
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        print("Cleaned up.")
    except RuntimeError as exc:
        print(f"Cleanup failed (server gone?): {exc}")


if __name__ == "__main__":
    main()
