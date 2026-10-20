"""Phase 0 smoke test: connect to CARLA, spawn an ego vehicle with an RGB camera,
save a few frames, then clean up.

Usage (CARLA server must already be running, e.g. `CarlaUE4.exe -quality-level=Low`):
    python carla_scripts/test_connection.py --host localhost --frames 10
"""

import argparse
import queue
import time
from pathlib import Path

import carla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--out", default="_out/test_connection")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"Connected. Client {client.get_client_version()} / server {client.get_server_version()}")
    print(f"Map: {world.get_map().name}")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS sim
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    actors = []
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.width))
        cam_bp.set_attribute("image_size_y", str(args.height))
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))  # roughly dashcam position
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        actors.append(camera)

        frames = queue.Queue()
        camera.listen(frames.put)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        for _ in range(args.frames):
            world.tick()
            image = frames.get(timeout=10.0)
            image.save_to_disk(str(out_dir / f"{image.frame:06d}.png"))
        elapsed = time.perf_counter() - start
        print(
            f"Saved {args.frames} frames to {out_dir.resolve()} ({args.frames / elapsed:.1f} FPS incl. I/O)"
        )
    finally:
        for actor in actors:
            if isinstance(actor, carla.Sensor):
                actor.stop()
            actor.destroy()
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        print("Cleaned up.")


if __name__ == "__main__":
    main()
