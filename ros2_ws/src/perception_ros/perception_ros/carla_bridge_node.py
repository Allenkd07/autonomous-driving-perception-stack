"""carla_bridge_node: runs one CARLA scenario in lockstep with the perception stack (the closed loop).

Every simulator tick: publish the camera frame (and the ego speed), wait for the decision node's control
command for that frame (matched by header stamp), apply it on top of the ego's lane following, tick. CARLA
runs in synchronous mode, so the simulated world waits for the stack: a slow pipeline (or CPU rendering)
costs wall-clock time, never correctness, and every run of a scenario is the same.

Publishes   image (sensor_msgs/Image, bgr8), camera_info (sensor_msgs/CameraInfo),
            ego_speed (std_msgs/Float32)
Subscribes  control (perception_msgs/ControlCommand), debug_image (sensor_msgs/Image; saved as frames),
            objects (perception_msgs/TrackedObjectArray; the nearest in-lane object goes into ticks.csv)
Parameters  scenario (configs/scenarios/*.yaml), host, port, carla_timeout_s, answer_timeout_s (per frame),
            first_answer_timeout_s (the first frame also waits for the models to load and warm up),
            graph_timeout_s (for the other nodes to appear),
            out_dir (summary.json, ticks.csv, frames/ with the overlays, camera/ with the camera's frames,
            objects.jsonl with every frame's objects as the stack saw them)

A frame not answered in time is driven without intervention and counted (unanswered_frames in the summary).
ticks.csv joins the simulator's ground truth with what the stack saw in each frame (est_* columns: the
nearest object it placed in the ego lane), so distance errors and the reason for every brake can be checked.
camera/ and objects.jsonl let a run be replayed through the stack offline (e.g. after changing it), along
the trajectory that was driven.
The node exits when the scenario ends; the launch file then shuts the other nodes down.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import cv2
import rclpy
from omegaconf import OmegaConf
from perception_msgs.msg import ControlCommand as ControlCommandMsg
from perception_msgs.msg import TrackedObjectArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header

from perception.sim.base import Simulator
from perception_ros.conversions import (
    bgr_to_image,
    camera_to_msg,
    command_from_msg,
    image_to_bgr,
    objects_from_msg,
    seconds_to_stamp,
    stamp_to_seconds,
)

# The stack waits for every frame, so frames must not be dropped: reliable, unlike a free-running camera
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
LATEST = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
TICK_S = 0.05
NO_ESTIMATE = {
    "est_track": None,
    "est_class": None,
    "est_distance_m": None,
    "est_ttc_s": None,
    "est_objects": 0,
}


def object_row(s) -> dict:
    """One object of objects.jsonl."""
    return {
        "track": s.track.track_id,
        "class": s.track.class_name,
        "score": round(float(s.track.score), 3),
        "box": [round(float(c), 1) for c in s.track.box],
        "distance_m": None if s.distance_m is None else round(s.distance_m, 2),
        "ttc_s": None if s.ttc_s is None else round(s.ttc_s, 2),
        "in_lane": bool(s.in_ego_lane),
    }


def nearest_in_lane(objects) -> dict:
    """The est_* columns of ticks.csv: the nearest object the stack placed in the ego lane (with a distance),
    and how many objects it tracked in total."""
    in_lane = [s for s in objects if s.in_ego_lane and s.distance_m is not None]
    if not in_lane:
        return {**NO_ESTIMATE, "est_objects": len(objects)}
    s = min(in_lane, key=lambda s: s.distance_m)
    return {
        "est_track": s.track.track_id,
        "est_class": s.track.class_name,
        "est_distance_m": round(s.distance_m, 2),
        "est_ttc_s": None if s.ttc_s is None else round(s.ttc_s, 2),
        "est_objects": len(objects),
    }


class CarlaBridgeNode(Node):
    def __init__(self, simulator: Simulator | None = None, **kwargs) -> None:
        super().__init__("carla_bridge_node", **kwargs)
        self.declare_parameter("scenario", "")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("carla_timeout_s", 60.0)
        self.declare_parameter("answer_timeout_s", 30.0)
        self.declare_parameter("first_answer_timeout_s", 600.0)
        self.declare_parameter("graph_timeout_s", 600.0)
        self.declare_parameter("out_dir", "")

        self.scenario_cfg: dict = {}
        if simulator is None:
            path = self.get_parameter("scenario").value
            if not path:
                raise ValueError("parameter 'scenario' (configs/scenarios/*.yaml) is required")
            self.scenario_cfg = OmegaConf.to_container(OmegaConf.load(path))
            from perception.sim.carla_sim import CarlaSimulator  # needs the CARLA client

            simulator = CarlaSimulator(
                self.scenario_cfg,
                host=self.get_parameter("host").value,
                port=self.get_parameter("port").value,
                timeout_s=self.get_parameter("carla_timeout_s").value,
            )
        self.sim = simulator
        out = self.get_parameter("out_dir").value
        self.out_dir = Path(out) if out else None
        if self.out_dir is not None:
            (self.out_dir / "frames").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "camera").mkdir(exist_ok=True)

        self.image_pub = self.create_publisher(Image, "image", RELIABLE)
        self.info_pub = self.create_publisher(CameraInfoMsg, "camera_info", RELIABLE)
        self.speed_pub = self.create_publisher(Float32, "ego_speed", RELIABLE)
        self.control_sub = self.create_subscription(ControlCommandMsg, "control", self.on_control, RELIABLE)
        self.create_subscription(Image, "debug_image", self.on_debug_image, LATEST)
        self.create_subscription(TrackedObjectArray, "objects", self.on_objects, RELIABLE)

        self.lock = threading.Lock()
        self.answered = threading.Event()
        self.expected = None  # stamp of the frame waiting for its command
        self.answer = None
        self.seen = threading.Event()  # the objects of the expected frame arrived
        self.estimate = NO_ESTIMATE
        self.estimates: dict[int, dict] = {}  # tick index -> est_* columns
        self.objects: list = []  # the expected frame's objects
        self.object_log: list[dict] = []  # objects.jsonl
        self.saved_frames = 0

    def on_control(self, msg: ControlCommandMsg) -> None:
        with self.lock:
            if self.expected is not None and msg.header.stamp == self.expected:
                self.answer = command_from_msg(msg)
                self.answered.set()

    def on_objects(self, msg: TrackedObjectArray) -> None:
        with self.lock:
            if self.expected is not None and msg.header.stamp == self.expected:
                self.objects = objects_from_msg(msg)
                self.estimate = nearest_in_lane(self.objects)
                self.seen.set()

    def on_debug_image(self, msg: Image) -> None:
        if self.out_dir is None:
            return
        index = round(stamp_to_seconds(msg.header.stamp) / TICK_S)
        cv2.imwrite(str(self.out_dir / "frames" / f"{index:06d}.jpg"), image_to_bgr(msg))
        self.saved_frames += 1

    def wait_for_graph(self) -> bool:
        """Wait until perception subscribes to the camera and decision publishes control: messages sent
        before discovery would be lost, and the first frame would wait in vain."""
        deadline = time.monotonic() + self.get_parameter("graph_timeout_s").value
        while time.monotonic() < deadline:
            # topic_name has the launch file's remapping applied; count_publishers("control")
            # would look at /control, not /perception/control, and wait in vain
            if self.image_pub.get_subscription_count() and self.count_publishers(self.control_sub.topic_name):
                return True
            time.sleep(0.2)
        return False

    def publish(self, step) -> Header:
        header = Header(stamp=seconds_to_stamp(step.timestamp), frame_id="camera")
        with self.lock:
            self.expected, self.answer, self.estimate, self.objects = header.stamp, None, NO_ESTIMATE, []
            self.answered.clear()
            self.seen.clear()
        self.speed_pub.publish(Float32(data=float(step.ego_speed)))
        self.info_pub.publish(camera_to_msg(step.camera, header))
        self.image_pub.publish(bgr_to_image(step.image, header))
        return header

    def run(self) -> dict:
        if not self.wait_for_graph():
            raise RuntimeError("perception_node / decision_node did not appear")
        step = self.sim.start()
        self.get_logger().info(f"scenario started: {self.scenario_cfg.get('name', '')}")
        frames, started = 0, time.monotonic()
        try:
            while step is not None and rclpy.ok():
                self.publish(step)
                timeout = self.get_parameter(
                    "first_answer_timeout_s" if frames == 0 else "answer_timeout_s"
                ).value
                command = self.answer if self.answered.wait(timeout) else None
                if command is None:
                    self.get_logger().warn(f"no command for t={step.timestamp:.2f}s within {timeout:.0f} s")
                else:
                    # Perception publishes the objects before decision answers, but the two topics reach
                    # this node independently: give the objects a moment
                    self.seen.wait(1.0)
                    if command.brake > 0:
                        self.get_logger().info(
                            f"t={step.timestamp:.2f}s {command.reason}", throttle_duration_sec=1.0
                        )
                index = round(step.timestamp / TICK_S)
                with self.lock:
                    self.estimates[index] = self.estimate
                    objects = [object_row(s) for s in self.objects]
                    self.object_log.append({"t": step.timestamp, "objects": objects})
                if self.out_dir is not None:
                    path = str(self.out_dir / "camera" / f"{index:06d}.jpg")
                    cv2.imwrite(path, step.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                step = self.sim.step(command)
                frames += 1
                if frames % 20 == 0:
                    rate = frames / (time.monotonic() - started)
                    self.get_logger().info(f"{frames} frames, {rate:.2f} frames/s wall clock")
        finally:
            summary = self.sim.close()
        time.sleep(1.0)  # the last overlay images are still in flight
        summary = {**summary, "frames": frames, "overlay_frames_saved": self.saved_frames}
        self.save(summary)
        self.get_logger().info("summary: " + json.dumps(summary, default=str))
        return summary

    def save(self, summary: dict) -> None:
        if self.out_dir is None:
            return
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        with open(self.out_dir / "objects.jsonl", "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in self.object_log)
        rows = self.sim.rows() if hasattr(self.sim, "rows") else []
        rows = [{**row, **self.estimates.get(round(row["t"] / TICK_S), NO_ESTIMATE)} for row in rows]
        if rows:
            with open(self.out_dir / "ticks.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarlaBridgeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
