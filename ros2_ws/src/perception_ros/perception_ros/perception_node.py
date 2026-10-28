"""perception_node: camera images in; tracked objects (distance, TTC, in-lane), lanes and an overlay out.

Runs the whole PerceptionPipeline in this one process, so every model shares a single CUDA context. The
pipeline is built from the same YAML config as scripts/run_pipeline.py; this node only converts messages.

Topic names are relative: launched in the "perception" namespace they become /perception/objects etc., and
image / camera_info are remapped to the camera's topics.
Subscribes  image (sensor_msgs/Image), camera_info (sensor_msgs/CameraInfo)
Publishes   objects (perception_msgs/TrackedObjectArray), lanes (perception_msgs/LaneInfo),
            debug_image (sensor_msgs/Image, overlay; only when publish_debug_image is true)
Service     ~/reset (std_srvs/Trigger): clear tracks and TTC history, e.g. between scenarios
Parameters  config (pipeline YAML), overrides (key=value list), camera_height_m and camera_pitch_deg (not in
            CameraInfo), publish_debug_image, log_every (frames between timing logs; 0 = off)

Images use best-effort QoS with a queue of one: when the pipeline is slower than the camera, old frames are
dropped and the newest is processed, instead of falling further and further behind. Results are reliable.
Every output carries the header of the image it came from.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import rclpy
from perception_msgs.msg import LaneInfo, TrackedObjectArray
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from perception.core.config import load_config
from perception.pipeline import PerceptionPipeline
from perception.viz.overlay import draw_output
from perception_ros.conversions import (
    bgr_to_image,
    camera_from_msg,
    frame_from_msgs,
    lanes_to_msg,
    objects_to_msg,
)

LATEST_FRAME_ONLY = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
)
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)


class PerceptionNode(Node):
    def __init__(self, pipeline: PerceptionPipeline | None = None, **kwargs) -> None:
        super().__init__("perception_node", **kwargs)
        self.declare_parameter("config", "configs/pipeline/default.yaml")
        self.declare_parameter("overrides", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("camera_height_m", 1.6)
        self.declare_parameter("camera_pitch_deg", 0.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("log_every", 100)

        if pipeline is None:
            config = self.get_parameter("config").value
            overrides = list(self.get_parameter("overrides").value or [])
            self.get_logger().info(f"building pipeline from {config} {overrides}")
            pipeline = PerceptionPipeline.from_config(load_config(config, overrides))
        self.pipeline = pipeline
        self.camera = None
        self.frames = 0
        self.timings: dict[str, list[float]] = defaultdict(list)

        self.objects_pub = self.create_publisher(TrackedObjectArray, "objects", RELIABLE)
        self.lanes_pub = self.create_publisher(LaneInfo, "lanes", RELIABLE)
        self.debug_pub = self.create_publisher(Image, "debug_image", LATEST_FRAME_ONLY)
        self.create_subscription(CameraInfoMsg, "camera_info", self.on_camera_info, LATEST_FRAME_ONLY)
        self.create_subscription(Image, "image", self.on_image, LATEST_FRAME_ONLY)
        self.create_service(Trigger, "~/reset", self.on_reset)

    def on_camera_info(self, msg: CameraInfoMsg) -> None:
        if self.camera is None:
            self.get_logger().info(f"camera {msg.width}x{msg.height}, fx {msg.k[0]:.1f}")
        self.camera = camera_from_msg(
            msg,
            height_m=self.get_parameter("camera_height_m").value,
            pitch_rad=math.radians(self.get_parameter("camera_pitch_deg").value),
        )

    def on_image(self, msg: Image) -> None:
        if self.camera is None:
            self.get_logger().warn("image before camera_info: skipped", throttle_duration_sec=5.0)
            return
        frame = frame_from_msgs(msg, self.camera, frame_id=self.frames)
        output = self.pipeline.process(frame)
        self.objects_pub.publish(objects_to_msg(output.objects, msg.header))
        self.lanes_pub.publish(lanes_to_msg(output.lanes, msg.header))
        if self.get_parameter("publish_debug_image").value:
            self.debug_pub.publish(bgr_to_image(draw_output(output), msg.header))
        self.frames += 1
        self._log_timings(output.timings_ms)

    def on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.pipeline.reset()
        response.success = True
        response.message = f"pipeline reset after {self.frames} frames"
        self.get_logger().info(response.message)
        return response

    def _log_timings(self, timings_ms: dict[str, float]) -> None:
        every = self.get_parameter("log_every").value
        if not every:
            return
        for stage, ms in timings_ms.items():
            self.timings[stage].append(ms)
        self.timings["total"].append(sum(timings_ms.values()))
        if self.frames % every == 0:
            means = ", ".join(f"{s} {np.mean(v):.1f}" for s, v in self.timings.items())
            self.get_logger().info(f"{self.frames} frames; mean ms over the last {every}: {means}")
            self.timings.clear()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
