"""player_node: plays recorded frames as a camera, so the ROS graph runs without the simulator.

Plays a CARLA recording (carla_scripts/record_sequence.py), a folder of images or a video, detected as in
perception.io.open_source.

Publishes   image (sensor_msgs/Image, bgr8), camera_info (sensor_msgs/CameraInfo),
            ego_speed (std_msgs/Float32, m/s; CARLA recordings only, from the recorded ego velocity)
Parameters  input (path), rate (Hz; 0 = the recording's own rate), loop, fov_deg / camera_height_m /
            pitch_deg (camera of image folders and videos; recordings carry their own), fps (image folders)

Header stamps are the recorded times, continued across loops so time never runs backwards (TTC histories
would otherwise see objects jump back in time).
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header

from perception.io import CarlaRecording, open_source
from perception_ros.conversions import bgr_to_image, camera_to_msg, seconds_to_stamp

CAMERA_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)


class PlayerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("player_node", **kwargs)
        self.declare_parameter("input", "")
        self.declare_parameter("rate", 0.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("fov_deg", 90.0)
        self.declare_parameter("camera_height_m", 1.5)
        self.declare_parameter("pitch_deg", 0.0)
        self.declare_parameter("fps", 10.0)

        path = self.get_parameter("input").value
        if not path:
            raise ValueError("parameter 'input' (recording, image folder or video) is required")
        self.source, fps = open_source(
            path,
            fov_deg=self.get_parameter("fov_deg").value,
            height_m=self.get_parameter("camera_height_m").value,
            pitch_deg=self.get_parameter("pitch_deg").value,
            fps=self.get_parameter("fps").value,
        )
        self.period = 1.0 / fps
        rate = self.get_parameter("rate").value or fps
        self.get_logger().info(f"playing {path} ({type(self.source).__name__}) at {rate:.1f} Hz")

        self.image_pub = self.create_publisher(Image, "image", CAMERA_QOS)
        self.info_pub = self.create_publisher(CameraInfoMsg, "camera_info", CAMERA_QOS)
        self.speed_pub = self.create_publisher(Float32, "ego_speed", RELIABLE)
        self.frames = iter(self.source)
        self.time_offset = 0.0
        self.last_timestamp = 0.0
        self.published = 0
        self.timer = self.create_timer(1.0 / rate, self.tick)

    def _next_frame(self):
        try:
            return next(self.frames)
        except StopIteration:
            if not self.get_parameter("loop").value:
                return None
            self.time_offset += self.last_timestamp + self.period
            self.frames = iter(self.source)
            return next(self.frames, None)

    def tick(self) -> None:
        frame = self._next_frame()
        if frame is None:
            self.get_logger().info(f"end of input after {self.published} frames")
            self.timer.cancel()
            return
        self.last_timestamp = frame.timestamp
        header = Header(stamp=seconds_to_stamp(frame.timestamp + self.time_offset), frame_id="camera")
        self.info_pub.publish(camera_to_msg(frame.camera, header))
        self.image_pub.publish(bgr_to_image(frame.image, header))
        if isinstance(self.source, CarlaRecording):
            velocity = self.source.frames.loc[frame.frame_id, "ego_velocity"]
            self.speed_pub.publish(Float32(data=float(np.linalg.norm(velocity))))
        self.published += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
