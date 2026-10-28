"""perception_node over real ROS topics, with a fake pipeline (no models, no GPU)."""

import numpy as np
from perception_msgs.msg import LaneInfo, TrackedObjectArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from perception.core.types import CameraInfo, ObjectState, PerceptionOutput, Track
from perception_ros.conversions import bgr_to_image, camera_to_msg, seconds_to_stamp
from perception_ros.perception_node import PerceptionNode

CAMERA = CameraInfo.from_fov(80, 60, 90.0, 1.6)


class FakePipeline:
    def __init__(self):
        self.frames = []
        self.resets = 0

    def process(self, frame):
        self.frames.append(frame)
        track = Track(
            box=np.array([1, 2, 30, 40], np.float32), score=0.9, class_id=3, class_name="person", track_id=5
        )
        state = ObjectState(track=track, distance_m=7.5, ttc_s=1.2, in_ego_lane=True)
        return PerceptionOutput(frame=frame, objects=[state], timings_ms={"detect": 1.0})

    def reset(self):
        self.resets += 1


def make_graph(ros):
    pipeline = FakePipeline()
    node = PerceptionNode(pipeline=pipeline)
    helper = ros.create_node("test_helper")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    return pipeline, node, helper, executor


def test_image_in_objects_and_lanes_out(ros, spin_until):
    pipeline, node, helper, executor = make_graph(ros)
    header = Header(stamp=seconds_to_stamp(3.5), frame_id="camera")
    image_pub = helper.create_publisher(Image, "image", QoSProfile(depth=1))
    info_pub = helper.create_publisher(CameraInfoMsg, "camera_info", QoSProfile(depth=1))
    objects, lanes = [], []
    helper.create_subscription(TrackedObjectArray, "objects", objects.append, QoSProfile(depth=10))
    helper.create_subscription(LaneInfo, "lanes", lanes.append, QoSProfile(depth=10))
    image = np.random.default_rng(0).integers(0, 255, (60, 80, 3), np.uint8)

    def publish():
        info_pub.publish(camera_to_msg(CAMERA, header))
        image_pub.publish(bgr_to_image(image, header))

    assert spin_until(executor, lambda: objects and lanes, publish=publish)

    (obj,) = objects[0].objects
    assert objects[0].header.stamp == header.stamp  # outputs carry the image's header
    assert (obj.track_id, obj.class_name, obj.in_ego_lane) == (5, "person", True)
    assert obj.distance_m == np.float32(7.5)
    frame = pipeline.frames[0]
    np.testing.assert_array_equal(frame.image, image)
    np.testing.assert_allclose(frame.camera.K, CAMERA.K)
    assert frame.camera.height_m == 1.6 and frame.timestamp == 3.5
    executor.shutdown()


def test_images_before_camera_info_are_skipped(ros, spin_until):
    pipeline, node, helper, executor = make_graph(ros)
    image_pub = helper.create_publisher(Image, "image", QoSProfile(depth=1))
    image = bgr_to_image(np.zeros((60, 80, 3), np.uint8), Header())

    spin_until(executor, lambda: False, timeout_s=1.0, publish=lambda: image_pub.publish(image))

    assert pipeline.frames == []
    executor.shutdown()


def test_reset_service(ros, spin_until):
    pipeline, node, helper, executor = make_graph(ros)
    client = helper.create_client(Trigger, "/perception_node/reset")
    assert spin_until(executor, lambda: client.service_is_ready())

    future = client.call_async(Trigger.Request())
    assert spin_until(executor, future.done)

    assert future.result().success and pipeline.resets == 1
    executor.shutdown()
