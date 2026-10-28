"""decision_node over real ROS topics with the real emergency-brake policy."""

import numpy as np
from perception_msgs.msg import ControlCommand as ControlCommandMsg
from perception_msgs.msg import TrackedObjectArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32, Header

from perception.core.types import ObjectState, Track
from perception.planning.emergency_brake import EmergencyBrakePolicy
from perception_ros.conversions import objects_to_msg, seconds_to_stamp
from perception_ros.decision_node import DecisionNode


def objects(t: float, ttc: float | None, in_lane: bool = True) -> TrackedObjectArray:
    track = Track(box=np.zeros(4, np.float32), score=0.9, class_id=3, class_name="person", track_id=12)
    state = ObjectState(track=track, distance_m=10.0, ttc_s=ttc, in_ego_lane=in_lane)
    return objects_to_msg([state], Header(stamp=seconds_to_stamp(t), frame_id="camera"))


def test_brakes_and_releases(ros, spin_until):
    node = DecisionNode(policy=EmergencyBrakePolicy(hold_s=1.0))
    helper = ros.create_node("test_helper")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    objects_pub = helper.create_publisher(TrackedObjectArray, "objects", QoSProfile(depth=10))
    speed_pub = helper.create_publisher(Float32, "ego_speed", QoSProfile(depth=10))
    commands = []
    helper.create_subscription(ControlCommandMsg, "control", commands.append, QoSProfile(depth=10))

    def publish_until_answered(msg):
        """The command decided from msg: matched by header stamp, since msg is re-sent until it arrives and
        replies to earlier messages may still be in flight."""

        def answer():
            return next((c for c in commands if c.header.stamp == msg.header.stamp), None)

        assert spin_until(executor, lambda: answer() is not None, publish=lambda: objects_pub.publish(msg))
        return answer()

    assert spin_until(
        executor, lambda: node.ego_speed == 8.0, publish=lambda: speed_pub.publish(Float32(data=8.0))
    )
    danger = publish_until_answered(objects(0.0, ttc=1.0))
    later = publish_until_answered(objects(2.0, ttc=None, in_lane=False))  # gone, after the 1 s hold

    assert danger.brake == 1.0 and "track 12" in danger.reason
    assert danger.header.stamp == seconds_to_stamp(0.0)
    assert later.brake == 0.0 and later.reason == ""
    assert not node.braking
    executor.shutdown()
