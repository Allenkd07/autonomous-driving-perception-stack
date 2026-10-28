"""decision_node: tracked objects in, control override out (the emergency-brake policy).

The policy comes from the same pipeline YAML as perception_node (its `policy:` section). Its command only
overrides the route follower: brake > 0 with a reason means brake now; brake 0 and an empty reason mean no
intervention (see perception.planning.emergency_brake).

Subscribes  objects (perception_msgs/TrackedObjectArray), ego_speed (std_msgs/Float32, m/s; 0 until received)
Publishes   control (perception_msgs/ControlCommand), with the header of the objects it was decided from
Service     ~/reset (std_srvs/Trigger)
Parameters  config (pipeline YAML), overrides (key=value list)
"""

from __future__ import annotations

import rclpy
from perception_msgs.msg import ControlCommand as ControlCommandMsg
from perception_msgs.msg import TrackedObjectArray
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from perception.core import registry
from perception.core.config import load_config
from perception_ros.conversions import command_to_msg, output_from_objects_msg

RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)


class DecisionNode(Node):
    def __init__(self, policy=None, **kwargs) -> None:
        super().__init__("decision_node", **kwargs)
        self.declare_parameter("config", "configs/pipeline/default.yaml")
        self.declare_parameter("overrides", Parameter.Type.STRING_ARRAY)
        if policy is None:
            config = self.get_parameter("config").value
            cfg = load_config(config, list(self.get_parameter("overrides").value or []))
            self.get_logger().info(f"policy {cfg.policy} from {config}")
            policy = registry.build("policy", cfg.policy)
        self.policy = policy
        self.ego_speed = 0.0
        self.decisions = 0
        self.braking = False

        self.control_pub = self.create_publisher(ControlCommandMsg, "control", RELIABLE)
        self.create_subscription(TrackedObjectArray, "objects", self.on_objects, RELIABLE)
        self.create_subscription(Float32, "ego_speed", self.on_ego_speed, RELIABLE)
        self.create_service(Trigger, "~/reset", self.on_reset)

    def on_ego_speed(self, msg: Float32) -> None:
        self.ego_speed = float(msg.data)

    def on_objects(self, msg: TrackedObjectArray) -> None:
        command = self.policy.decide(output_from_objects_msg(msg, self.decisions), self.ego_speed)
        self.control_pub.publish(command_to_msg(command, msg.header))
        self.decisions += 1
        # Log transitions only; a message per frame would bury everything else
        braking = command.brake > 0
        if braking and not self.braking:
            self.get_logger().warn(f"{command.reason} (ego {self.ego_speed:.1f} m/s)")
        elif self.braking and not braking:
            self.get_logger().info("braking released")
        self.braking = braking

    def on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self.braking = False
        response.success = True
        response.message = f"policy reset after {self.decisions} decisions"
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
