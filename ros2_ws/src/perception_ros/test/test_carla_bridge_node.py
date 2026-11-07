"""carla_bridge_node's lockstep loop over real ROS topics, with a fake simulator (no CARLA) and a helper node
standing in for perception + decision."""

import csv
import json
import threading

import numpy as np
from perception_msgs.msg import ControlCommand as ControlCommandMsg
from perception_msgs.msg import TrackedObjectArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image

from perception.core.types import CameraInfo, ControlCommand, ObjectState, Track
from perception.sim.base import SimStep
from perception_ros.carla_bridge_node import CarlaBridgeNode
from perception_ros.conversions import bgr_to_image, command_to_msg, objects_to_msg, stamp_to_seconds

CAMERA = CameraInfo.from_fov(64, 48, 90.0, 1.6)


class FakeSimulator:
    """frames ticks 0.05 s apart; records the command it got for each frame."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.index = 0
        self.received: list[ControlCommand | None] = []
        self.closed = False

    def _step(self) -> SimStep:
        image = np.full((48, 64, 3), self.index, np.uint8)
        return SimStep(image=image, timestamp=self.index * 0.05, camera=CAMERA, ego_speed=8.0)

    def start(self) -> SimStep:
        return self._step()

    def step(self, override):
        self.received.append(override)
        self.index += 1
        return self._step() if self.index < self.frames else None

    def close(self) -> dict:
        self.closed = True
        return {"collision": False}

    def rows(self) -> list[dict]:
        return [{"t": i * 0.05, "brake": c.brake if c else None} for i, c in enumerate(self.received)]


def test_waits_for_remapped_graph(ros):
    # As in carla.launch.py: the bridge's topics are remapped, decision publishes in the perception namespace
    bridge = CarlaBridgeNode(
        simulator=FakeSimulator(frames=1),
        parameter_overrides=[Parameter("graph_timeout_s", value=3.0)],
        cli_args=["--ros-args", "-r", "image:=/camera/image_raw", "-r", "control:=/perception/control"],
    )
    perception = ros.create_node("perception_node", namespace="perception")
    perception.create_subscription(Image, "/camera/image_raw", lambda msg: None, QoSProfile(depth=10))
    decision = ros.create_node("decision_node", namespace="perception")
    decision.create_publisher(ControlCommandMsg, "control", QoSProfile(depth=10))

    assert bridge.wait_for_graph()
    for node in (bridge, perception, decision):
        node.destroy_node()


def test_lockstep_loop_applies_each_frames_command(ros, tmp_path):
    sim = FakeSimulator(frames=6)
    bridge = CarlaBridgeNode(
        simulator=sim,
        parameter_overrides=[
            Parameter("out_dir", value=str(tmp_path)),
            Parameter("answer_timeout_s", value=0.5),
            Parameter("first_answer_timeout_s", value=10.0),
            Parameter("graph_timeout_s", value=10.0),
        ],
    )
    stack = ros.create_node("fake_stack")
    control_pub = stack.create_publisher(ControlCommandMsg, "control", QoSProfile(depth=10))
    debug_pub = stack.create_publisher(Image, "debug_image", QoSProfile(depth=10))
    objects_pub = stack.create_publisher(TrackedObjectArray, "objects", QoSProfile(depth=10))

    def state(track_id: int, distance: float, in_lane: bool) -> ObjectState:
        box = np.array([0, 0, 10, 10], np.float32)
        track = Track(box=box, score=0.9, class_id=2, class_name="car", track_id=track_id)
        return ObjectState(track=track, distance_m=distance, ttc_s=3.0, in_ego_lane=in_lane)

    def on_image(msg: Image) -> None:
        t = stamp_to_seconds(msg.header.stamp)
        if abs(t - 0.10) < 1e-6:
            return  # this frame gets no answer: the bridge must time out and drive on
        brake = 1.0 if t >= 0.15 else 0.0
        reason = "EMERGENCY_BRAKE track 1" if brake else ""
        debug_pub.publish(bgr_to_image(np.zeros((48, 64, 3), np.uint8), msg.header))
        # the nearer object is beside the lane: the bridge logs the in-lane one
        objects = [state(5, 8.0, False), state(1, 40.0 - 100 * t, True), state(2, 60.0, True)]
        objects_pub.publish(objects_to_msg(objects, msg.header))
        control_pub.publish(command_to_msg(ControlCommand(0.0, brake, 0.0, reason), msg.header))

    stack.create_subscription(Image, "image", on_image, QoSProfile(depth=10))
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(stack)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        summary = bridge.run()
    finally:
        executor.shutdown()

    assert sim.closed and len(sim.received) == 6
    assert [c is None for c in sim.received] == [False, False, True, False, False, False]
    assert [c.brake for c in sim.received if c is not None] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert sim.received[3].reason == "EMERGENCY_BRAKE track 1"
    assert summary["frames"] == 6
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["frames"] == 6 and saved["collision"] is False
    with open(tmp_path / "ticks.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert [r["est_track"] for r in rows] == ["1", "1", "", "1", "1", "1"]  # frame 0.10 s went unanswered
    assert [r["est_distance_m"] for r in rows[:2]] == ["40.0", "35.0"]
    assert rows[0]["est_objects"] == "3" and rows[0]["est_ttc_s"] == "3.0"
    logged = [json.loads(line) for line in (tmp_path / "objects.jsonl").read_text().splitlines()]
    assert len(logged) == 6 and logged[2]["objects"] == []  # the unanswered frame
    assert [o["track"] for o in logged[0]["objects"]] == [5, 1, 2] and not logged[0]["objects"][0]["in_lane"]
    assert len(list((tmp_path / "camera").iterdir())) == 6
    assert sorted(p.name for p in (tmp_path / "frames").iterdir())[:2] == ["000000.jpg", "000001.jpg"]
