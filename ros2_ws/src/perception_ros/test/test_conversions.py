"""Round trips between ROS messages and perception types. Needs ROS and the built workspace:

source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash && source .venv/bin/activate
pytest ros2_ws/src/perception_ros/test
"""

import math

import numpy as np
import pytest
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from perception.core.types import CameraInfo, ControlCommand, LaneResult, ObjectState, Track
from perception_ros.conversions import (
    bgr_to_image,
    camera_from_msg,
    camera_to_msg,
    command_from_msg,
    command_to_msg,
    frame_from_msgs,
    image_to_bgr,
    lanes_from_msg,
    lanes_to_msg,
    objects_from_msg,
    objects_to_msg,
    seconds_to_stamp,
    stamp_to_seconds,
)

CAMERA = CameraInfo.from_fov(80, 60, 90.0, 1.6)


def header(t: float = 12.25) -> Header:
    return Header(stamp=seconds_to_stamp(t), frame_id="camera")


def image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (60, 80, 3), np.uint8)


def test_stamps():
    assert stamp_to_seconds(seconds_to_stamp(12.25)) == pytest.approx(12.25)
    assert seconds_to_stamp(3.999999999).sec in (3, 4)


def test_image_round_trip_and_frame():
    bgr = image()
    msg = bgr_to_image(bgr, header())

    np.testing.assert_array_equal(image_to_bgr(msg), bgr)
    frame = frame_from_msgs(msg, CAMERA, frame_id=7)
    assert frame.timestamp == pytest.approx(12.25) and frame.frame_id == 7
    np.testing.assert_array_equal(frame.image, bgr)


@pytest.mark.parametrize("encoding", ["rgb8", "bgra8"])
def test_other_encodings_and_row_padding(encoding):
    bgr = image()
    if encoding == "rgb8":
        pixels = bgr[:, :, ::-1]
    else:
        pixels = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
    channels = pixels.shape[2]
    step = 80 * channels + 16  # rows padded, as some drivers do
    rows = np.zeros((60, step), np.uint8)
    rows[:, : 80 * channels] = pixels.reshape(60, -1)
    msg = Image(height=60, width=80, encoding=encoding, step=step, data=rows.tobytes())

    np.testing.assert_array_equal(image_to_bgr(msg), bgr)


def test_unsupported_encoding():
    with pytest.raises(ValueError, match="mono16"):
        image_to_bgr(Image(height=1, width=1, encoding="mono16", step=2, data=b"\x00\x00"))


def test_camera_round_trip():
    camera = camera_from_msg(camera_to_msg(CAMERA, header()), height_m=1.6, pitch_rad=0.02)

    np.testing.assert_allclose(camera.K, CAMERA.K)
    assert (camera.width, camera.height, camera.height_m, camera.pitch_rad) == (80, 60, 1.6, 0.02)


def test_objects_round_trip_with_unknowns():
    track = Track(
        box=np.array([1, 2, 30, 40], np.float32), score=0.8, class_id=3, class_name="person", track_id=12
    )
    states = [
        ObjectState(track=track, distance_m=8.5, ttc_s=1.25, in_ego_lane=True),
        ObjectState(track=track, distance_m=None, ttc_s=None, in_ego_lane=False),
    ]

    msg = objects_to_msg(states, header())
    back = objects_from_msg(msg)

    assert math.isnan(msg.objects[1].distance_m) and math.isnan(msg.objects[1].ttc_s)
    assert back[0].distance_m == pytest.approx(8.5) and back[0].ttc_s == pytest.approx(1.25)
    assert back[1].distance_m is None and back[1].ttc_s is None and not back[1].in_ego_lane
    assert back[0].track.track_id == 12 and back[0].track.class_name == "person"
    np.testing.assert_array_equal(back[0].track.box, track.box)


def test_lanes_round_trip():
    lanes = LaneResult(
        left=np.array([0.0, -1.0, 700.0]), right=None, ego_lane_mask=np.ones((2, 2), bool), confidence=0.5
    )

    back = lanes_from_msg(lanes_to_msg(lanes, header()))

    np.testing.assert_allclose(back.left, lanes.left)
    assert back.right is None and back.ego_lane_mask is None and back.confidence == 0.5
    assert lanes_from_msg(lanes_to_msg(None, header())).left is None


def test_command_round_trip():
    command = ControlCommand(throttle=0.0, brake=1.0, steer=0.0, reason="EMERGENCY_BRAKE track 12 ttc=1.3s")

    assert command_from_msg(command_to_msg(command, header())) == command
