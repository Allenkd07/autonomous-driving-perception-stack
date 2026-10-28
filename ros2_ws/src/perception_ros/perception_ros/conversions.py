"""ROS messages <-> perception.core.types. All mapping between the two lives here, so nodes stay thin.

Images are converted with numpy instead of cv_bridge: Humble's cv_bridge is built against Ubuntu's numpy,
not the venv's, and the plain reshape covers the encodings used here (bgr8, rgb8, bgra8, rgba8).
Unknown distances and TTCs travel as NaN, since ROS messages have no optional fields.
"""

from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Time
from perception_msgs.msg import ControlCommand as ControlCommandMsg
from perception_msgs.msg import LaneInfo, TrackedObject, TrackedObjectArray
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from perception.core.types import CameraInfo, ControlCommand, Frame, LaneResult, ObjectState, Track

# encoding -> (channels, channel order to get BGR)
ENCODINGS = {"bgr8": (3, [0, 1, 2]), "rgb8": (3, [2, 1, 0]), "bgra8": (4, [0, 1, 2]), "rgba8": (4, [2, 1, 0])}


def stamp_to_seconds(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def seconds_to_stamp(seconds: float) -> Time:
    sec = math.floor(seconds)
    return Time(sec=int(sec), nanosec=int(round((seconds - sec) * 1e9)) % 1_000_000_000)


def _optional(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _from_optional(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


# --- images and cameras ---------------------------------------------------------------------------------


def image_to_bgr(msg: Image) -> np.ndarray:
    """(H, W, 3) uint8 BGR copy of an image message."""
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}; expected one of {sorted(ENCODINGS)}")
    channels, order = ENCODINGS[msg.encoding]
    rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    pixels = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    return np.ascontiguousarray(pixels[:, :, order])


def bgr_to_image(image: np.ndarray, header: Header) -> Image:
    h, w = image.shape[:2]
    return Image(
        header=header,
        height=h,
        width=w,
        encoding="bgr8",
        is_bigendian=0,
        step=w * 3,
        data=np.ascontiguousarray(image, np.uint8).tobytes(),
    )


def camera_from_msg(msg: CameraInfoMsg, height_m: float, pitch_rad: float = 0.0) -> CameraInfo:
    """Intrinsics from a CameraInfo message; mounting height and pitch are not in it, so they are given."""
    K = np.array(msg.k, np.float64).reshape(3, 3)
    return CameraInfo(width=msg.width, height=msg.height, K=K, height_m=height_m, pitch_rad=pitch_rad)


def camera_to_msg(camera: CameraInfo, header: Header) -> CameraInfoMsg:
    P = np.c_[camera.K, np.zeros(3)]
    return CameraInfoMsg(
        header=header,
        width=camera.width,
        height=camera.height,
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        k=camera.K.ravel().tolist(),
        r=np.eye(3).ravel().tolist(),
        p=P.ravel().tolist(),
    )


def frame_from_msgs(image: Image, camera: CameraInfo, frame_id: int) -> Frame:
    return Frame(
        image=image_to_bgr(image),
        timestamp=stamp_to_seconds(image.header.stamp),
        frame_id=frame_id,
        camera=camera,
    )


# --- perception results ---------------------------------------------------------------------------------


def objects_to_msg(objects: list[ObjectState], header: Header) -> TrackedObjectArray:
    return TrackedObjectArray(
        header=header,
        objects=[
            TrackedObject(
                track_id=int(s.track.track_id),
                class_id=int(s.track.class_id),
                class_name=s.track.class_name,
                score=float(s.track.score),
                box_xyxy=np.asarray(s.track.box, np.float32).tolist(),
                distance_m=_optional(s.distance_m),
                ttc_s=_optional(s.ttc_s),
                in_ego_lane=bool(s.in_ego_lane),
            )
            for s in objects
        ],
    )


def objects_from_msg(msg: TrackedObjectArray) -> list[ObjectState]:
    return [
        ObjectState(
            track=Track(
                box=np.array(o.box_xyxy, np.float32),
                score=float(o.score),
                class_id=int(o.class_id),
                class_name=o.class_name,
                track_id=int(o.track_id),
            ),
            distance_m=_from_optional(o.distance_m),
            ttc_s=_from_optional(o.ttc_s),
            in_ego_lane=bool(o.in_ego_lane),
        )
        for o in msg.objects
    ]


def lanes_to_msg(lanes: LaneResult | None, header: Header) -> LaneInfo:
    """The ego-lane mask is not sent (too large for every frame); objects already carry in_ego_lane."""
    if lanes is None:
        return LaneInfo(header=header, left_valid=False, right_valid=False, confidence=0.0)
    return LaneInfo(
        header=header,
        left_valid=lanes.left is not None,
        left_coeffs=[] if lanes.left is None else np.asarray(lanes.left, np.float32).tolist(),
        right_valid=lanes.right is not None,
        right_coeffs=[] if lanes.right is None else np.asarray(lanes.right, np.float32).tolist(),
        confidence=float(lanes.confidence),
    )


def lanes_from_msg(msg: LaneInfo) -> LaneResult:
    return LaneResult(
        left=np.array(msg.left_coeffs, np.float64) if msg.left_valid else None,
        right=np.array(msg.right_coeffs, np.float64) if msg.right_valid else None,
        ego_lane_mask=None,
        confidence=float(msg.confidence),
    )


def command_to_msg(command: ControlCommand, header: Header) -> ControlCommandMsg:
    return ControlCommandMsg(
        header=header,
        throttle=float(command.throttle),
        brake=float(command.brake),
        steer=float(command.steer),
        reason=command.reason,
    )


def command_from_msg(msg: ControlCommandMsg) -> ControlCommand:
    return ControlCommand(throttle=msg.throttle, brake=msg.brake, steer=msg.steer, reason=msg.reason)
