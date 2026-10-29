"""Draw a PerceptionOutput on its frame: lanes, tracked objects with distance and TTC, stage timings.

Objects in the ego lane get a thick box, and turn red when their time to collision is below ttc_warning_s
(what an emergency-brake policy reacts to). draw_ground_truth shows a CARLA recording's ground truth instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from perception.core.types import PerceptionOutput

if TYPE_CHECKING:  # perception.io imports this module (VideoSink)
    from perception.io.carla_recording import GroundTruthObject

DANGER = (0, 0, 255)
UNLABELLED = (160, 160, 160)
LANE_TINT = np.array([0, 180, 0], np.float32)
LANE_LINE = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def track_colour(track_id: int) -> tuple[int, int, int]:
    """A stable, distinct BGR colour per track id (hues spread by a step coprime with 180)."""
    hsv = np.uint8([[[(track_id * 47) % 180, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _label(image: np.ndarray, text: str, x: int, y: int, colour: tuple[int, int, int], scale: float) -> None:
    """Text on a filled background, above (x, y) when there is room, else below."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    top = y - h - base - 2 if y - h - base - 2 >= 0 else y
    cv2.rectangle(image, (x, top), (x + w + 4, top + h + base + 2), colour, -1)
    cv2.putText(image, text, (x + 2, top + h + 1), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_output(output: PerceptionOutput, ttc_warning_s: float = 2.0) -> np.ndarray:
    image = output.frame.image.copy()
    height, width = image.shape[:2]
    scale = max(0.4, width / 1600)

    lanes = output.lanes
    if lanes is not None and lanes.ego_lane_mask is not None:
        mask = lanes.ego_lane_mask
        image[mask] = (0.7 * image[mask] + 0.3 * LANE_TINT).astype(np.uint8)
        rows = np.flatnonzero(mask.any(axis=1))
        ys = np.arange(rows.min(), rows.max() + 1) if len(rows) else np.arange(0)
        for fit in (lanes.left, lanes.right):
            if fit is not None and len(ys):
                points = np.c_[np.polyval(fit, ys), ys].astype(np.int32)
                cv2.polylines(image, [points], False, LANE_LINE, 2, cv2.LINE_AA)

    for state in output.objects:
        track = state.track
        danger = state.in_ego_lane and state.ttc_s is not None and state.ttc_s < ttc_warning_s
        colour = DANGER if danger else track_colour(track.track_id)
        x1, y1, x2, y2 = (int(round(c)) for c in track.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3 if state.in_ego_lane else 1)
        parts = [f"{track.track_id} {track.class_name}"]
        if state.distance_m is not None:
            parts.append(f"{state.distance_m:.0f} m")
        if state.ttc_s is not None:
            parts.append(f"TTC {state.ttc_s:.1f} s")
        _label(image, " ".join(parts), x1, y1, colour, scale)

    total = sum(output.timings_ms.values())
    stages = "  ".join(f"{name} {ms:.0f}" for name, ms in output.timings_ms.items())
    header = f"{1000 / total:.0f} FPS | ms: {stages}" if total > 0 else stages
    _label(image, header, 0, 0, (255, 255, 255), scale)
    return image


def draw_ground_truth(
    image: np.ndarray,
    objects: list[GroundTruthObject],
    unlabelled: np.ndarray,
    dynamic: np.ndarray | None = None,
    actor_ids: np.ndarray | None = None,
    header: str = "",
) -> np.ndarray:
    """Draw a CARLA recording's 2D ground truth (perception.io.carla_recording), to check it by eye.

    Each actor gets its visible box (thick) and amodal box (thin) in its own colour. Given the instance
    image's vehicle/person mask (dynamic) and per-pixel actor ids (CarlaRecording.actor_ids), those pixels
    are tinted in their actor's colour, or grey if they belong to no ground-truth object; grey boxes are
    unlabelled regions, which evaluation ignores. A tint offset from its box means the sensors or poses are
    out of sync.
    """
    image = image.copy()
    scale = max(0.4, image.shape[1] / 1600)
    if dynamic is not None and actor_ids is not None:
        tint = np.zeros_like(image)
        tint[dynamic] = UNLABELLED
        for obj in objects:
            tint[dynamic & (actor_ids == obj.actor_id)] = track_colour(obj.actor_id)
        image[dynamic] = (0.5 * image[dynamic] + 0.5 * tint[dynamic]).astype(np.uint8)

    for x1, y1, x2, y2 in unlabelled.astype(int):
        cv2.rectangle(image, (x1, y1), (x2, y2), UNLABELLED, 1)
    for obj in objects:
        colour = track_colour(obj.actor_id)
        x1, y1, x2, y2 = obj.amodal_box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 1)
        x1, y1, x2, y2 = obj.box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        _label(image, f"{obj.actor_id} {obj.class_name} {obj.distance_m:.0f} m", x1, y1, colour, scale)

    if header:
        _label(image, header, 0, 0, (255, 255, 255), scale)
    return image
