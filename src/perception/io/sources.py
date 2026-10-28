"""FrameSources for recorded camera data: a video file or a folder of images.

Neither format stores the camera, so its geometry is given here: horizontal field of view, mounting height
and pitch. Distances from the ground plane are only as good as these numbers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from perception.core.registry import register
from perception.core.types import CameraInfo, Frame

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def make_camera(width: int, height: int, fov_deg: float, height_m: float, pitch_deg: float) -> CameraInfo:
    camera = CameraInfo.from_fov(width, height, fov_deg, height_m)
    return dataclasses.replace(camera, pitch_rad=float(np.radians(pitch_deg)))


@register("source", "video")
class VideoSource:
    """Frames of a video file; timestamps from the frame index and the file's frame rate (or fps)."""

    def __init__(
        self,
        path: str | Path,
        fov_deg: float = 90.0,
        height_m: float = 1.5,
        pitch_deg: float = 0.0,
        fps: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.fov_deg, self.height_m, self.pitch_deg = fov_deg, height_m, pitch_deg
        self.fps = fps

    @property
    def frame_rate(self) -> float:
        """fps if given, else the file's frame rate (30 if the file doesn't say)."""
        if self.fps:
            return self.fps
        capture = cv2.VideoCapture(str(self.path))
        try:
            return capture.get(cv2.CAP_PROP_FPS) or 30.0
        finally:
            capture.release()

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        fps = self.frame_rate
        camera = None
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if camera is None:
                    h, w = image.shape[:2]
                    camera = make_camera(w, h, self.fov_deg, self.height_m, self.pitch_deg)
                yield Frame(image=image, timestamp=index / fps, frame_id=index, camera=camera)
                index += 1
        finally:
            capture.release()


@register("source", "images")
class ImageFolderSource:
    """Images of a folder in name order, as a sequence at fps frames per second."""

    def __init__(
        self,
        folder: str | Path,
        fov_deg: float = 90.0,
        height_m: float = 1.5,
        pitch_deg: float = 0.0,
        fps: float = 10.0,
    ) -> None:
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.fov_deg, self.height_m, self.pitch_deg = fov_deg, height_m, pitch_deg
        self.fps = fps

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        cameras: dict[tuple[int, int], CameraInfo] = {}
        for index, path in enumerate(self.paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            if (w, h) not in cameras:
                cameras[w, h] = make_camera(w, h, self.fov_deg, self.height_m, self.pitch_deg)
            yield Frame(image=image, timestamp=index / self.fps, frame_id=index, camera=cameras[w, h])
