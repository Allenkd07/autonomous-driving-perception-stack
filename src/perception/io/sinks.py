"""Sinks for pipeline output written to disk."""

from __future__ import annotations

from pathlib import Path

import cv2

from perception.core.registry import register
from perception.core.types import PerceptionOutput
from perception.viz.overlay import draw_output


@register("sink", "video")
class VideoSink:
    """An MP4 video of the overlays (perception.viz.overlay); the size comes from the first frame."""

    def __init__(self, path: str | Path, fps: float = 20.0, ttc_warning_s: float = 2.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self.ttc_warning_s = ttc_warning_s
        self._writer: cv2.VideoWriter | None = None
        self.frames = 0

    def write(self, output: PerceptionOutput) -> None:
        image = draw_output(output, self.ttc_warning_s)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self._writer.write(image)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
