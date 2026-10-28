"""FrameSource and Sink adapters (video, image folder, CARLA, W&B)."""

from pathlib import Path

from perception.io.carla_recording import CarlaRecording
from perception.io.sinks import VideoSink
from perception.io.sources import ImageFolderSource, VideoSource

__all__ = ["CarlaRecording", "ImageFolderSource", "VideoSink", "VideoSource", "open_source"]


def open_source(
    path: str | Path, fov_deg: float = 90.0, height_m: float = 1.5, pitch_deg: float = 0.0, fps: float = 10.0
):
    """(source, frames per second) for a path: a folder with meta.json is a CARLA recording (which knows its
    camera and rate), any other folder an image sequence at fps, anything else a video file."""
    path = Path(path)
    camera = {"fov_deg": fov_deg, "height_m": height_m, "pitch_deg": pitch_deg}
    if path.is_dir() and (path / "meta.json").exists():
        recording = CarlaRecording(path)
        return recording, float(recording.meta.get("fps", 20))
    if path.is_dir():
        return ImageFolderSource(path, fps=fps, **camera), fps
    video = VideoSource(path, **camera)
    return video, video.frame_rate
