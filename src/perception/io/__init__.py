"""FrameSource and Sink adapters (video, image folder, CARLA, W&B)."""

from perception.io.carla_recording import CarlaRecording
from perception.io.sinks import VideoSink
from perception.io.sources import ImageFolderSource, VideoSource

__all__ = ["CarlaRecording", "ImageFolderSource", "VideoSink", "VideoSource"]
