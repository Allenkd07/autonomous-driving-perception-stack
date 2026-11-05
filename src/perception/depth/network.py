"""Monocular depth networks: MiDaS and Depth Anything V2.

Both take an RGB image normalised with the ImageNet mean and std and return one depth channel, so one class
runs either (exported by scripts/export_midas.py and scripts/export_depth_anything.py). What the output means
depends on the training:

relative  (MiDaS, Depth Anything V2) trained on many datasets with unknown and differing camera scales, so it
          predicts inverse depth (larger = nearer) correct only up to an unknown scale and shift per image:
              output = a / depth + b,   a > 0, a and b different in every frame
          Turning it into metres takes a reference per frame (perception.fusion.depth_scale).
metric    (Depth Anything V2 fine-tuned on Virtual KITTI 2, a synthetic driving dataset) predicts metres
          directly, but only for the focal length it was fine-tuned with: an object's size in pixels is
          f * size / depth, so the network reads depth from apparent size, and a wider lens (smaller f)
          makes everything look farther. With train_focal_px set, the output is rescaled by
          f_camera / f_train, both in pixels at the network's input (the canonical-camera idea of Metric3D).
"""

from __future__ import annotations

import cv2
import numpy as np

from perception.core import registry
from perception.core.registry import register
from perception.core.types import DepthMap, Frame

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@register("depth", "depth_anything")
@register("depth", "midas")
class NetworkDepth:
    """DepthEstimator running an exported depth network.

    model        exported network (.ts for the torch backend, .onnx / .engine later)
    input_size   (width, height) the model was exported with; frames of another aspect are stretched to it
    metric       the output is in metres (metric models) rather than relative inverse depth
    train_focal_px  metric models: horizontal focal length of the training images at the network input, in
                 pixels (Virtual KITTI 2: fx 725.0 at 1242 x 375, resized to a 518 px short side -> 1001.5);
                 None leaves the output as predicted
    backend      InferenceBackend type: torch, onnx, tensorrt
    device, half passed to the backend
    """

    def __init__(
        self,
        model: str,
        input_size: tuple[int, int] = (256, 192),
        metric: bool = False,
        train_focal_px: float | None = None,
        backend: str = "torch",
        device: str | None = None,
        half: bool = False,
    ) -> None:
        self.input_w, self.input_h = input_size
        self.metric = metric
        self.train_focal_px = train_focal_px
        self.backend = registry.build(
            "backend",
            {
                "type": backend,
                "model": model,
                "inputs": ["images"],
                "outputs": ["depth"],
                "device": device,
                "half": half,
            },
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR uint8 (H, W, 3) -> (1, 3, h, w) float32, RGB normalised as in training."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def focal_scale(self, fx: float, image_width: int) -> float:
        """Factor from the metric output to metres for a camera with focal length fx (pixels, full image)."""
        return fx * self.input_w / image_width / self.train_focal_px

    def estimate(self, frame: Frame) -> DepthMap:
        out = self.backend.run({"images": self.preprocess(frame.image)})["depth"]
        h, w = frame.image.shape[:2]
        depth = cv2.resize(out.reshape(self.input_h, self.input_w), (w, h), interpolation=cv2.INTER_LINEAR)
        if self.metric and self.train_focal_px:
            depth = depth * self.focal_scale(frame.camera.K[0, 0], w)
        return DepthMap(depth=depth.astype(np.float32), is_metric=self.metric)


@register("depth", "ground_truth")
class GroundTruthDepth:
    """The frame's ground-truth depth in metres (CARLA's depth camera): an upper bound for experiments."""

    def estimate(self, frame: Frame) -> DepthMap:
        if frame.depth_gt is None:
            raise ValueError("GroundTruthDepth needs frames with depth_gt (CARLA recordings)")
        return DepthMap(depth=frame.depth_gt.astype(np.float32), is_metric=True)
