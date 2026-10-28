"""Appearance embeddings for tracking: one unit-length feature vector per detected box.

Pre- and post-processing live here; the model itself runs on any InferenceBackend (torch / onnx / tensorrt),
so swapping the backend doesn't change the features beyond numerical precision.
"""

from __future__ import annotations

import cv2
import numpy as np

from perception.core.interfaces import InferenceBackend

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_batch(image: np.ndarray, boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Crop xyxy boxes from a BGR image into a (N, 3, H, W) float32 batch: RGB, ImageNet-normalised.

    Boxes are clipped to the image and kept at least one pixel wide and tall. size is (width, height).
    """
    h, w = image.shape[:2]
    batch = np.empty((len(boxes), 3, size[1], size[0]), np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1 = min(max(int(x1), 0), w - 1)
        y1 = min(max(int(y1), 0), h - 1)
        x2 = max(min(int(np.ceil(x2)), w), x1 + 1)
        y2 = max(min(int(np.ceil(y2)), h), y1 + 1)
        crop = cv2.resize(image[y1:y2, x1:x2], size, interpolation=cv2.INTER_LINEAR)
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        batch[i] = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return batch


class AppearanceEmbedder:
    """Runs an embedding model (input "images", output "features") on box crops.

    input_size is (width, height). Features are L2-normalised, so cosine distance is 1 - dot product.
    """

    def __init__(self, backend: InferenceBackend, input_size: tuple[int, int] = (128, 128)) -> None:
        self.backend = backend
        self.input_size = (int(input_size[0]), int(input_size[1]))

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 0), np.float32)
        features = self.backend.run({"images": crop_batch(image, boxes, self.input_size)})["features"]
        features = features.reshape(len(boxes), -1)
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
