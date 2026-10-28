from pathlib import Path

import numpy as np
import pytest

from perception.tracking.embedder import IMAGENET_MEAN, IMAGENET_STD, AppearanceEmbedder, crop_batch

REID_MODEL = Path("weights/reid_resnet18.ts")


def test_crop_batch_is_normalised_rgb_nchw():
    image = np.zeros((100, 200, 3), np.uint8)
    image[:, :] = (255, 0, 0)  # pure blue in BGR

    batch = crop_batch(image, np.array([[10, 10, 50, 90]], np.float32), size=(32, 64))

    assert batch.shape == (1, 3, 64, 32) and batch.dtype == np.float32
    expected = (np.array([0.0, 0.0, 1.0], np.float32) - IMAGENET_MEAN) / IMAGENET_STD  # RGB
    np.testing.assert_allclose(batch[0, :, 0, 0], expected, rtol=1e-6)


def test_crop_batch_clips_boxes_to_the_image():
    image = np.full((100, 200, 3), 128, np.uint8)
    boxes = np.array([[-20, -20, 30, 30], [190, 90, 260, 140], [250, 150, 260, 160]], np.float32)

    batch = crop_batch(image, boxes, size=(16, 16))

    assert batch.shape == (3, 3, 16, 16) and np.isfinite(batch).all()


class SumBackend:
    def run(self, inputs):
        images = inputs["images"]
        return {"features": np.stack([images.sum(axis=(2, 3)), np.ones((len(images), 3))], axis=1)}


def test_embedder_returns_unit_vectors():
    image = np.random.default_rng(0).integers(0, 255, (100, 200, 3), np.uint8)
    embed = AppearanceEmbedder(SumBackend(), input_size=(16, 16))

    features = embed(image, np.array([[0, 0, 50, 50], [60, 20, 120, 90]], np.float32))

    assert features.shape == (2, 6)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, rtol=1e-6)
    assert embed(image, np.zeros((0, 4), np.float32)).shape[0] == 0


@pytest.mark.skipif(not REID_MODEL.exists(), reason="run scripts/export_reid.py first")
def test_exported_model_tells_crops_apart():
    from perception.backends.torch_backend import TorchBackend

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 400, 3), np.uint8)
    image[:, 200:] = np.linspace(0, 255, 200, dtype=np.uint8)[None, :, None]  # smooth gradient on the right
    embed = AppearanceEmbedder(TorchBackend(REID_MODEL, device="cpu"), input_size=(128, 128))
    noise, noise_shifted, gradient = embed(
        image, np.array([[10, 10, 110, 110], [14, 14, 114, 114], [250, 50, 350, 150]], np.float32)
    )

    # A slightly shifted crop of the same content looks more alike than a crop of something else
    assert noise @ noise_shifted > noise @ gradient
