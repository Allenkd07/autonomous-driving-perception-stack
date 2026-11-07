import numpy as np
import pytest
import torch

from perception.backends.raw_input import RawImageInput
from perception.core.types import CameraInfo, Frame
from perception.depth.network import IMAGENET_MEAN, IMAGENET_STD, NetworkDepth
from perception.lanes.segmentation import SegLaneDetector


class TinyDepth(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)[:, 0]


class TinySeg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.drivable = torch.nn.Conv2d(3, 2, 3, padding=1)
        self.lanes = torch.nn.Conv2d(3, 2, 1)

    def forward(self, x):
        return self.drivable(x), self.lanes(x)


def bgr_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_normalises_inside_as_the_float_pipeline_does():
    net = TinyDepth().eval()
    image = bgr_image(8, 10)
    rgb = image[..., ::-1].astype(np.float32) / 255.0
    x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]

    with torch.no_grad():
        raw = RawImageInput(net, IMAGENET_MEAN, IMAGENET_STD)(torch.from_numpy(image[None]))
        expected = net(torch.from_numpy(np.ascontiguousarray(x, np.float32)))

    torch.testing.assert_close(raw, expected, atol=1e-5, rtol=1e-5)


def test_probabilities_are_the_class_1_softmax_of_every_output():
    net = TinySeg().eval()
    image = torch.from_numpy(bgr_image(6, 8)[None])

    with torch.no_grad():
        drivable, lanes = RawImageInput(net, probabilities=True)(image)
        logits = net(image.permute(0, 3, 1, 2)[:, [2, 1, 0]].float() / 255.0)

    assert drivable.shape == lanes.shape == (1, 6, 8)
    torch.testing.assert_close(drivable, torch.softmax(logits[0], dim=1)[:, 1])
    torch.testing.assert_close(lanes, torch.softmax(logits[1], dim=1)[:, 1])


def save(module: torch.nn.Module, example: torch.Tensor, path) -> str:
    torch.jit.save(torch.jit.trace(module.eval(), example), str(path))
    return str(path)


@pytest.fixture
def frame():
    image = bgr_image(60, 80, seed=2)
    return Frame(image=image, timestamp=0.0, frame_id=0, camera=CameraInfo.from_fov(80, 60, 90.0, 1.5))


def test_depth_stage_gives_the_same_depth_with_a_raw_input_model(tmp_path, frame):
    net = TinyDepth()
    plain = save(net, torch.zeros(1, 3, 28, 42), tmp_path / "plain.ts")
    raw = save(
        RawImageInput(net, IMAGENET_MEAN, IMAGENET_STD),
        torch.zeros(1, 28, 42, 3, dtype=torch.uint8),
        tmp_path / "raw.ts",
    )

    a = NetworkDepth(plain, input_size=(42, 28), device="cpu").estimate(frame).depth
    b = NetworkDepth(raw, input_size=(42, 28), device="cpu", raw_input=True).estimate(frame).depth

    np.testing.assert_allclose(b, a, atol=1e-4)


def test_lane_stage_gives_the_same_probabilities_with_a_raw_input_model(tmp_path, frame):
    net = TinySeg()
    plain = save(net, torch.zeros(1, 3, 36, 64), tmp_path / "plain.ts")
    example = torch.zeros(1, 36, 64, 3, dtype=torch.uint8)
    raw = save(RawImageInput(net, probabilities=True), example, tmp_path / "raw.ts")

    a = SegLaneDetector(plain, input_size=(64, 36), device="cpu").segment(frame.image)
    b = SegLaneDetector(raw, input_size=(64, 36), device="cpu", raw_input=True).segment(frame.image)

    for pa, pb in zip(a, b, strict=True):
        np.testing.assert_allclose(pb, pa, atol=1e-5)
