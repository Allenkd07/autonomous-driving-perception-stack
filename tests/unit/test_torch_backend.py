import numpy as np
import pytest
import torch

from perception.backends.torch_backend import TorchBackend
from perception.core import registry
from perception.core.interfaces import InferenceBackend


class MeanAndMax(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3)), images.amax(dim=(1, 2, 3))


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.ts"
    torch.jit.save(torch.jit.script(MeanAndMax()), str(path))
    return path


def test_runs_torchscript_and_names_outputs(model_file):
    backend = TorchBackend(model_file, outputs=("mean", "max"), device="cpu")
    images = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)

    out = backend.run({"images": images})

    assert isinstance(backend, InferenceBackend)
    np.testing.assert_allclose(out["mean"], images.mean(axis=(2, 3)))
    np.testing.assert_allclose(out["max"], images.max(axis=(1, 2, 3)))
    assert out["mean"].dtype == np.float32


def test_wrong_number_of_output_names_is_an_error(model_file):
    backend = TorchBackend(model_file, outputs=("features",), device="cpu")

    with pytest.raises(ValueError):
        backend.run({"images": np.zeros((1, 3, 4, 4), np.float32)})


class FixedCast(torch.nn.Module):
    """A float32 cast baked in, as a trace records it (Depth Anything's position embeddings)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
        self.offset = torch.nn.Parameter(torch.ones(1, 2, 1, 1))

    def forward(self, images):
        return self.conv(images) + self.offset.to(torch.float32)


@pytest.mark.gpu
def test_half_runs_a_model_with_float32_casts(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    path = tmp_path / "cast.ts"
    model = FixedCast().eval()
    torch.jit.save(torch.jit.trace(model, torch.zeros(1, 3, 4, 4)), str(path))
    images = np.random.default_rng(0).standard_normal((1, 3, 4, 4)).astype(np.float32)

    out = TorchBackend(path, outputs=("y",), device="cuda", half=True).run({"images": images})

    with torch.no_grad():
        expected = model(torch.from_numpy(images)).numpy()
    assert out["y"].dtype == np.float32
    np.testing.assert_allclose(out["y"], expected, atol=1e-2)


def test_registered_as_torch():
    assert registry.get("backend", "torch") is TorchBackend
