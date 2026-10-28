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


def test_registered_as_torch():
    assert registry.get("backend", "torch") is TorchBackend
