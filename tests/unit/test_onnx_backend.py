import numpy as np
import pytest
import torch

from perception.backends.onnx_backend import OnnxBackend
from perception.core import registry
from perception.core.interfaces import InferenceBackend

pytest.importorskip("onnxruntime")


class MeanAndMax(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3)), images.amax(dim=(1, 2, 3))


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.onnx"
    torch.onnx.export(
        MeanAndMax(),
        (torch.zeros(1, 3, 4, 4),),
        str(path),
        input_names=["images"],
        output_names=["mean", "max"],
        dynamic_axes={"images": {0: "n"}, "mean": {0: "n"}, "max": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def test_runs_onnx_and_names_outputs(model_file):
    backend = OnnxBackend(model_file, outputs=("mean", "max"), device="cpu")
    images = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)

    out = backend.run({"images": images})

    assert isinstance(backend, InferenceBackend)
    np.testing.assert_allclose(out["mean"], images.mean(axis=(2, 3)), rtol=1e-6)
    np.testing.assert_allclose(out["max"], images.max(axis=(1, 2, 3)))
    assert out["mean"].dtype == np.float32


def test_output_order_follows_the_given_names(model_file):
    backend = OnnxBackend(model_file, outputs=("max",), device="cpu")

    out = backend.run({"images": np.ones((1, 3, 4, 4), np.float32)})

    assert list(out) == ["max"]


def test_unknown_tensor_name_is_an_error(model_file):
    with pytest.raises(ValueError, match="features"):
        OnnxBackend(model_file, outputs=("features",), device="cpu")


def test_half_is_refused(model_file):
    with pytest.raises(ValueError):
        OnnxBackend(model_file, outputs=("mean",), device="cpu", half=True)


def test_registered_as_onnx():
    assert registry.get("backend", "onnx") is OnnxBackend
