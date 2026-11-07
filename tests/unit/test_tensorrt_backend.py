import numpy as np
import pytest
import torch

from perception.backends.tensorrt_backend import TensorRTBackend, build_engine
from perception.core import registry
from perception.core.interfaces import InferenceBackend

pytestmark = pytest.mark.gpu


class ConvAndMean(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, images):
        features = self.conv(images)
        return features, features.mean(dim=(2, 3))


@pytest.fixture(scope="module")
def onnx_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("trt") / "model.onnx"
    torch.onnx.export(
        ConvAndMean().eval(),
        (torch.zeros(1, 3, 8, 8),),
        str(path),
        input_names=["images"],
        output_names=["features", "mean"],
        dynamic_axes={"images": {0: "n"}, "features": {0: "n"}, "mean": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    return path


@pytest.fixture(scope="module")
def engine_file(onnx_file):
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return build_engine(onnx_file, onnx_file.with_suffix(".engine"), "fp32", (1, 2, 4), 0.25, verbose=False)


def test_matches_torch_with_a_dynamic_batch(onnx_file, engine_file):
    backend = TensorRTBackend(engine_file, outputs=("features", "mean"))
    model = ConvAndMean().eval()
    rng = np.random.default_rng(0)

    for n in (1, 3):  # any batch inside the profile's (1, 4) range
        images = rng.standard_normal((n, 3, 8, 8)).astype(np.float32)
        out = backend.run({"images": images})
        with torch.no_grad():
            features, mean = model(torch.from_numpy(images))

        assert isinstance(backend, InferenceBackend)
        assert out["features"].shape == (n, 4, 8, 8)
        assert out["features"].dtype == np.float32
        np.testing.assert_allclose(out["features"], features.numpy(), atol=1e-4)
        np.testing.assert_allclose(out["mean"], mean.numpy(), atol=1e-4)


def test_unknown_tensor_name_is_an_error(engine_file):
    with pytest.raises(ValueError, match="depth"):
        TensorRTBackend(engine_file, outputs=("depth",))


def test_registered_as_tensorrt():
    assert registry.get("backend", "tensorrt") is TensorRTBackend
