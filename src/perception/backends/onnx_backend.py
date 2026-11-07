"""ONNX Runtime backend: runs an exported .onnx model with the CUDA execution provider (or on the CPU)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from perception.core.registry import register


@register("backend", "onnx")
class OnnxBackend:
    """ONNX model behind the InferenceBackend interface, run by ONNX Runtime.

    model    .onnx file
    inputs   input names, as in the ONNX graph
    outputs  output names, as in the ONNX graph
    device   "cpu", "cuda", "cuda:0", ...; None picks CUDA when ONNX Runtime has it
    half     not supported: precision is fixed by the exported graph (FP16 is the TensorRT backend's job)
    """

    def __init__(
        self,
        model: str | Path,
        inputs: Sequence[str] = ("images",),
        outputs: Sequence[str] = ("features",),
        device: str | None = None,
        half: bool = False,
    ) -> None:
        import onnxruntime as ort

        if half:
            raise ValueError("the onnx backend runs the graph as exported; use the tensorrt backend for FP16")
        cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        if device is None:
            device = "cuda" if cuda else "cpu"
        if device.startswith("cuda"):
            if not cuda:
                raise RuntimeError("ONNX Runtime was built without CUDA (install onnxruntime-gpu)")
            # CUDA and cuDNN come from the nvidia-* pip packages that PyTorch installs
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            device_id = int(device.partition(":")[2] or 0)
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(str(model), options, providers=providers)
        self.device = device

        graph_inputs = {i.name for i in self.session.get_inputs()}
        graph_outputs = {o.name for o in self.session.get_outputs()}
        missing = (set(inputs) - graph_inputs) | (set(outputs) - graph_outputs)
        if missing:
            raise ValueError(
                f"{model} has no tensor(s) {sorted(missing)}: inputs {sorted(graph_inputs)}, "
                f"outputs {sorted(graph_outputs)}"
            )
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {name: np.ascontiguousarray(inputs[name]) for name in self.inputs}
        # session.run copies results back to the host, so the GPU work is finished when it returns
        values = self.session.run(list(self.outputs), feed)
        return {name: np.asarray(v, np.float32) for name, v in zip(self.outputs, values, strict=True)}
