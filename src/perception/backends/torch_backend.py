"""PyTorch backend: runs a TorchScript model, so the same model file works without its Python class."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from perception.core.registry import register


@register("backend", "torch")
class TorchBackend:
    """TorchScript model behind the InferenceBackend interface.

    model    TorchScript file (torch.jit.save)
    inputs   input names, in the order of the module's forward() arguments
    outputs  output names, in the order of its return values
    device   "cpu", "cuda", "cuda:0", ...; None picks CUDA when available
    half     FP16 on CUDA through autocast (each op picks FP16 or FP32, as PyTorch recommends); inputs and
             outputs stay float32 numpy arrays. Not model.half(): traced models keep the float32 casts of
             their trace (Depth Anything's position embeddings) and then fail on FP16 activations.
    """

    def __init__(
        self,
        model: str | Path,
        inputs: Sequence[str] = ("images",),
        outputs: Sequence[str] = ("features",),
        device: str | None = None,
        half: bool = False,
    ) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = half and self.device.type == "cuda"
        self.model = torch.jit.load(str(model), map_location=self.device).eval()
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self._torch
        args = [torch.from_numpy(np.ascontiguousarray(inputs[name])).to(self.device) for name in self.inputs]
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=self.half):
            out = self.model(*args)
        if isinstance(out, torch.Tensor):
            out = (out,)
        return {name: t.float().cpu().numpy() for name, t in zip(self.outputs, out, strict=True)}
