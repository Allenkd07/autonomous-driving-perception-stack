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
    half     FP16 on CUDA; inputs and outputs stay float32 numpy arrays
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
        module = torch.jit.load(str(model), map_location=self.device).eval()
        self.model = module.half() if self.half else module
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self._torch
        args = []
        for name in self.inputs:
            t = torch.from_numpy(np.ascontiguousarray(inputs[name])).to(self.device)
            args.append(t.half() if self.half and t.is_floating_point() else t)
        with torch.inference_mode():
            out = self.model(*args)
        if isinstance(out, torch.Tensor):
            out = (out,)
        return {name: t.float().cpu().numpy() for name, t in zip(self.outputs, out, strict=True)}
