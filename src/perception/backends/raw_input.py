"""Fold a network's input normalisation (and a segmentation softmax) into the exported graph.

The stages then hand the backend the resized camera image as it is, BGR uint8 (1, h, w, 3), and the model
does the rest on the GPU: BGR -> RGB, HWC -> CHW, /255, mean/std. On the CPU that took a few milliseconds per
frame for the float conversion alone, and the float32 tensor is 4x the bytes to copy to the GPU. Under
TensorRT FP16 the extra ops fuse into the first layer. Resizing stays outside, so an engine doesn't depend on
the camera's resolution.

Used by the export scripts (--raw-input); the stages take such models with raw_input: true.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class RawImageInput(torch.nn.Module):
    """Wraps a network trained on RGB float input in [0, 1] (optionally mean/std normalised).

    mean, std      per-channel RGB values in [0, 1] (ImageNet's for the depth networks); None skips them
    probabilities  the wrapped network returns 2-class logits (N, 2, h, w), one tensor or several: return
                   the class-1 probability (N, h, w) of each instead (softmax over the 2 classes)
    """

    def __init__(
        self,
        net: torch.nn.Module,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        probabilities: bool = False,
    ) -> None:
        super().__init__()
        self.net = net
        self.normalise = mean is not None
        if self.normalise:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.probabilities = probabilities

    def forward(self, image: torch.Tensor):
        # (N, h, w, 3) BGR uint8 -> (N, 3, h, w) RGB float in [0, 1]. Cast first: TensorRT takes a uint8
        # input only straight into a cast (no uint8 intermediate tensors)
        x = image.float().permute(0, 3, 1, 2)[:, [2, 1, 0]] / 255.0
        if self.normalise:
            x = (x - self.mean) / self.std
        out = self.net(x)
        if not self.probabilities:
            return out
        if isinstance(out, torch.Tensor):
            return torch.sigmoid(out[:, 1] - out[:, 0])
        return tuple(torch.sigmoid(o[:, 1] - o[:, 0]) for o in out)
