"""Export an ImageNet backbone as an appearance embedding model for DeepSORT (classifier removed).

    python scripts/export_reid.py                              # resnet18 -> weights/reid_resnet18.{ts,onnx}
    python scripts/export_reid.py --arch mobilenet_v3_small

Writes TorchScript (for TorchBackend) and ONNX (for ONNX Runtime / TensorRT), both with
input "images" (N, 3, H, W) float32 RGB ImageNet-normalised and output "features" (N, D), not normalised.
ImageNet features are generic, not trained for re-identification; a ReID model can be exported to the
same format and swapped in through the tracker's reid_model setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision
from torch import nn

# Attribute holding each architecture's classifier head
HEADS = {
    "resnet18": "fc",
    "resnet50": "fc",
    "mobilenet_v3_small": "classifier",
    "mobilenet_v3_large": "classifier",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch", choices=sorted(HEADS), default="resnet18")
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()

    model = getattr(torchvision.models, args.arch)(weights="DEFAULT")
    setattr(model, HEADS[args.arch], nn.Identity())
    model.eval()
    w, h = args.input_size
    example = torch.zeros(2, 3, h, w)
    with torch.inference_mode():
        dim = model(example).shape[1]

    out = args.out_dir / f"reid_{args.arch}"
    torch.jit.save(torch.jit.trace(model, example), f"{out}.ts")
    torch.onnx.export(
        model,
        (example,),
        f"{out}.onnx",
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "n"}, "features": {0: "n"}},
        opset_version=17,
    )
    print(f"wrote {out}.ts and {out}.onnx: (N, 3, {h}, {w}) -> (N, {dim})")


if __name__ == "__main__":
    main()
