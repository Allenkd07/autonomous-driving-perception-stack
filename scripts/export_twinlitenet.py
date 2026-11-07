"""Export TwinLiteNetPlus (drivable area + lane line segmentation, pretrained on BDD100K) for SegLaneDetector.

    git clone --depth 1 https://github.com/chequanghuy/TwinLiteNetPlus ~/external/TwinLiteNetPlus
    # weights (nano/small/medium/large.pth): Google Drive folder linked from that README
    python scripts/export_twinlitenet.py --size medium --weights ~/external/twinlitenetplus_weights/medium.pth

Writes weights/twinlitenetplus_<size>.{ts,onnx}: input "images" (1, 3, 384, 640) float32 RGB in [0, 1]
(the image letterboxed to 640x360 plus 12 px grey rows above and below, as in training); outputs
"drivable" and "lanes", (1, 2, 384, 640) logits each (channel 1 = drivable area / lane line).
With --raw-input (files end in _raw, for TensorRT) the input is the letterboxed BGR uint8 image
(1, 384, 640, 3) and the outputs are the class-1 probabilities (1, 384, 640): conversion and softmax run
inside the model (perception.backends.raw_input); the lane stage needs raw_input: true.
The model code is imported from the cloned repo (MIT licence), so it isn't vendored here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from perception.backends.raw_input import RawImageInput

INPUT_H, INPUT_W = 384, 640


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", choices=["nano", "small", "medium", "large"], default="medium")
    parser.add_argument("--weights", type=Path, required=True, help="the size's .pth state dict")
    parser.add_argument("--repo", type=Path, default=Path.home() / "external" / "TwinLiteNetPlus")
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    parser.add_argument("--raw-input", action="store_true", help="BGR uint8 in, probabilities out")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))  # the repo's `model` package
    from model.model import TwinLiteNetPlus

    model = TwinLiteNetPlus(SimpleNamespace(config=args.size))
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}  # saved from DataParallel
    model.load_state_dict(state)
    model.eval()

    example = torch.rand(1, 3, INPUT_H, INPUT_W)
    shape = (1, 2, INPUT_H, INPUT_W)
    if args.raw_input:
        model = RawImageInput(model, probabilities=True).eval()
        example = torch.randint(0, 256, (1, INPUT_H, INPUT_W, 3), dtype=torch.uint8)
        shape = (1, INPUT_H, INPUT_W)
    with torch.inference_mode():
        drivable, lanes = model(example)
    assert drivable.shape == lanes.shape == shape, (drivable.shape, lanes.shape)

    out = args.out_dir / f"twinlitenetplus_{args.size}{'_raw' if args.raw_input else ''}"
    out.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, f"{out}.ts")
    with torch.inference_mode():
        diff = max(
            (a - b).abs().max().item() for a, b in zip(traced(example), (drivable, lanes), strict=True)
        )
    torch.onnx.export(
        model,
        (example,),
        f"{out}.onnx",
        input_names=["images"],
        output_names=["drivable", "lanes"],
        opset_version=17,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"wrote {out}.ts and {out}.onnx ({params / 1e6:.2f}M params, trace max diff {diff:.1e})")


if __name__ == "__main__":
    main()
