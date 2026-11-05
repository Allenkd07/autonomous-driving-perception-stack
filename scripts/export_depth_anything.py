"""Export Depth Anything V2 Small (relative, or metric fine-tuned on Virtual KITTI 2) for the depth stage.

    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 ~/external/Depth-Anything-V2
    # Apache-2.0 weights from Hugging Face (depth-anything/Depth-Anything-V2-Small and
    # depth-anything/Depth-Anything-V2-Metric-VKITTI-Small) into ~/external/depth_anything_weights/
    python scripts/export_depth_anything.py --kind relative --width 686 --height 518
    python scripts/export_depth_anything.py --kind metric --width 686 --height 518

Writes weights/depth_anything_v2_small_<kind>_<w>x<h>.{ts,onnx}: input "images" (1, 3, h, w) float32 RGB
normalised with the ImageNet mean and std; output "depth" (1, h, w):
    relative   inverse depth up to a scale and shift per image, like MiDaS (larger = nearer)
    metric     depth in metres, up to --max-depth (80 m for the Virtual KITTI model)
The model was trained with the shorter side at 518 px and both sides multiples of 14 (its ViT patch size):
686 x 518 for a 4:3 camera; 518 x 392 is a faster, coarser choice. The model code is imported from the
cloned repo (Apache-2.0), so it isn't vendored here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

WEIGHTS = {
    "relative": "depth_anything_v2_vits.pth",
    "metric": "depth_anything_v2_metric_vkitti_vits.pth",
}
SMALL = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kind", choices=list(WEIGHTS), default="relative")
    parser.add_argument("--width", type=int, default=686)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=80.0, help="metric model's depth range")
    parser.add_argument(
        "--weights-dir", type=Path, default=Path.home() / "external" / "depth_anything_weights"
    )
    parser.add_argument("--repo", type=Path, default=Path.home() / "external" / "Depth-Anything-V2")
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()
    if args.width % 14 or args.height % 14:
        parser.error("width and height must be multiples of 14")

    # both variants define the package depth_anything_v2; the metric one lives under metric_depth/
    sys.path.insert(0, str(args.repo / "metric_depth" if args.kind == "metric" else args.repo))
    from depth_anything_v2.dpt import DepthAnythingV2

    kwargs = {**SMALL, "max_depth": args.max_depth} if args.kind == "metric" else SMALL
    model = DepthAnythingV2(**kwargs)
    state = torch.load(args.weights_dir / WEIGHTS[args.kind], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    example = torch.rand(1, 3, args.height, args.width)
    with torch.inference_mode():
        depth = model(example)
    assert depth.shape == (1, args.height, args.width), depth.shape

    out = args.out_dir / f"depth_anything_v2_small_{args.kind}_{args.width}x{args.height}"
    out.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, f"{out}.ts")
    with torch.inference_mode():
        diff = (traced(example) - depth).abs().max().item()
    torch.onnx.export(
        model, (example,), f"{out}.onnx", input_names=["images"], output_names=["depth"], opset_version=17
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"wrote {out}.ts and {out}.onnx ({params / 1e6:.2f}M params, trace max diff {diff:.1e})")


if __name__ == "__main__":
    main()
