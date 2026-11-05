"""Export MiDaS v2.1 Small (relative monocular depth) for the depth stage (perception.depth.network).

    git clone --depth 1 https://github.com/isl-org/MiDaS ~/external/MiDaS
    git clone --depth 1 https://github.com/rwightman/gen-efficientnet-pytorch \\
        ~/external/gen-efficientnet-pytorch
    curl -L -o ~/external/midas_weights/midas_v21_small_256.pt \\
        https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
    python scripts/export_midas.py --width 256 --height 192     # 4:3 (CARLA); 16:9 (BDD100K): 256 x 160

Writes weights/midas_small_<w>x<h>.{ts,onnx}: input "images" (1, 3, h, w) float32 RGB normalised with the
ImageNet mean and std; output "depth" (1, h, w), relative inverse depth (larger = nearer, known only up to
a scale and shift per image). MiDaS was trained at 256 px on the longer side keeping the aspect ratio, with
both sides multiples of 32, so the input size is fixed per camera aspect (static shapes also suit TensorRT).
The model code is imported from the cloned repos (MIT licence), so it isn't vendored here; the EfficientNet
backbone, which MiDaS fetches with torch.hub, is loaded from the local clone instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument(
        "--weights", type=Path, default=Path.home() / "external" / "midas_weights" / "midas_v21_small_256.pt"
    )
    parser.add_argument("--repo", type=Path, default=Path.home() / "external" / "MiDaS")
    parser.add_argument(
        "--backbone-repo", type=Path, default=Path.home() / "external" / "gen-efficientnet-pytorch"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()
    if args.width % 32 or args.height % 32:
        parser.error("width and height must be multiples of 32")

    hub_load = torch.hub.load

    def local_hub_load(repo: str, model: str, *a, **kw):
        if repo == "rwightman/gen-efficientnet-pytorch":  # ImageNet weights not needed: MiDaS's replace them
            return hub_load(str(args.backbone_repo), model, *a, **{**kw, "pretrained": False}, source="local")
        raise RuntimeError(f"unexpected torch.hub download: {repo}")

    torch.hub.load = local_hub_load
    sys.path.insert(0, str(args.repo))  # the repo's `midas` package
    from midas.midas_net_custom import MidasNet_small

    model = MidasNet_small(
        None, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True,
        blocks={"expand": True},
    )  # fmt: skip
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))
    model.eval()

    example = torch.rand(1, 3, args.height, args.width)
    with torch.inference_mode():
        depth = model(example)
    assert depth.shape == (1, args.height, args.width), depth.shape

    out = args.out_dir / f"midas_small_{args.width}x{args.height}"
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
