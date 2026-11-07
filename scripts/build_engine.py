"""Build a TensorRT engine from an ONNX export, for the tensorrt backend.

    python scripts/build_engine.py weights/twinlitenetplus_medium.onnx
    python scripts/build_engine.py weights/depth_anything_v2_small_relative_686x518.onnx --precision fp32
    python scripts/build_engine.py weights/reid_resnet18.onnx --batch 1 16 64

Writes <onnx stem>_<precision>.engine next to the ONNX file (or --out). The engine is specific to this GPU
and TensorRT version, so build it on the machine that runs it; building takes minutes, since TensorRT times
candidate kernels for every layer. YOLO goes through Ultralytics' own export instead (format=engine), which
keeps the metadata its predictor needs.

Inputs with a dynamic batch (ReID crops) need an optimisation profile: --batch MIN OPT MAX. TensorRT tunes
its kernels for OPT and accepts any batch from MIN to MAX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from perception.backends.tensorrt_backend import build_engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--batch", type=int, nargs=3, default=(1, 8, 32), metavar=("MIN", "OPT", "MAX"))
    parser.add_argument("--workspace-gb", type=float, default=1.0, help="scratch memory for the builder")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.onnx.with_name(f"{args.onnx.stem}_{args.precision}.engine")
    build_engine(args.onnx, out, args.precision, tuple(args.batch), args.workspace_gb)


if __name__ == "__main__":
    main()
