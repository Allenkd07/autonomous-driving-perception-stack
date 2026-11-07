"""Export a trained YOLO detector to ONNX and a TensorRT engine through Ultralytics.

    python scripts/export_yolo.py weights/yolo11s_bdd100k_random.pt
    python scripts/export_yolo.py weights/yolo11s_bdd100k_random.pt --imgsz 384 640 --formats engine

Writes <stem>.onnx and <stem>.engine next to the weights. Ultralytics stores the class names and input size
in the files, so YoloDetector loads them like the .pt. The engine is FP16 and specific to this GPU and
TensorRT version.

Static input shape (h, w), 480 x 640 by default: CARLA's 800 x 600 camera letterboxes to exactly that, so no
compute goes into padding (a square 640 x 640 engine spends a quarter of its pixels on grey bars). BDD100K's
16:9 frames still fit (scaled by the long side as with the .pt, padded top and bottom), so mAP is
comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    # engine first: Ultralytics goes through its own ONNX export for it, which the onnx step then replaces
    parser.add_argument("--formats", nargs="+", choices=["onnx", "engine"], default=["engine", "onnx"])
    parser.add_argument("--workspace-gb", type=float, default=1.0)
    args = parser.parse_args()

    from ultralytics import YOLO

    for fmt in args.formats:
        options = {"half": True, "workspace": args.workspace_gb} if fmt == "engine" else {"simplify": True}
        path = YOLO(str(args.weights)).export(format=fmt, imgsz=list(args.imgsz), device=0, **options)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
