"""Evaluate trained detectors on BDD100K val: all, day and night, per class.

    python scripts/eval_detector.py                                   # weights/yolo11s_bdd100k.pt
    python scripts/eval_detector.py --weights yolo11s_bdd100k.pt yolo11s_bdd100k.engine --no-wandb

Uses Ultralytics' standard validator on our 9-class labels, so any format Ultralytics can load
(.pt, .onnx, .engine) is scored the same way. Also reports the mean AP50-95 over the 7 classes the
COCO baseline can score (scripts/eval_coco_baseline.py), for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Classes the COCO baseline can score (no rider or traffic sign)
SHARED_WITH_COCO = ["car", "truck", "bus", "person", "bicycle", "motorcycle", "traffic light"]
SUBSETS = {"all": "images/val", "day": "val_day.txt", "night": "val_night.txt"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yolo", type=Path, default=Path("data/bdd100k/yolo"))
    parser.add_argument("--weights", nargs="+", default=["yolo11s_bdd100k.pt"])
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    base_yaml = (args.yolo / "bdd100k.yaml").read_text()
    results = []
    for weights in args.weights:
        model = YOLO(args.weights_dir / weights)
        for subset, val in SUBSETS.items():
            # Same dataset, with val pointing at this subset; removed again after scoring
            data = args.yolo / f"bdd100k_eval_{subset}.yaml"
            data.write_text(base_yaml.replace("val: images/val", f"val: {val}"))
            try:
                metrics = model.val(
                    data=str(data),
                    imgsz=args.imgsz,
                    batch=args.batch,
                    project=str(Path("runs/eval").resolve()),  # relative paths get nested under runs/detect/
                    name=f"{Path(weights).stem}_{subset}",
                    exist_ok=True,
                    plots=subset == "all",
                    verbose=False,
                )
            finally:
                data.unlink()
            per_class = {name: float(metrics.box.maps[c]) for c, name in model.names.items()}
            shared = sum(per_class[c] for c in SHARED_WITH_COCO) / len(SHARED_WITH_COCO)
            results.append(
                {
                    "model": Path(weights).name,
                    "subset": subset,
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "mAP50": float(metrics.box.map50),
                    "mAP50-95": float(metrics.box.map),
                    "mAP50-95/shared_with_coco": shared,
                    **{f"AP50-95/{k}": v for k, v in per_class.items()},
                }
            )
            print(
                f"{weights} {subset}: mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}"
                f"  (7 COCO classes {shared:.3f})"
            )

    if not args.no_wandb:
        import pandas as pd
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="eval",
            name=f"eval-{'-'.join(Path(w).stem for w in args.weights)}",
            config=vars(args) | {"yolo": str(args.yolo), "weights_dir": str(args.weights_dir)},
            tags=["phase2", "eval"],
        )
        run.use_artifact("bdd100k-yolo:latest")
        for weights in args.weights:
            # Link the detector artifact logged by train_detector.py, when there is one
            try:
                run.use_artifact(f"{Path(weights).stem}:latest")
            except wandb.errors.CommError:
                pass
        run.log({"eval/results": wandb.Table(dataframe=pd.DataFrame(results))})
        run.finish()


if __name__ == "__main__":
    main()
