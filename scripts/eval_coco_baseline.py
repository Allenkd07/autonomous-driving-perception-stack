"""Baseline: evaluate COCO-pretrained YOLO models on BDD100K val, before any fine-tuning.

    python scripts/eval_coco_baseline.py                          # yolo11n + yolo11s, all/day/night val
    python scripts/eval_coco_baseline.py --models yolo11s.pt --no-wandb

The pretrained models predict COCO's 80 classes, so the val labels are rewritten with COCO class
ids (into <yolo>/../yolo_coco/) and scored with Ultralytics' standard validator, the same code
that scores the fine-tuned model later. `rider` is scored as COCO `person` (COCO has no rider
class and labels riders as people); `traffic sign` has no COCO counterpart and is left out.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from perception.io.bdd100k_yolo import CLASSES

# Our class -> COCO class id
TO_COCO = {
    "car": 2,
    "truck": 7,
    "bus": 5,
    "person": 0,
    "rider": 0,
    "bicycle": 1,
    "motorcycle": 3,
    "traffic light": 9,
}
COCO_CLASSES = sorted(set(TO_COCO.values()))


def build_coco_val(yolo: Path, out: Path, coco_names: dict[int, str]) -> Path:
    """Write val labels with COCO ids next to a link to the val images; return the dataset yaml."""
    id_map = {i: TO_COCO[name] for i, name in enumerate(CLASSES) if name in TO_COCO}
    lbl_dir = out / "labels" / "val"
    if out.exists():
        shutil.rmtree(out)
    lbl_dir.mkdir(parents=True)
    (out / "images").mkdir()
    (out / "images" / "val").symlink_to((yolo / "images" / "val").resolve())

    for src in sorted((yolo / "labels" / "val").glob("*.txt")):
        lines = []
        for line in src.read_text().splitlines():
            cls, *box = line.split()
            if int(cls) in id_map:
                lines.append(" ".join([str(id_map[int(cls)]), *box]))
        (lbl_dir / src.name).write_text("\n".join(lines) + ("\n" if lines else ""))

    # Image lists, not the images/val folder: given a folder, Ultralytics resolves the symlink and
    # would read the labels next to the original images (our class ids) instead of these
    for name in ("val_day.txt", "val_night.txt"):
        shutil.copy(yolo / name, out / name)
    (out / "val_all.txt").write_text(
        "".join(f"./images/val/{p.stem}.jpg\n" for p in sorted(lbl_dir.glob("*.txt")))
    )

    names = "\n".join(f"  {i}: {n}" for i, n in coco_names.items())
    yaml = out / "bdd100k_coco.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: val_all.txt\nval: val_all.txt\nnames:\n{names}\n")
    return yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yolo", type=Path, default=Path("data/bdd100k/yolo"))
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    args.weights_dir.mkdir(exist_ok=True)
    first = YOLO(args.weights_dir / args.models[0])
    out = args.yolo.parent / "yolo_coco"
    yaml = build_coco_val(args.yolo, out, first.names)

    results = []
    for model_name in args.models:
        model = YOLO(args.weights_dir / model_name)
        for subset in ("all", "day", "night"):
            # Same dataset, with val pointing at the image list for this subset
            data = out / f"bdd100k_coco_{subset}.yaml"
            data.write_text(yaml.read_text().replace("val: val_all.txt", f"val: val_{subset}.txt"))
            metrics = model.val(
                data=str(data),
                imgsz=args.imgsz,
                batch=args.batch,
                classes=COCO_CLASSES,
                project=str(Path("runs/baseline").resolve()),  # relative paths get nested under runs/detect/
                name=f"{Path(model_name).stem}_{subset}",
                exist_ok=True,
                plots=subset == "all",
                verbose=False,
            )
            per_class = {model.names[c]: float(metrics.box.maps[c]) for c in COCO_CLASSES}
            results.append(
                {
                    "model": Path(model_name).stem,
                    "subset": subset,
                    "mAP50": float(metrics.box.map50),
                    "mAP50-95": float(metrics.box.map),
                    **{f"AP50-95/{k}": v for k, v in per_class.items()},
                }
            )
            print(f"{model_name} {subset}: mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")

    if not args.no_wandb:
        import pandas as pd
        import wandb

        run = wandb.init(
            project="perception-stack", job_type="eval", name="coco-baseline", tags=["phase2", "baseline"]
        )
        run.use_artifact("bdd100k-yolo:latest")
        run.log({"baseline/results": wandb.Table(dataframe=pd.DataFrame(results))})
        run.finish()


if __name__ == "__main__":
    main()
