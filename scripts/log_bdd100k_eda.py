"""Log the BDD100K EDA summary and the YOLO dataset version to Weights & Biases.

    python scripts/log_bdd100k_eda.py

Needs the label index (data/bdd100k_index.py) and the YOLO dataset (data/bdd100k_to_yolo.py).
Logs one run (job_type "eda") with per-class and per-condition tables and bar charts, and a
`bdd100k-yolo` dataset artifact holding the YOLO config and the exact train/val image lists,
so training runs can record which dataset version they used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb

from perception.io.bdd100k_yolo import CLASSES, RARE_CLASSES

SCALE_640 = 640 / 1280


def class_stats(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    h = (boxes.y2 - boxes.y1) * SCALE_640
    area = (boxes.x2 - boxes.x1) * h * SCALE_640
    b = boxes.assign(h=h, small=area < 32**2)
    stats = b.groupby("category", observed=True).agg(
        boxes=("h", "size"),
        images=("image_id", "nunique"),
        small_pct=("small", "mean"),
        median_height_px=("h", "median"),
        occluded_pct=("occluded", "mean"),
        truncated_pct=("truncated", "mean"),
    )
    stats["images_pct"] = stats.images / len(images)
    pct = ["small_pct", "occluded_pct", "truncated_pct", "images_pct"]
    stats[pct] = (stats[pct] * 100).round(1)
    stats["median_height_px"] = stats.median_height_px.round(1)
    return stats.sort_values("boxes", ascending=False).reset_index()


def share_table(series: pd.Series, name: str) -> wandb.Table:
    share = (series.value_counts(normalize=True) * 100).round(1)
    return wandb.Table(dataframe=share.rename_axis(name).reset_index(name="percent"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/bdd100k"))
    parser.add_argument("--project", default="perception-stack")
    args = parser.parse_args()

    images = pd.read_parquet(args.root / "index" / "images.parquet")
    boxes = pd.read_parquet(args.root / "index" / "boxes.parquet")
    train_images = images[images.split == "train"]
    train_boxes = boxes[boxes.split == "train"]
    yolo = args.root / "yolo"

    run = wandb.init(project=args.project, job_type="eda", name="bdd100k-eda", tags=["bdd100k", "phase1"])

    stats = class_stats(train_images, train_boxes)
    stats_table = wandb.Table(dataframe=stats)
    run.log(
        {
            "classes/table": stats_table,
            "classes/boxes": wandb.plot.bar(
                stats_table, "category", "boxes", title="Boxes per class (train)"
            ),
            "classes/small_pct": wandb.plot.bar(
                stats_table, "category", "small_pct", title="Small boxes at 640 px (% of class)"
            ),
        }
    )
    for col in ("timeofday", "weather", "scene"):
        table = share_table(train_images[col], col)
        run.log(
            {f"conditions/{col}": wandb.plot.bar(table, col, "percent", title=f"{col} (% of train images)")}
        )

    split_share = boxes.groupby(["split", "category"], observed=True).size().unstack("split")
    split_share = split_share / split_share.sum() * 100
    small = ((boxes.x2 - boxes.x1) * (boxes.y2 - boxes.y1) * SCALE_640**2 < 32**2).mean()
    run.summary.update(
        {
            "images": len(images),
            "boxes": len(boxes),
            "small_box_pct_640": round(float(small) * 100, 1),
            "night_image_pct": round(float((train_images.timeofday == "night").mean()) * 100, 1),
            "max_train_val_class_share_gap_pct": round(
                float((split_share.train - split_share.val).abs().max()), 2
            ),
        }
    )

    train_ids = sorted(p.stem for p in (yolo / "labels" / "train").glob("*.txt"))
    list_file = yolo / "train_images.txt"
    list_file.write_text("".join(f"./images/train/{i}.jpg\n" for i in train_ids))
    artifact = wandb.Artifact(
        "bdd100k-yolo",
        type="dataset",
        description="BDD100K detection in YOLO format: train subset and full val (images not uploaded)",
        metadata={
            "classes": list(CLASSES),
            "train_images": len(train_ids),
            "val_images": int((images.split == "val").sum()),
            "rare_classes_fully_included": list(RARE_CLASSES),
            "input_index_boxes": len(boxes),
        },
    )
    for name in ("bdd100k.yaml", "train_images.txt", "val_day.txt", "val_night.txt"):
        artifact.add_file(str(yolo / name))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
