"""Index the BDD100K 2018 labels into two Parquet tables for exploration and conversion.

    python data/bdd100k_index.py                      # train + val
    python data/bdd100k_index.py --splits val         # one split

Writes, under <root>/index/:
    images.parquet   one row per image: split, weather, scene, timeofday, n_boxes, n_lanes, n_drivable
    boxes.parquet    one row per box: split, image_id, category, x1, y1, x2, y2, occluded, truncated, ...
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from perception.io.bdd100k import parse_label_file


def index_split(root: Path, split: str, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((root / "labels" / "2018" / "100k" / split).glob("*.json"))
    if not files:
        raise SystemExit(f"no label files found for split '{split}' under {root}")

    images, boxes = [], []
    with Pool(workers) as pool:
        for image, image_boxes in pool.imap_unordered(parse_label_file, files, chunksize=256):
            images.append(image)
            boxes.extend(image_boxes)

    images_df = pd.DataFrame(images).assign(split=split)
    boxes_df = pd.DataFrame(boxes).assign(split=split)
    return images_df, boxes_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/bdd100k"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    all_images, all_boxes = [], []
    for split in args.splits:
        start = time.perf_counter()
        images_df, boxes_df = index_split(args.root, split, args.workers)
        elapsed = time.perf_counter() - start
        print(f"{split}: {len(images_df):,} images, {len(boxes_df):,} boxes ({elapsed:.0f} s)")
        all_images.append(images_df)
        all_boxes.append(boxes_df)

    images = pd.concat(all_images, ignore_index=True).sort_values(["split", "image_id"], ignore_index=True)
    boxes = pd.concat(all_boxes, ignore_index=True).sort_values(["split", "image_id"], ignore_index=True)
    for col in ("split", "weather", "scene", "timeofday"):
        images[col] = images[col].astype("category")
    for col in ("split", "category", "traffic_light_color"):
        boxes[col] = boxes[col].astype("category")

    out = args.root / "index"
    out.mkdir(exist_ok=True)
    images.to_parquet(out / "images.parquet", index=False)
    boxes.to_parquet(out / "boxes.parquet", index=False)
    print(f"wrote {out}/images.parquet and {out}/boxes.parquet")


if __name__ == "__main__":
    main()
