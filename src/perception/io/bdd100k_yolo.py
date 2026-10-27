"""Turn the BDD100K label index into a YOLO detection dataset.

YOLO format: one .txt per image, one line per box, `class_id cx cy w h` with all four numbers
normalised to 0-1 by the image size. Images without boxes get an empty file (background).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perception.io.bdd100k import IMAGE_HEIGHT, IMAGE_WIDTH

# Class id = position in this tuple. `train` is left out: 136 train-split boxes, none in CARLA.
CLASSES = (
    "car", "truck", "bus", "person", "rider", "bicycle", "motorcycle", "traffic light", "traffic sign"
)  # fmt: skip
RARE_CLASSES = ("bicycle", "rider", "motorcycle")
# BDD100K name -> class name. COCO's names, so that fine-tuning from COCO weights keeps what the
# model already knows about these classes (Ultralytics matches pretrained class rows by name).
RENAME = {"bike": "bicycle", "motor": "motorcycle"}
MIN_BOX_PX = 2.0


def clean_boxes(boxes: pd.DataFrame) -> pd.DataFrame:
    """Rename categories to class names and keep boxes of the training classes that are at least
    MIN_BOX_PX wide and tall."""
    category = boxes.category.astype(str).replace(RENAME)
    w = boxes.x2 - boxes.x1
    h = boxes.y2 - boxes.y1
    keep = category.isin(CLASSES) & (w >= MIN_BOX_PX) & (h >= MIN_BOX_PX)
    return boxes.assign(category=category)[keep]


def select_train_subset(
    image_ids: pd.Series,
    boxes: pd.DataFrame,
    size: int,
    rare_classes: tuple[str, ...] = RARE_CLASSES,
    seed: int = 0,
) -> list[str]:
    """Pick `size` images: every image with a rare class first, then a random fill.

    Rare-class images are otherwise too scarce to learn from (bicycle, rider and motorcycle each appear
    in 3-6% of images). If there are more rare-class images than `size`, a random `size` of them.
    """
    rng = np.random.default_rng(seed)
    all_ids = np.sort(image_ids.unique())
    rare = np.sort(boxes.loc[boxes.category.isin(rare_classes), "image_id"].unique())
    if len(rare) >= size:
        return sorted(rng.choice(rare, size=size, replace=False))

    rest = np.setdiff1d(all_ids, rare)
    fill = rng.choice(rest, size=min(size - len(rare), len(rest)), replace=False)
    return sorted(np.concatenate([rare, fill]))


def yolo_label_lines(boxes: pd.DataFrame) -> list[str]:
    """YOLO label lines for the boxes of one image."""
    # astype(str) first: on a categorical column, map() keeps unused categories (e.g. "train") as
    # NaN and turns the ids into floats ("8.0"), which YOLO rejects
    class_id = boxes.category.astype(str).map({name: i for i, name in enumerate(CLASSES)})
    if class_id.isna().any():
        raise ValueError(f"unknown classes: {sorted(set(boxes.category[class_id.isna()]))}")
    class_id = class_id.astype(int)
    cx = (boxes.x1 + boxes.x2) / 2 / IMAGE_WIDTH
    cy = (boxes.y1 + boxes.y2) / 2 / IMAGE_HEIGHT
    w = (boxes.x2 - boxes.x1) / IMAGE_WIDTH
    h = (boxes.y2 - boxes.y1) / IMAGE_HEIGHT
    return [
        f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"
        for c, x, y, bw, bh in zip(class_id, cx, cy, w, h, strict=True)
    ]
