"""Evaluate a lane detector on BDD100K val: ego-lane IoU against the drivable area, by condition.

    python scripts/eval_lanes.py                                          # classical, 2000 images
    python scripts/eval_lanes.py --config configs/lanes/classical_bdd100k.yaml top=0.3 --images 500
    python scripts/eval_lanes.py --save-examples 20 --no-wandb            # overlays in runs/eval_lanes/

Needs index/polygons.parquet (data/bdd100k_polygons.py). Images are sampled at random (fixed seed) and
each is a separate scene, so the detector is reset before every image. Metrics: perception.eval.lanes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from perception.core import registry
from perception.core.types import CameraInfo, Frame
from perception.eval.lanes import LaneEvaluator
from perception.io.bdd100k import IMAGE_HEIGHT, IMAGE_WIDTH
from perception.io.bdd100k_lanes import ego_lane_mask

CONDITIONS = {
    "timeofday": ["daytime", "night", "dawn/dusk"],
    "weather": ["clear", "rainy", "snowy", "overcast"],
}


def overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, result) -> np.ndarray:
    """Prediction in green, ground-truth ego lane outlined in red, fitted lines in yellow."""
    out = image.copy()
    out[pred] = (0.5 * out[pred] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 3)
    for fit in (result.left, result.right):
        if fit is not None:
            ys = np.arange(int(0.4 * out.shape[0]), out.shape[0])
            pts = np.c_[np.polyval(fit, ys), ys].astype(np.int32)
            cv2.polylines(out, [pts], False, (0, 255, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/lanes/classical_bdd100k.yaml"))
    parser.add_argument("--root", type=Path, default=Path("data/bdd100k"))
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-examples", type=int, default=0, help="write this many overlay images")
    parser.add_argument("--out", type=Path, default=Path("runs/eval_lanes"))
    parser.add_argument("--no-wandb", action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    )
    detector = registry.build("lanes", cfg)
    polygons = pd.read_parquet(args.root / "index" / "polygons.parquet")
    by_image = dict(tuple(polygons.groupby("image_id", observed=True)))
    images = pd.read_parquet(args.root / "index" / "images.parquet")
    images = images[images.split == "val"].sample(n=args.images, random_state=args.seed)
    camera = CameraInfo.from_fov(IMAGE_WIDTH, IMAGE_HEIGHT, 90.0, 1.4)  # not used by image-only detectors
    no_polygons = polygons.iloc[:0]

    evaluators = {"all": LaneEvaluator()}
    for attr, values in CONDITIONS.items():
        evaluators.update({f"{attr}={v}": LaneEvaluator() for v in values})
    if args.save_examples:
        (args.out / "examples").mkdir(parents=True, exist_ok=True)

    elapsed = 0.0
    for n, image_row in enumerate(images.itertuples(index=False)):
        image = cv2.imread(str(args.root / "images" / "100k" / "val" / f"{image_row.image_id}.jpg"))
        frame = Frame(image=image, timestamp=0.0, frame_id=n, camera=camera)
        detector.reset()
        start = time.perf_counter()
        result = detector.detect(frame)
        elapsed += time.perf_counter() - start

        rows = by_image.get(image_row.image_id, no_polygons)
        gt = ego_lane_mask(rows)
        pred = result.ego_lane_mask if result.ego_lane_mask is not None else np.zeros_like(gt)
        groups = ["all"] + [f"{a}={getattr(image_row, a)}" for a in CONDITIONS]
        for group in groups:
            if group in evaluators:
                evaluators[group].update_ego(pred, gt)
        if n < args.save_examples:
            cv2.imwrite(
                str(args.out / "examples" / f"{image_row.image_id}.jpg"), overlay(image, pred, gt, result)
            )

    results = []
    for group, ev in evaluators.items():
        m = ev.summary()
        results.append({"group": group, **{k: v for k, v in m.items() if k.startswith("ego")}})
    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"{1000 * elapsed / len(images):.1f} ms per image")

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="eval",
            group="lanes",
            name=f"eval-lanes-{cfg['type']}",
            config={**vars(args), "detector": cfg},
            tags=["phase3", "lanes"],
        )
        run.log({"eval/lanes": wandb.Table(dataframe=table), "ms_per_image": 1000 * elapsed / len(images)})
        run.finish()


if __name__ == "__main__":
    main()
