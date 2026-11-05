"""Evaluate lane detectors on CARLA recordings (carla_scripts/record_sequence.py) against CARLA tags.

    python scripts/eval_lanes_carla.py data/carla/town10_noon_s0 data/carla/town05_clearnoon_s2 --calibrate
    python scripts/eval_lanes_carla.py data/carla/town10_noon_s0 data/carla/town05_clearnoon_s2 \\
        --configs configs/lanes/classical_carla.yaml configs/lanes/segmentation_carla.yaml
    python scripts/eval_lanes_carla.py data/carla/* --configs configs/lanes/segmentation_carla.yaml \\
        model=weights/twinlitenetplus_medium.ts             # key=value overrides apply to every config

CARLA labels road (tag 1) and painted road lines (tag 24) per pixel, but not which lane is the ego lane, so:
- lane lines and drivable area (segmentation detector only): the network's masks against the tags, i.e.
  how well a model trained on real BDD100K dashcams transfers to CARLA's rendering
- ego lane: against an oracle, SegLaneDetector's rows geometry run on the ground-truth tag masks, scored
  only on frames where the oracle finds both lines. The oracle shares the segmentation detector's geometry,
  so this measures the network's masks (and the classical detector as a whole), not that geometry.
--calibrate prints the ego-lane width at the bottom row measured from the oracle's lines, for the detectors'
lane_width (the vanishing point follows from the camera: pitch 0 puts it at the image centre).

Recordings are video, so each detector is reset per recording and keeps its temporal smoothing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from perception.core import registry
from perception.eval.lanes import LaneEvaluator
from perception.io.carla_recording import CarlaRecording

ROAD_TAG, ROAD_LINE_TAG = 1, 24


def ground_truth(rec: CarlaRecording, frames: range) -> list[tuple[np.ndarray, np.ndarray]]:
    """(lane-line paint, road) bool masks per frame; paint is on the road, so it counts as road."""
    out = []
    for i in frames:
        tag, _ = rec.instance(i)
        paint = tag == ROAD_LINE_TAG
        out.append((paint, paint | (tag == ROAD_TAG)))
    return out


def calibrate(oracle, truths: list[tuple[np.ndarray, np.ndarray]], bottom: float) -> None:
    """Print the ego-lane width at the bottom row, from the oracle's lines on frames where it finds both."""
    widths = []
    for paint, road in truths:
        oracle.reset()
        r = oracle.lanes_from_masks(paint, road)
        if r.confidence == 1.0:
            y = bottom * paint.shape[0]
            widths.append((np.polyval(r.right, y) - np.polyval(r.left, y)) / paint.shape[1])
    if not widths:
        print("oracle found both lines in no frame")
        return
    q1, median, q3 = np.percentile(widths, [25, 50, 75])
    print(
        f"oracle frames with both lines: {len(widths)}/{len(truths)}; "
        f"lane width at row {bottom}: median {median:.3f} (IQR {q1:.3f}-{q3:.3f}) of image width"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--configs", nargs="+", type=Path, default=[])
    parser.add_argument(
        "--oracle-config",
        type=Path,
        default=Path("configs/lanes/segmentation_carla.yaml"),
        help="geometry for the oracle (its model is ignored)",
    )
    parser.add_argument("--step", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args, overrides = parser.parse_known_args()
    # nargs="+" lists also swallow key=value overrides: move them out wherever they were given
    overrides += [str(a) for a in [*args.recordings, *args.configs] if "=" in str(a)]
    args.recordings = [a for a in args.recordings if "=" not in str(a)]
    args.configs = [a for a in args.configs if "=" not in str(a)]

    oracle_cfg = {**OmegaConf.to_container(OmegaConf.load(args.oracle_config)), "model": None}
    oracle = registry.build("lanes", oracle_cfg)
    recordings = [CarlaRecording(path) for path in args.recordings]
    truths = [ground_truth(rec, range(0, len(rec), args.step)) for rec in recordings]

    if args.calibrate:
        calibrate(oracle, [t for ts in truths for t in ts], oracle_cfg.get("bottom", 0.9))
        if not args.configs:
            return

    oracle_masks = []
    for ts in truths:
        oracle.reset()
        masks = []
        for paint, road in ts:
            r = oracle.lanes_from_masks(paint, road)
            masks.append(r.ego_lane_mask if r.confidence == 1.0 else None)
        oracle_masks.append(masks)

    rows = []
    for config in args.configs:
        cfg = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.load(config), OmegaConf.from_dotlist(overrides))
        )
        detector = registry.build("lanes", cfg)
        ev = LaneEvaluator()
        inter = union = 0
        elapsed, n = 0.0, 0
        for rec, ts, masks in zip(recordings, truths, oracle_masks, strict=True):
            detector.reset()
            for (paint, road), gt_ego, i in zip(ts, masks, range(0, len(rec), args.step), strict=True):
                frame = rec.frame(i, with_depth=False)
                start = time.perf_counter()
                result = detector.detect(frame)
                elapsed += time.perf_counter() - start
                n += 1
                if gt_ego is not None:
                    pred = result.ego_lane_mask if result.ego_lane_mask is not None else np.zeros_like(gt_ego)
                    ev.update_ego(pred, gt_ego)
                lanes = getattr(detector, "lanes", None)
                if lanes is not None:
                    ev.update_markings(lanes >= detector.lane_threshold, paint)
                    drivable = detector.drivable >= detector.drivable_threshold
                    inter += int((drivable & road).sum())
                    union += int((drivable | road).sum())
        m = ev.summary()
        model = Path(cfg["model"]).stem if cfg.get("model") else None
        rows.append(
            {
                "detector": config.stem + (f" ({model})" if model else ""),
                "ego_iou": m["ego_iou"],
                "ego_precision": m["ego_precision"],
                "ego_recall": m["ego_recall"],
                "ego_frames": m["ego_images"],
                "line_precision": m["marking_precision"] if union else float("nan"),
                "line_recall": m["marking_recall"] if union else float("nan"),
                "line_f1": m["marking_f1"] if union else float("nan"),
                "drivable_iou": inter / union if union else float("nan"),
                "ms_per_frame": 1000 * elapsed / max(n, 1),
            }
        )
    if not rows:
        return
    table = pd.DataFrame(rows)
    print(f"{sum(len(t) for t in truths)} frames from {len(recordings)} recordings")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="eval",
            group="lanes",
            name="eval-lanes-carla-"
            + "-".join(Path(c).stem.removesuffix("_carla") for c in args.configs)
            + "".join(
                f"-{o.split('=', 1)[1].rsplit('/', 1)[-1].removesuffix('.ts')}"
                for o in overrides
                if o.startswith("model=")
            ),
            config={**vars(args), "recordings": [str(r) for r in args.recordings], "overrides": overrides},
            tags=["phase3", "lanes", "carla"],
        )
        run.log({"eval/lanes_carla": wandb.Table(dataframe=table)})
        run.finish()


if __name__ == "__main__":
    main()
