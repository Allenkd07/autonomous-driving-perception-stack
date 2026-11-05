"""Evaluate per-object distance estimation on CARLA recordings against the actors' 3D boxes.

    python scripts/eval_distance.py data/carla/town10_noon_s0 data/carla/town05_clearnoon_s2 \\
        --depth configs/depth/midas_small_carla.yaml
    python scripts/eval_distance.py data/carla/* --depth configs/depth/midas_small_carla.yaml \\
        --lanes configs/lanes/segmentation_carla.yaml fused.depth_inv_sigma=0.008 depth_scale.box_quantile=0.5

Inputs are the ground-truth boxes (each visible object's tight box), so the numbers measure distance
estimation alone, not the detector's misses or loose boxes. The truth is the forward distance to the
nearest point of the object's 3D box. Methods:

    ground_plane     box bottom on a flat road (no depth model)
    net_road         the depth network, calibrated per frame on the road's geometry (deployable)
    fused            ground_plane and net_road weighted by their expected errors (deployable)
    net_oracle       the network calibrated per frame on CARLA's depth: the best a per-frame scale can do
    depth_gt         CARLA's depth itself, through the same box statistic: the box statistic's own error
A metric network (Depth Anything V2 fine-tuned on Virtual KITTI) is used as it is, without calibration, by
net_road and fused; net_oracle still rescales it, to show what a per-frame correction could add.

Objects are split by whether the ground plane can see their contact with the road:
    clean       the visible box's bottom is the object's bottom
    occluded    its bottom is hidden behind another object (visible box bottom well above the 3D box's)
    truncated   cut off by the image's edge
The network's dense error (AbsRel over pixels up to 80 m) is also reported for each calibration.
key=value overrides set method parameters: depth.*, ground_plane.*, depth_scale.*, fused.*
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from perception.core import registry
from perception.core.types import DepthMap, Track
from perception.eval.distance import dense_abs_rel, distance_bin, distance_table
from perception.fusion.depth_scale import fit_inverse_depth, to_metric
from perception.io.carla_recording import CarlaRecording, GroundTruthObject

METHODS = ["ground_plane", "net_road", "fused", "net_oracle", "depth_gt"]
DENSE_MAX_M = 80.0
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def category(obj: GroundTruthObject, width: int, height: int, edge_px: float = 2.0) -> str:
    x1, _, x2, y2 = obj.box
    if x1 <= edge_px or x2 >= width - edge_px or y2 >= height - edge_px:
        return "truncated"
    amodal_bottom = obj.amodal_box[3]
    amodal_height = obj.amodal_box[3] - obj.amodal_box[1]
    if amodal_bottom - y2 > max(3.0, 0.05 * amodal_height):
        return "occluded"
    return "clean"


def build_methods(params: dict) -> dict:
    """Distance estimators per method, from the overrides' ground_plane / depth_scale / fused sections."""
    gp, ds, fused = (params.get(k, {}) for k in ("ground_plane", "depth_scale", "fused"))
    return {
        "ground_plane": registry.build("distance", {"type": "ground_plane", **gp}),
        "net_road": registry.build("distance", {"type": "depth_scale", **ds, "reference": "ground"}),
        "fused": registry.build(
            "distance",
            {"type": "fused", "ground_plane": gp, "depth_scale": {**ds, "reference": "ground"}, **fused},
        ),
        "net_oracle": registry.build("distance", {"type": "depth_scale", **ds, "reference": "gt"}),
        "depth_gt": registry.build("distance", {"type": "depth_scale", **ds}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--depth", type=Path, required=True, help="depth network config")
    parser.add_argument("--lanes", type=Path, help="lane config: its ego-lane mask places the road samples")
    parser.add_argument("--step", type=int, default=2, help="use every n-th frame")
    parser.add_argument("--min-visible-px", type=int, default=100)
    parser.add_argument("--min-distance-m", type=float, default=1.0, help="objects nearer (beside the car)")
    parser.add_argument("--out", type=Path, help="write the per-object table here (parquet)")
    parser.add_argument("--no-wandb", action="store_true")
    args, overrides = parser.parse_known_args()
    overrides += [str(a) for a in args.recordings if "=" in str(a)]
    args.recordings = [a for a in args.recordings if "=" not in str(a)]
    params = OmegaConf.to_container(OmegaConf.from_dotlist(overrides))

    depth_cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.depth), params.get("depth", {})))
    network = registry.build("depth", depth_cfg)
    lanes_detector = (
        registry.build("lanes", OmegaConf.to_container(OmegaConf.load(args.lanes))) if args.lanes else None
    )
    methods = build_methods(params)

    rows, dense = [], []
    net_ms = []
    for path in args.recordings:
        rec = CarlaRecording(path)
        for estimator in methods.values():
            estimator.reset()
        if lanes_detector is not None:
            lanes_detector.reset()
        for i in range(0, len(rec), args.step):
            frame = rec.frame(i)
            objects = [
                o
                for o in rec.ground_truth(i, min_visible_px=args.min_visible_px)
                if o.distance_m >= args.min_distance_m
            ]
            start = time.perf_counter()
            predicted = network.estimate(frame)
            net_ms.append(1000 * (time.perf_counter() - start))
            lanes = lanes_detector.detect(frame) if lanes_detector is not None else None

            tracks = [
                Track(box=o.box, score=1.0, class_id=0, class_name=o.class_name, track_id=o.object_id)
                for o in objects
            ]
            truth_depth = DepthMap(frame.depth_gt, is_metric=True)
            as_inverse = DepthMap(1.0 / np.maximum(predicted.depth, 1e-3), is_metric=False)
            estimates = {}
            for name, estimator in methods.items():
                depth = truth_depth if name == "depth_gt" else predicted
                if name == "net_oracle" and predicted.is_metric:
                    depth = as_inverse
                states = estimator.estimate(tracks, depth, lanes, frame)
                estimates[name] = [s.distance_m if s.distance_m is not None else np.nan for s in states]
            for k, o in enumerate(objects):
                rows.append(
                    {
                        "recording": path.name,
                        "frame": i,
                        "object_id": o.object_id,
                        "class_name": o.class_name,
                        "group": "vehicle" if o.class_name in VEHICLES else "person",
                        "category": category(o, frame.camera.width, frame.camera.height),
                        "truth": o.distance_m,
                        **{name: estimates[name][k] for name in methods},
                    }
                )

            gt = frame.depth_gt[::4, ::4]
            valid = (gt > 0.5) & (gt < DENSE_MAX_M)
            inverse = (as_inverse if predicted.is_metric else predicted).depth[::4, ::4]
            entry = {"recording": path.name, "frame": i}
            road_scale = (1.0, 0.0) if predicted.is_metric else methods["net_road"].scale
            for name, scale in (
                ("road", road_scale),
                ("oracle", fit_inverse_depth(inverse[valid], gt[valid])),
            ):
                # predictions capped at the evaluated range, as in the KITTI depth benchmark
                pred = np.minimum(to_metric(inverse, *scale), DENSE_MAX_M) if scale else None
                entry[f"abs_rel_{name}"] = dense_abs_rel(pred, gt, valid) if scale else np.nan
            dense.append(entry)

    objects = pd.DataFrame(rows)
    objects["bin"] = distance_bin(objects["truth"].to_numpy())
    dense = pd.DataFrame(dense)
    if args.out:
        objects.to_parquet(args.out)

    pd.set_option("display.width", 200)
    fmt = {"float_format": lambda x: f"{x:.3f}", "index": False}
    clean = objects[objects["category"] == "clean"]
    overall = distance_table(objects, METHODS)
    by_category = distance_table(objects, METHODS, by="category")
    by_bin = distance_table(clean, METHODS, by="bin")
    by_group = distance_table(clean, METHODS, by="group")
    print(f"{len(objects)} objects in {len(dense)} frames from {len(args.recordings)} recordings")
    print(objects["category"].value_counts().to_string())
    print("\nall objects\n" + overall.to_string(**fmt))
    print("\nby category\n" + by_category.to_string(**fmt))
    print("\nclean objects by distance\n" + by_bin.to_string(**fmt))
    print("\nclean objects by class group\n" + by_group.to_string(**fmt))
    calibrated = dense["abs_rel_road"].notna().mean()
    road = "as predicted (metric)" if network.metric else "road calibration"
    print(
        f"\nnetwork dense AbsRel (<{DENSE_MAX_M:.0f} m, median over frames): "
        f"{road} {dense['abs_rel_road'].median():.3f} (in {calibrated:.1%} of frames), "
        f"oracle calibration {dense['abs_rel_oracle'].median():.3f}"
    )
    print(f"network {np.mean(net_ms[1:]):.1f} ms per frame")

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="eval",
            group="depth",
            name="eval-distance-" + Path(depth_cfg.get("model", "depth")).stem,
            config={
                **{k: str(v) for k, v in vars(args).items()},
                "recordings": [str(r) for r in args.recordings],
                "depth": depth_cfg,
                "overrides": overrides,
            },
            tags=["phase4", "depth", "distance", "carla"],
        )
        run.log(
            {
                "eval/distance": wandb.Table(dataframe=overall),
                "eval/distance_by_category": wandb.Table(dataframe=by_category),
                "eval/distance_by_bin": wandb.Table(dataframe=by_bin.astype({"bin": str})),
                "eval/distance_by_group": wandb.Table(dataframe=by_group),
                "dense/abs_rel_road": float(dense["abs_rel_road"].median()),
                "dense/abs_rel_oracle": float(dense["abs_rel_oracle"].median()),
                "dense/fit_rate_road": float(calibrated),
                "network_ms": float(np.mean(net_ms[1:])),
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
