"""Compare trackers on CARLA recordings (carla_scripts/record_sequence.py) against their ground truth.

    python scripts/eval_tracking.py data/carla/town10_noon_s0 data/carla/town03_wet_s1
    python scripts/eval_tracking.py data/carla/* --trackers deepsort deepsort.max_cosine_distance=0.3

The detector runs once per frame and its detections are cached, so every tracker sees the same input and
reruns with other tracker settings skip detection. Reports MOTA, IDF1, ID switches, ... for all objects
and per group, plus each tracker's time per frame (including DeepSORT's appearance model).

Scoring rules (perception.eval.tracking, perception.io.carla_recording):
- ground truth is each actor's visible box; actors with fewer than --min-visible-px visible pixels, and
  vehicles/people without a recorded actor (parked map cars, actors out of range), are ignore regions
- rider predictions are left out: a CARLA bicycle or motorcycle is one actor with its rider, while
  BDD100K (and so the detector) labels the rider separately
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from perception.core import registry
from perception.core.types import Detection
from perception.eval.tracking import METRICS, TrackingEvaluator, summarize
from perception.io.carla_recording import CarlaRecording, GroundTruthObject

GROUPS = {
    "vehicle": {"car", "truck", "bus"},
    "person": {"person"},
    "two-wheeler": {"bicycle", "motorcycle"},
}
GROUPS = {"all": set().union(*GROUPS.values()), **GROUPS}


def detections_for(rec: CarlaRecording, detector_cfg: dict, cache_dir: Path) -> list[list[Detection]]:
    """The detector's output for every frame, from the cache when it has been computed before."""
    weights = Path(detector_cfg["weights"]).stem
    cache = cache_dir / f"{rec.root.name}_{weights}_conf{detector_cfg['conf']}_{detector_cfg['imgsz']}.npz"
    if not cache.exists():
        detector = registry.build("detector", detector_cfg)
        rows = []
        for i in range(len(rec)):
            for d in detector.detect(rec.frame(i, with_depth=False)):
                rows.append((i, *d.box, d.score, d.class_id, d.class_name))
        cache.parent.mkdir(parents=True, exist_ok=True)
        table = np.array(rows, dtype=object).reshape(-1, 8)
        np.savez(cache, frame=table[:, 0].astype(int), boxes=table[:, 1:5].astype(np.float32),
                 scores=table[:, 5].astype(np.float32), class_ids=table[:, 6].astype(int),
                 class_names=table[:, 7].astype(str))  # fmt: skip
        print(f"cached {len(rows):,} detections -> {cache}")
    data = np.load(cache)
    per_frame: list[list[Detection]] = [[] for _ in range(len(rec))]
    for i, box, score, cid, name in zip(
        data["frame"], data["boxes"], data["scores"], data["class_ids"], data["class_names"], strict=True
    ):
        per_frame[i].append(Detection(box=box, score=float(score), class_id=int(cid), class_name=str(name)))
    return per_frame


def ground_truth_for(rec: CarlaRecording) -> list[tuple[list[GroundTruthObject], np.ndarray]]:
    """Every frame's actors (with any visible pixel) and unlabelled vehicle/person boxes."""
    return [(rec.ground_truth(i, min_visible_px=1), rec.unlabelled_boxes(i)) for i in range(len(rec))]


def evaluate(
    rec: CarlaRecording, detections, ground_truth, tracker, min_visible_px: int
) -> tuple[dict[str, TrackingEvaluator], float]:
    """Run the tracker over one recording; returns an evaluator per group and the mean ms per frame."""
    evaluators = {group: TrackingEvaluator() for group in GROUPS}
    tracker.reset()
    elapsed = 0.0
    for i in range(len(rec)):
        frame = rec.frame(i, with_depth=False)
        start = time.perf_counter()
        tracks = tracker.update(detections[i], frame)
        elapsed += time.perf_counter() - start

        objects, unlabelled = ground_truth[i]
        for group, classes in GROUPS.items():
            scored = [o for o in objects if o.class_name in classes and o.visible_px >= min_visible_px]
            scored_ids = {o.actor_id for o in scored}
            ignore = [o.box for o in objects if o.actor_id not in scored_ids]
            preds = [t for t in tracks if t.class_name in classes]
            evaluators[group].update(
                i,
                [o.actor_id for o in scored],
                np.array([o.box for o in scored]),
                [t.track_id for t in preds],
                np.array([t.box for t in preds]),
                np.concatenate([np.array(ignore, np.float32).reshape(-1, 4), unlabelled]),
            )
    return evaluators, 1000 * elapsed / max(len(rec), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--trackers", nargs="+", default=["bytetrack", "deepsort"])
    parser.add_argument("--weights", default="weights/yolo11s_bdd100k.pt")
    parser.add_argument("--conf", type=float, default=0.1, help="detector threshold (low, for ByteTrack)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-visible-px", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/eval_tracking"))
    parser.add_argument("--no-wandb", action="store_true")
    args, overrides = parser.parse_known_args()
    # tracker.key=value overrides, e.g. deepsort.max_cosine_distance=0.3
    tracker_args = OmegaConf.to_container(OmegaConf.from_dotlist(overrides))

    detector_cfg = {"type": "yolo", "weights": args.weights, "conf": args.conf, "imgsz": args.imgsz}
    recordings = [CarlaRecording(path) for path in args.recordings]
    detections = [detections_for(rec, detector_cfg, args.cache_dir) for rec in recordings]
    ground_truth = [ground_truth_for(rec) for rec in recordings]

    results = []
    for name in args.trackers:
        tracker = registry.build("tracker", {"type": name, **tracker_args.get(name, {})})
        per_group = {group: [] for group in GROUPS}
        ms = []
        for rec, dets, gt in zip(recordings, detections, ground_truth, strict=True):
            evaluators, ms_per_frame = evaluate(rec, dets, gt, tracker, args.min_visible_px)
            for group, ev in evaluators.items():
                per_group[group].append(ev)
            ms.append(ms_per_frame)
            print(f"{name} {rec.root.name}: {ms_per_frame:.1f} ms/frame")
        for group, evaluators in per_group.items():
            metrics = summarize(evaluators)
            results.append({"tracker": name, "group": group, **metrics, "ms_per_frame": np.mean(ms)})

    import pandas as pd

    table = pd.DataFrame(results)
    columns = ["tracker", "group", *METRICS, "ms_per_frame"]
    print(table[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="eval",
            group="tracking",
            name=f"eval-tracking-{'-'.join(args.trackers)}",
            config={
                **vars(args),
                "recordings": [str(r) for r in args.recordings],
                "tracker_args": tracker_args,
            },
            tags=["phase2", "tracking"],
        )
        run.log({"eval/tracking": wandb.Table(dataframe=table[columns])})
        run.finish()


if __name__ == "__main__":
    main()
