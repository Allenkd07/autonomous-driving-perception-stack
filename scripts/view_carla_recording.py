"""Check a CARLA recording (carla_scripts/record_sequence.py) before evaluating on it: a ground-truth overlay
video plus a few consistency numbers.

    python scripts/view_carla_recording.py data/carla/smoke_town10
    python scripts/view_carla_recording.py data/carla/town03_wet_s1 --step 5 --out runs/gt_town03.mp4

What to look for in the video (colours: perception.viz.overlay.draw_ground_truth):
- every moving car, bike and pedestrian has a coloured box, and its pixels are tinted in the same colour
  (its instance id matched to its actor); a tint in another actor's colour means a wrong match
- grey tint and boxes are vehicles/people without a recorded actor: parked map cars, actors out of range;
  a moving object in grey means it is missing from actors.parquet or its instance id wasn't matched
- boxes sit on their object in every frame, also when the ego vehicle turns; a box lagging behind or ahead
  of its object means the poses and images are from different ticks
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from perception.io.carla_recording import DYNAMIC_TAGS, CarlaRecording
from perception.viz.overlay import draw_ground_truth


def check_timing(rec: CarlaRecording) -> None:
    fps = rec.meta.get("fps", 20)
    steps = np.diff(rec.frames.timestamp.to_numpy())
    ticks = np.diff(rec.frames.carla_frame.to_numpy())
    print(f"{len(rec)} frames, {steps.sum():.1f} s simulated")
    print(f"  time step: {steps.min():.4f}-{steps.max():.4f} s (expected {1 / fps:.4f})")
    if (ticks != 1).any():
        print(f"  WARNING: {(ticks != 1).sum()} gaps in carla_frame (dropped ticks)")


def check_matching(rec: CarlaRecording, actors: pd.DataFrame) -> None:
    unique = actors.drop_duplicates("actor_id")
    print(f"{len(unique)} recorded actors: {dict(Counter(unique.class_name))}")
    mapping = rec.instance_to_actor
    shared = [a for a, n in Counter(mapping.values()).items() if n > 1]
    print(f"  {len(mapping)} instance ids matched to {len(set(mapping.values()))} actors")
    if shared:
        print(f"  note: actors with several instance ids: {sorted(shared)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, help="overlay video (default: <recording>/gt_overlay.mp4)")
    parser.add_argument("--step", type=int, default=1, help="draw every n-th frame")
    parser.add_argument("--min-visible-px", type=int, default=100)
    args = parser.parse_args()

    rec = CarlaRecording(args.recording)
    check_timing(rec)
    check_matching(rec, pd.read_parquet(args.recording / "actors.parquet"))

    out = args.out or args.recording / "gt_overlay.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = rec.meta.get("fps", 20) / args.step
    size = (rec.camera.width, rec.camera.height)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    gt_counts, unlabelled_counts, seen = [], [], Counter()
    for index in rec.frames.index[:: args.step]:
        frame = rec.frame(int(index), with_depth=False)
        tag, actor_ids = rec.actor_ids(int(index))
        objects = rec.ground_truth(int(index), args.min_visible_px)
        unlabelled = rec.unlabelled_boxes(int(index))
        gt_counts.append(len(objects))
        unlabelled_counts.append(len(unlabelled))
        seen.update(obj.class_name for obj in objects)
        speed = np.linalg.norm(rec.frames.loc[index, "ego_velocity"]) * 3.6
        header = f"frame {index}  t {frame.timestamp:.2f} s  ego {speed:.0f} km/h  GT {len(objects)}"
        dynamic = np.isin(tag, DYNAMIC_TAGS)
        image = draw_ground_truth(frame.image, objects, unlabelled, dynamic, actor_ids, header)
        writer.write(image)
    writer.release()

    for name, counts in (("ground truth", gt_counts), ("unlabelled regions", unlabelled_counts)):
        print(f"{name} per frame: mean {np.mean(counts):.1f}, max {max(counts)}")
    print(f"object-frames per class: {dict(seen)}")
    print(f"wrote {len(gt_counts)} frames to {out}")


if __name__ == "__main__":
    main()
