"""Run the perception pipeline on a CARLA recording, a video or an image folder, and write an overlay video.

    python scripts/run_pipeline.py data/carla/town10_noon_s0 depth=null distance.type=ground_plane
    python scripts/run_pipeline.py clip.mp4 --config configs/pipeline/bdd100k.yaml --fov 60 --pitch 1
    python scripts/run_pipeline.py frames/ --fps 5 --max-frames 100 tracker.type=deepsort

The input type is detected: a folder with meta.json is a CARLA recording (which knows its camera), any
other folder is a sequence of images, anything else a video. For videos and images the camera's field of
view, height and pitch set the ground-plane distances. Any pipeline config key can be overridden as
key=value. Prints the mean time per stage at the end.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from perception.core.config import load_config
from perception.io import VideoSink, open_source
from perception.pipeline import PerceptionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline/default.yaml"))
    parser.add_argument("--out", type=Path, help="overlay video (default: runs/pipeline/<input name>.mp4)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, default=10.0, help="frame rate of an image folder")
    parser.add_argument("--fov", type=float, default=90.0, help="horizontal field of view, degrees")
    parser.add_argument("--camera-height", type=float, default=1.5, help="metres above the road")
    parser.add_argument("--pitch", type=float, default=0.0, help="degrees, positive looking down")
    parser.add_argument("--ttc-warning", type=float, default=2.0, help="TTC (s) that turns in-lane boxes red")
    args, overrides = parser.parse_known_args()

    pipeline = PerceptionPipeline.from_config(load_config(args.config, overrides))
    source, fps = open_source(args.input, args.fov, args.camera_height, args.pitch, args.fps)
    out = args.out or Path("runs/pipeline") / f"{args.input.stem}.mp4"
    sink = VideoSink(out, fps=fps, ttc_warning_s=args.ttc_warning)

    timings = defaultdict(list)
    try:
        for i, frame in enumerate(source):
            if args.max_frames is not None and i >= args.max_frames:
                break
            output = pipeline.process(frame)
            sink.write(output)
            for stage, ms in output.timings_ms.items():
                timings[stage].append(ms)
            timings["total"].append(sum(output.timings_ms.values()))
    finally:
        sink.close()

    print(f"wrote {sink.frames} frames to {out}")
    for stage, values in timings.items():
        # The first frame includes model warm-up (CUDA kernels, memory), so it is left out of the mean
        steady = values[1:] or values
        print(f"{stage:>9}: {np.mean(steady):6.1f} ms  (first frame {values[0]:.0f} ms)")
    if timings["total"]:
        print(f"{1000 / np.mean(timings['total'][1:] or timings['total']):.1f} FPS")


if __name__ == "__main__":
    main()
