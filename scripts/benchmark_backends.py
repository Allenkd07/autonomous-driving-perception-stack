"""Benchmark the models on each inference backend: latency, GPU memory and output change vs PyTorch FP32.

    python scripts/benchmark_backends.py data/carla/town10_noon_s0
    python scripts/benchmark_backends.py data/carla/town10_noon_s0 --models depth --runs 300 --no-wandb

Per model and variant (torch FP32 / FP16, ONNX Runtime, TensorRT FP32 / FP16; variants whose model file is
missing are skipped):
    latency   model call only, on real frames: host -> GPU copy, forward, GPU -> host copy (the backends
              return numpy arrays, so the GPU has finished when the call returns). YOLO: the whole
              Ultralytics predict call, so its pre/post-processing and NMS are included.
    memory    GPU memory the process holds (device total used, minus before CUDA started): CUDA context,
              weights, workspace and the allocator's cache, since the backends allocate outside PyTorch
    fidelity  outputs vs torch FP32 on the same frames. Depth: mean relative difference. Lanes: IoU of
              the thresholded drivable / lane-line masks. YOLO: F1 of matched boxes (same class, IoU >= 0.5).
lanes_raw and depth_raw are the --raw-input exports (uint8 in, normalisation inside the model): they are
compared with, and their speed-up taken against, the original model's torch FP32 run, so the latency
includes what moved from the CPU into the model.

Each variant runs in its own process, so one backend's memory doesn't count against the next. Accuracy on
the task itself (mAP, lane IoU, distance error) comes from the eval scripts with the engine swapped in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WEIGHTS = Path("weights")
NETWORKS = {
    # name: (file stem, output names, input size (w, h))
    "lanes": ("twinlitenetplus_medium", ("drivable", "lanes"), (640, 384)),
    "depth": ("depth_anything_v2_small_relative_686x518", ("depth",), (686, 518)),
    "depth_518x392": ("depth_anything_v2_small_relative_518x392", ("depth",), (518, 392)),
    # exported with --raw-input: BGR uint8 in, normalisation (and the lane softmax) inside the model
    "lanes_raw": ("twinlitenetplus_medium_raw", ("drivable", "lanes"), (640, 384)),
    "depth_raw": ("depth_anything_v2_small_relative_686x518_raw", ("depth",), (686, 518)),
}
# raw-input model -> the model it was exported from: its torch FP32 run is the reference for both
RAW = {"lanes_raw": "lanes", "depth_raw": "depth"}
NETWORK_VARIANTS = {
    # variant: (backend, file suffix, half)
    "torch-fp32": ("torch", ".ts", False),
    "torch-fp16": ("torch", ".ts", True),
    "onnx-fp32": ("onnx", ".onnx", False),
    "tensorrt-fp32": ("tensorrt", "_fp32.engine", False),
    "tensorrt-fp16": ("tensorrt", "_fp16.engine", False),
}
YOLO = "yolo11s_bdd100k_random"
YOLO_IMGSZ = (480, 640)  # (h, w) of the ONNX / engine exports (scripts/export_yolo.py)
YOLO_VARIANTS = {
    "torch-fp32": (".pt", False),
    "torch-fp16": (".pt", True),
    "onnx-fp32": (".onnx", False),
    "tensorrt-fp16": (".engine", False),
}
REFERENCE = "torch-fp32"


def model_file(model: str, variant: str) -> Path:
    if model == "yolo":
        return WEIGHTS / f"{YOLO}{YOLO_VARIANTS[variant][0]}"
    return WEIGHTS / f"{NETWORKS[model][0]}{NETWORK_VARIANTS[variant][1]}"


def variants(model: str) -> list[str]:
    return list(YOLO_VARIANTS if model == "yolo" else NETWORK_VARIANTS)


# --- worker: one model and variant, in its own process ---------------------------------------------------


def gpu_used_mb() -> float:
    import pynvml

    pynvml.nvmlInit()
    used = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used
    return used / 2**20


def load_frames(recording: Path, count: int):
    from perception.io import CarlaRecording

    source = CarlaRecording(recording)
    indices = np.linspace(0, len(source) - 1, count).round().astype(int)
    return [source.frame(int(i), with_depth=False) for i in indices]


def worker(args: argparse.Namespace) -> None:
    base_mb = gpu_used_mb()  # before this process touches CUDA
    import torch

    from perception.core import registry

    frames = load_frames(args.recording, args.frames)
    if args.model == "yolo":
        from perception.detection.yolo import YoloDetector

        # imgsz as the exported engine (h, w): CARLA's 800x600 letterboxed to 640x480 with no padding
        detector = YoloDetector(
            model_file("yolo", args.variant),
            conf=0.1,
            imgsz=YOLO_IMGSZ,
            half=YOLO_VARIANTS[args.variant][1],
            device=0,
        )

        def call(i: int):
            return detector.detect(frames[i % len(frames)])

        def keep(result) -> np.ndarray:
            return np.array([[*d.box, d.class_id] for d in result], np.float32).reshape(-1, 5)
    else:
        stem, outputs, (w, h) = NETWORKS[args.model]
        backend_type, _, half = NETWORK_VARIANTS[args.variant]
        backend = registry.build(
            "backend",
            {
                "type": backend_type,
                "model": str(model_file(args.model, args.variant)),
                "inputs": ["images"],
                "outputs": list(outputs),
                "half": half,
            },
        )
        raw = args.model in RAW
        if RAW.get(args.model, args.model) == "lanes":
            from perception.lanes.segmentation import SegLaneDetector

            pre = SegLaneDetector(model=None, input_size=(w, h), raw_input=raw)
            inputs = [pre.preprocess(f.image)[0] for f in frames]
        else:
            from perception.depth.network import NetworkDepth

            pre = NetworkDepth.__new__(NetworkDepth)  # only its preprocessing: no model loaded
            pre.input_w, pre.input_h, pre.raw_input = w, h, raw
            inputs = [pre.preprocess(f.image) for f in frames]

        def call(i: int):
            return backend.run({"images": inputs[i % len(inputs)]})

        def keep(result) -> dict:
            return result

    kept = [keep(call(i)) for i in range(len(frames))]  # also the warm-up (lazy init, cuDNN autotune)
    for i in range(args.warmup):
        call(i)
    times = []
    for i in range(args.runs):
        start = time.perf_counter()
        call(i)
        times.append((time.perf_counter() - start) * 1000.0)
    torch.cuda.synchronize()
    mem_mb = gpu_used_mb() - base_mb

    if args.model == "yolo":
        np.savez(args.outputs, *kept)
    else:
        np.savez(args.outputs, **{f"{k}_{i}": v for i, out in enumerate(kept) for k, v in out.items()})
    times = np.array(times)
    result = {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "gpu_mem_mb": float(mem_mb),
    }
    Path(args.result).write_text(json.dumps(result))


# --- fidelity vs the reference ---------------------------------------------------------------------------


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:4] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def detection_f1(pred: list[np.ndarray], ref: list[np.ndarray]) -> float:
    """Greedy one-to-one matching per frame: same class and IoU >= 0.5."""
    tp = fp = fn = 0
    for p, r in zip(pred, ref, strict=True):
        iou = box_iou(p, r) * (p[:, None, 4] == r[None, :, 4]) if len(p) and len(r) else np.zeros((len(p), 0))
        matched = 0
        while iou.size and iou.max() >= 0.5:
            i, j = np.unravel_index(iou.argmax(), iou.shape)
            iou[i, :] = iou[:, j] = 0
            matched += 1
        tp += matched
        fp += len(p) - matched
        fn += len(r) - matched
    return 2 * tp / max(2 * tp + fp + fn, 1)


def class1(x: np.ndarray) -> np.ndarray:
    """Pixels where class 1 wins: from 2-class logits (1, 2, h, w), or probabilities (1, h, w) (raw input)."""
    return x[0, 1] > x[0, 0] if x.ndim == 4 else x[0] > 0.5


def fidelity(model: str, outputs: Path, reference: Path) -> dict[str, float]:
    out, ref = np.load(outputs), np.load(reference)
    if model == "yolo":
        return {"det_f1": detection_f1([out[k] for k in out.files], [ref[k] for k in ref.files])}
    if model.startswith("depth"):
        rel = [np.mean(np.abs(out[k] - ref[k]) / np.maximum(np.abs(ref[k]), 1e-3)) for k in ref.files]
        return {"depth_rel_diff": float(np.mean(rel))}
    scores = {}
    for name in ("drivable", "lanes"):
        ious = []
        for k in (k for k in ref.files if k.startswith(name)):
            a, b = class1(out[k]), class1(ref[k])
            union = np.logical_or(a, b).sum()
            ious.append(np.logical_and(a, b).sum() / union if union else 1.0)
        scores[f"{name}_iou"] = float(np.mean(ious))
    return scores


# --- main ------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recording", type=Path, help="CARLA recording whose frames are the inputs")
    parser.add_argument(
        "--models", nargs="+", default=["yolo", "lanes", "lanes_raw", "depth", "depth_raw", "depth_518x392"]
    )
    parser.add_argument("--variants", nargs="+", help="default: all whose model file exists")
    parser.add_argument("--frames", type=int, default=50, help="frames kept for the fidelity comparison")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("runs/benchmark/backends.json"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("MODEL", "VARIANT"), help=argparse.SUPPRESS)
    parser.add_argument("--outputs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        args.model, args.variant = args.worker
        worker(args)
        return

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for model in args.models:
            names = [v for v in variants(model) if not args.variants or v in args.variants]
            if REFERENCE not in names:
                names.insert(0, REFERENCE)  # fidelity needs it
            for variant in names:
                path = model_file(model, variant)
                if not path.exists():
                    print(f"skip {model} {variant}: no {path}")
                    continue
                outputs, result = Path(tmp) / f"{model}_{variant}.npz", Path(tmp) / f"{model}_{variant}.json"
                cmd = [sys.executable, __file__, str(args.recording), "--worker", model, variant]
                cmd += ["--frames", str(args.frames), "--warmup", str(args.warmup), "--runs", str(args.runs)]
                cmd += ["--outputs", str(outputs), "--result", str(result)]
                print(f"{model} {variant} ({path})", flush=True)
                if subprocess.run(cmd).returncode != 0:
                    print(f"  failed: {model} {variant}")
                    continue
                row = {"model": model, "variant": variant, "file": str(path)}
                row.update(json.loads(result.read_text()))
                reference = Path(tmp) / f"{RAW.get(model, model)}_{REFERENCE}.npz"
                if reference.exists():
                    row.update(fidelity(model, outputs, reference))
                rows.append(row)
                print("  " + "  ".join(f"{k} {v:.4g}" for k, v in row.items() if isinstance(v, float)))

    for row in rows:
        base = RAW.get(row["model"], row["model"])
        ref = [r for r in rows if r["model"] == base and r["variant"] == REFERENCE]
        row["speedup"] = ref[0]["mean_ms"] / row["mean_ms"] if ref else float("nan")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    metrics = ["det_f1", "drivable_iou", "lanes_iou", "depth_rel_diff"]
    print(f"\n| model | variant | p50 ms | p95 ms | FPS | speed-up | GPU MB | {' | '.join(metrics)} |")
    print("|" + " --- |" * (7 + len(metrics)))
    for r in rows:
        fid = " | ".join(f"{r[m]:.4f}" if m in r else "" for m in metrics)
        print(
            f"| {r['model']} | {r['variant']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
            f"{1000 / r['mean_ms']:.0f} | {r['speedup']:.1f}x | {r['gpu_mem_mb']:.0f} | {fid} |"
        )
    print(f"wrote {args.out}")

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="benchmark",
            group="optimization",
            name=f"benchmark-backends-{args.recording.name}",
            config={**{k: str(v) for k, v in vars(args).items()}},
            tags=["phase6", "tensorrt", "benchmark"],
        )
        columns = sorted({k for r in rows for k in r})
        run.log({"backends": wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])})
        run.finish()


if __name__ == "__main__":
    main()
