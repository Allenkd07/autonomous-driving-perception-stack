"""YOLO detector through Ultralytics, which loads .pt, .onnx and .engine weights behind one API.

The rest of the system only sees the Detector interface and never imports Ultralytics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from perception.core.registry import register
from perception.core.types import Detection, Frame


def class_ids(names: Mapping[int, str], wanted: Sequence[str] | None) -> list[int] | None:
    """Model class ids for the wanted class names (None keeps every class)."""
    if wanted is None:
        return None
    by_name = {name: i for i, name in names.items()}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(f"classes {unknown} not in the model; it has {sorted(by_name)}")
    return [by_name[name] for name in wanted]


def to_detections(
    xyxy: np.ndarray, scores: np.ndarray, cls: np.ndarray, names: Mapping[int, str]
) -> list[Detection]:
    """One Detection per row of the model's (N, 4) boxes, (N,) scores and (N,) class ids."""
    return [
        Detection(box=box.astype(np.float32), score=float(score), class_id=int(c), class_name=names[int(c)])
        for box, score, c in zip(xyxy, scores, cls, strict=True)
    ]


@register("detector", "yolo")
class YoloDetector:
    """Ultralytics YOLO behind the Detector interface.

    weights   .pt, .onnx or .engine; class names come from the weights' metadata
    conf      minimum score kept
    iou       NMS IoU threshold
    classes   class names to keep (None keeps all); filtered inside NMS, not afterwards
    device    "cpu", 0, "cuda:0", ...; None lets Ultralytics pick
    half      FP16 inference for .pt on CUDA (an .engine's precision is fixed when it is built)
    """

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        classes: Sequence[str] | None = None,
        device: str | int | None = None,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO

        # task= stops Ultralytics guessing (and warning) for .onnx / .engine weights
        self.model = YOLO(str(weights), task="detect")
        self.names: dict[int, str] = dict(self.model.names)
        self.class_ids = class_ids(self.names, classes)
        self.predict_args = dict(conf=conf, iou=iou, imgsz=imgsz, device=device, half=half, verbose=False)

    def detect(self, frame: Frame) -> list[Detection]:
        # Ultralytics treats numpy input as BGR, which is what Frame holds
        (result,) = self.model.predict(frame.image, classes=self.class_ids, **self.predict_args)
        boxes = result.boxes
        return to_detections(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), self.names
        )
