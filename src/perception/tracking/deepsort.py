"""DeepSORT (Wojke et al., 2017): Kalman motion model + appearance features, matched in two stages.

Per frame:
1. Predict every track's box with the Kalman filter.
2. Matching cascade: confirmed tracks are matched to detections by appearance (smallest cosine distance to
   the track's recent features), level by level from the most recently seen tracks to the longest lost, so a
   fresh track wins over a stale one. Pairs the Kalman filter finds implausible (Mahalanobis gate) are
   excluded.
3. IoU matching for what's left: new (tentative) tracks and tracks missed only in the last frame.
4. A track is confirmed after n_init consecutive hits; a tentative track is deleted on its first miss and a
   confirmed one after track_buffer misses.

The motion model is Ultralytics' KalmanFilterXYAH, the same one ByteTrack uses, so a comparison between the
two trackers isolates association (appearance + cascade vs. two-round score-based IoU matching).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import lap
import numpy as np

from perception.core import registry
from perception.core.registry import register
from perception.core.types import Detection, Frame, Track
from perception.tracking.embedder import AppearanceEmbedder

CHI2_95_4DOF = 9.4877  # 95% gate on the squared Mahalanobis distance of a 4-D (x, y, a, h) measurement
GATED = 1e5  # cost of a pair that must not match


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """Box corners -> (centre x, centre y, aspect ratio w/h, height), the Kalman measurement."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w / h, h], np.float64)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) IoU between xyxy boxes a (N, 4) and b (M, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost: np.ndarray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost one-to-one matching; pairs above max_cost stay unmatched.

    Returns (row, col) matches, unmatched rows and unmatched cols.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    _, row_to_col, col_to_row = lap.lapjv(cost.astype(np.float64), extend_cost=True, cost_limit=max_cost)
    matches = [(r, int(c)) for r, c in enumerate(row_to_col) if c >= 0]
    unmatched_rows = [r for r in range(rows) if row_to_col[r] < 0]
    return matches, unmatched_rows, [c for c in range(cols) if col_to_row[c] < 0]


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


@dataclass(eq=False)
class _Track:
    track_id: int
    mean: np.ndarray  # Kalman state (x, y, a, h, vx, vy, va, vh)
    covariance: np.ndarray
    first_frame: int
    detection: Detection  # last matched detection, for class and score
    features: deque  # recent appearance features (the track's gallery)
    hits: int = 1
    time_since_update: int = 0
    state: TrackState = field(default=TrackState.TENTATIVE)


class DeepSort:
    """The DeepSORT association logic, fed with detections and their appearance features.

    max_cosine_distance  appearance matching threshold (depends on the embedding model; tune per model)
    max_iou_distance     IoU-stage threshold on 1 - IoU
    track_buffer         frames a confirmed track survives without a match (the paper's max_age)
    n_init               consecutive hits before a track is confirmed and reported
    nn_budget            features kept per track
    """

    def __init__(
        self,
        max_cosine_distance: float = 0.2,
        max_iou_distance: float = 0.7,
        track_buffer: int = 30,
        n_init: int = 3,
        nn_budget: int = 100,
    ) -> None:
        from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

        self.kf = KalmanFilterXYAH()
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance
        self.track_buffer = track_buffer
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_Track] = []
        self._next_id = 1
        self._frame = 0

    def update(self, detections: Sequence[Detection], features: np.ndarray) -> list[Track]:
        """features: (N, D) unit-length rows, one per detection."""
        self._frame += 1
        for t in self.tracks:
            t.mean, t.covariance = self.kf.predict(t.mean, t.covariance)
            t.time_since_update += 1

        measurements = np.array([xyxy_to_xyah(d.box) for d in detections]).reshape(-1, 4)
        matches, unmatched_tracks, unmatched_dets = self._match(detections, measurements, features)

        for ti, di in matches:
            t = self.tracks[ti]
            t.mean, t.covariance = self.kf.update(t.mean, t.covariance, measurements[di])
            t.features.append(features[di])
            t.detection = detections[di]
            t.hits += 1
            t.time_since_update = 0
            if t.state is TrackState.TENTATIVE and t.hits >= self.n_init:
                t.state = TrackState.CONFIRMED
        for ti in unmatched_tracks:
            t = self.tracks[ti]
            if t.state is TrackState.TENTATIVE or t.time_since_update > self.track_buffer:
                t.state = TrackState.DELETED
        for di in unmatched_dets:
            mean, covariance = self.kf.initiate(measurements[di])
            gallery = deque([features[di]], maxlen=self.nn_budget)
            self.tracks.append(_Track(self._next_id, mean, covariance, self._frame, detections[di], gallery))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.state is not TrackState.DELETED]

        return [
            Track(
                box=xyah_to_xyxy(t.mean),
                score=t.detection.score,
                class_id=t.detection.class_id,
                class_name=t.detection.class_name,
                track_id=t.track_id,
                age=self._frame - t.first_frame,
            )
            for t in self.tracks
            if t.state is TrackState.CONFIRMED and t.time_since_update == 0
        ]

    def _match(
        self, detections: Sequence[Detection], measurements: np.ndarray, features: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        confirmed = [i for i, t in enumerate(self.tracks) if t.state is TrackState.CONFIRMED]
        tentative = [i for i, t in enumerate(self.tracks) if t.state is TrackState.TENTATIVE]

        # Matching cascade: after predict, a track seen last frame has time_since_update == 1
        matches: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        for level in range(1, self.track_buffer + 1):
            if not unmatched_dets:
                break
            level_tracks = [i for i in confirmed if self.tracks[i].time_since_update == level]
            if not level_tracks:
                continue
            cost = self._appearance_cost(level_tracks, unmatched_dets, measurements, features)
            level_matches, _, unmatched = assign(cost, self.max_cosine_distance)
            matches += [(level_tracks[r], unmatched_dets[c]) for r, c in level_matches]
            unmatched_dets = [unmatched_dets[c] for c in unmatched]
        matched_tracks = {ti for ti, _ in matches}
        unmatched_confirmed = [i for i in confirmed if i not in matched_tracks]

        # IoU stage: tentative tracks plus confirmed ones that only just lost appearance contact
        iou_tracks = tentative + [i for i in unmatched_confirmed if self.tracks[i].time_since_update == 1]
        unmatched_tracks = [i for i in unmatched_confirmed if self.tracks[i].time_since_update != 1]
        if iou_tracks and unmatched_dets:
            track_boxes = np.stack([xyah_to_xyxy(self.tracks[i].mean) for i in iou_tracks])
            det_boxes = np.stack([detections[i].box for i in unmatched_dets])
            cost = 1.0 - iou_matrix(track_boxes, det_boxes)
            iou_matches, unmatched_rows, unmatched_cols = assign(cost, self.max_iou_distance)
            matches += [(iou_tracks[r], unmatched_dets[c]) for r, c in iou_matches]
            unmatched_tracks += [iou_tracks[r] for r in unmatched_rows]
            unmatched_dets = [unmatched_dets[c] for c in unmatched_cols]
        else:
            unmatched_tracks += iou_tracks
        return matches, unmatched_tracks, unmatched_dets

    def _appearance_cost(
        self, track_ids: list[int], det_ids: list[int], measurements: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Smallest cosine distance between each track's gallery and each detection, Mahalanobis-gated."""
        det_features = features[det_ids]
        cost = np.empty((len(track_ids), len(det_ids)))
        for row, ti in enumerate(track_ids):
            t = self.tracks[ti]
            cost[row] = (1.0 - np.stack(t.features) @ det_features.T).min(axis=0)
            gate = self.kf.gating_distance(t.mean, t.covariance, measurements[det_ids]) > CHI2_95_4DOF
            cost[row, gate] = GATED
        return cost


@register("tracker", "deepsort")
class DeepSortTracker:
    """DeepSORT behind the Tracker interface: appearance embedder + DeepSort association.

    reid_model       embedding model file for reid_backend (see scripts/export_reid.py)
    reid_input_size  crop size (width, height) the model expects
    min_score        detections below this are ignored; DeepSORT has no low-score round like ByteTrack's,
                     so this matches ByteTrack's track_high_thresh for a like-for-like comparison
    The other arguments are DeepSort's.
    """

    def __init__(
        self,
        reid_model: str = "weights/reid_resnet18.ts",
        reid_backend: str = "torch",
        reid_device: str | None = None,
        reid_input_size: Sequence[int] = (128, 128),
        min_score: float = 0.25,
        max_cosine_distance: float = 0.2,
        max_iou_distance: float = 0.7,
        track_buffer: int = 30,
        n_init: int = 3,
        nn_budget: int = 100,
    ) -> None:
        backend_cfg = {"type": reid_backend, "model": reid_model, "device": reid_device}
        backend = registry.build("backend", backend_cfg)
        self.embedder = AppearanceEmbedder(backend, tuple(reid_input_size))
        self.min_score = min_score
        self.deepsort = DeepSort(max_cosine_distance, max_iou_distance, track_buffer, n_init, nn_budget)

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        kept = [d for d in detections if d.score >= self.min_score]
        boxes = np.array([d.box for d in kept], np.float32).reshape(-1, 4)
        return self.deepsort.update(kept, self.embedder(frame.image, boxes))

    def reset(self) -> None:
        self.deepsort.reset()
