"""ByteTrack through Ultralytics' BYTETracker, fed with our Detections instead of a YOLO Results object.

ByteTrack matches tracks to detections in two rounds: high-score detections first, then the leftover tracks
against low-score ones (often occluded or blurred objects), so the detector threshold must stay below
`track_high_thresh` or the second round has nothing to work with.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from perception.core.registry import register
from perception.core.types import Detection, Frame, Track


class TrackerInput:
    """The Results-like view BYTETracker reads: xywh (centre, size), conf, cls, boolean indexing."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> TrackerInput:
        if not detections:
            return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
        xyxy = np.stack([d.box for d in detections]).astype(np.float32)
        xywh = np.concatenate([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], axis=1)
        conf = np.array([d.score for d in detections], np.float32)
        return cls(xywh, conf, np.array([d.class_id for d in detections], np.float32))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> TrackerInput:
        return TrackerInput(self.xywh[mask], self.conf[mask], self.cls[mask])


@register("tracker", "bytetrack")
class ByteTrackTracker:
    """ByteTrack behind the Tracker interface. Defaults are Ultralytics' bytetrack.yaml.

    track_high_thresh  first-round detection score
    track_low_thresh   second-round detections score above this (and below track_high_thresh)
    new_track_thresh   unmatched detections at or above this start a new track
    track_buffer       frames a lost track is kept for re-identification
    match_thresh       maximum matching cost (1 - IoU, fused with the score if fuse_score)

    Returned boxes are the Kalman-filtered track boxes, not the raw detections. A new track is reported
    from its second matched frame (except on the first frame), which filters one-frame false positives.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self._tracker = BYTETracker(args)
        self.track_buffer = track_buffer
        self._frame = 0
        self._seen: dict[int, tuple[int, int]] = {}  # track id -> (first, last) frame it was reported

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        self._frame += 1
        rows = self._tracker.update(TrackerInput.from_detections(detections))
        tracks = []
        for x1, y1, x2, y2, track_id, score, _, idx in rows:
            det = detections[int(idx)]  # the detection matched this frame; class comes from it
            first, _ = self._seen.get(int(track_id), (self._frame, self._frame))
            self._seen[int(track_id)] = (first, self._frame)
            tracks.append(
                Track(
                    box=np.array([x1, y1, x2, y2], np.float32),
                    score=float(score),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    track_id=int(track_id),
                    age=self._frame - first,
                )
            )
        # Forget ids the tracker has dropped, so the age table doesn't grow over a long drive
        self._seen = {i: s for i, s in self._seen.items() if self._frame - s[1] <= self.track_buffer + 1}
        return tracks

    def reset(self) -> None:
        self._tracker.reset()
        self._frame = 0
        self._seen.clear()
