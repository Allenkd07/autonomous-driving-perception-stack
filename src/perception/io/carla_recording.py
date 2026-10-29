"""CARLA recordings (carla_scripts/record_sequence.py): pipeline frames and 2D ground truth for evaluation.

A recording holds raw sensor images plus every vehicle's and pedestrian's 3D box per frame; 2D ground truth
is derived here, so its rules can change without re-recording:

    box          tight box around the actor's visible pixels in the instance segmentation image (what a
                 detector trained on BDD100K predicts)
    amodal_box   the projected 3D box: the full extent, including occluded parts
    visible_px   number of visible pixels; small or hidden objects are dropped by min_visible_px

CARLA's instance ids are not actor ids, so each instance id is matched to an actor by geometry: the actor
whose projected 3D box and depth range hold the id's pixels (match_frame), voted over the whole recording
(match_instances), since an object keeps its instance id from frame to frame.

Vehicles and people without a recorded actor (parked cars that are part of the map, actors beyond the
recording range) are returned separately by unlabelled_boxes, to be ignored in evaluation.

Recording layout (<root>/):
    rgb/<i>.jpg  instance/<i>.png  depth/<i>.png    i = FRAME_NAME.format(index)
    frames.parquet  index, carla_frame, timestamp, cam_to_world (16), ego_velocity (3)
    actors.parquet  index, actor_id, blueprint, class_name, corners (8 x 3, world), velocity (3)
    meta.json       camera (width, height, fov, mount), map, weather, seed, ...
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from perception.core.registry import register
from perception.core.types import CameraInfo, Frame

FRAME_NAME = "{:06d}"
RGB_DIR, INSTANCE_DIR, DEPTH_DIR = "rgb", "instance", "depth"
# CARLA 0.9.15 semantic tags of things we track (carla.CityObjectLabel): Pedestrians .. Bicycle
DYNAMIC_TAGS = np.arange(12, 20)
PEDESTRIAN_TAG = 12
RIDER_TAG = 13  # part of its bicycle or motorcycle actor, with the same instance id
# Blueprint base_type -> our class names (BDD100K's, renamed to COCO's; see perception.io.bdd100k_yolo)
BASE_TYPE_CLASS = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}
MIN_DEPTH_M = 0.1  # corners closer than this to the camera plane are treated as behind it


@dataclass
class GroundTruthObject:
    actor_id: int
    class_name: str
    box: np.ndarray  # visible xyxy
    amodal_box: np.ndarray  # projected 3D box, clipped to the image
    visible_px: int
    distance_m: (
        float  # forward distance from the camera to the box's nearest point (what braking cares about)
    )
    velocity: np.ndarray  # world frame, m/s

    @property
    def object_id(self) -> int:
        """Unique per object in a frame, for tracking evaluation: the actor id, negated for a rider (the
        same CARLA actor as its bicycle or motorcycle)."""
        return -self.actor_id if self.class_name == "rider" else self.actor_id


def decode_instance(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instance segmentation image (BGR, as saved) -> (semantic tag, instance id) per pixel.

    CARLA puts the tag in red and the instance id in green (low byte) and blue (high byte). The id is stable
    per object but unrelated to the actor id; see match_frame.
    """
    tag = bgr[..., 2]
    ids = bgr[..., 1].astype(np.uint16) | (bgr[..., 0].astype(np.uint16) << 8)
    return tag, ids


def decode_depth(bgr: np.ndarray) -> np.ndarray:
    """CARLA depth camera image (BGR, as saved) -> depth in metres along the camera's forward axis."""
    r, g, b = (bgr[..., c].astype(np.float32) for c in (2, 1, 0))
    return (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1) * 1000.0


def world_to_camera(points: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points (N, 3) -> camera coordinates (N, 3) in our convention: x right, y down, z forward.

    CARLA's camera axes are x forward, y right, z up.
    """
    homogeneous = np.c_[points, np.ones(len(points))]
    x_fwd, y_right, z_up = (np.linalg.inv(cam_to_world) @ homogeneous.T)[:3]
    return np.stack([y_right, -z_up, x_fwd], axis=1)


def amodal_box(corners_cam: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Image box of a 3D box's corners (camera coordinates), clipped to the image; None if not in view.

    Corners behind the camera are dropped, which makes the box an approximation for objects that are
    partly behind the camera plane (e.g. alongside the ego vehicle).
    """
    front = corners_cam[corners_cam[:, 2] > MIN_DEPTH_M]
    if len(front) == 0:
        return None
    uv = (K @ front.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1 = np.clip(uv.min(axis=0), 0, [width, height])
    x2, y2 = np.clip(uv.max(axis=0), 0, [width, height])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return np.array([x1, y1, x2, y2], np.float32)


def match_frame(
    tag: np.ndarray,
    ids: np.ndarray,
    depth: np.ndarray | None,
    actors: pd.DataFrame,
    cam_to_world: np.ndarray,
    camera: CameraInfo,
    min_px: int = 50,
    min_score: float = 0.5,
    box_padding: float = 0.1,
    depth_margin_m: float = 1.0,
) -> dict[int, int | None]:
    """One frame's best actor for each instance id on a vehicle or person (None: no actor fits).

    An actor's score is the fraction of the id's pixels inside its amodal box (padded by box_padding of the
    box size, since CARLA's 3D boxes can be slightly smaller than the mesh) whose depth lies within its 3D
    box's depth range +- depth_margin_m; the depth check keeps a parked car seen through a moving car's box
    from matching it. People only match pedestrian pixels and vehicles the rest: the class itself can't be
    checked, since CARLA tags some trucks (by blueprint) as cars. Ids with fewer than min_px pixels are left
    out, the best score must reach min_score.
    """
    candidates = []
    for actor in actors.itertuples(index=False):
        corners_cam = world_to_camera(np.asarray(actor.corners, np.float64).reshape(8, 3), cam_to_world)
        amodal = amodal_box(corners_cam, camera.K, camera.width, camera.height)
        if amodal is None:
            continue
        pad = box_padding * (amodal[2:] - amodal[:2])
        near, far = corners_cam[:, 2].min() - depth_margin_m, corners_cam[:, 2].max() + depth_margin_m
        is_person = actor.class_name == "person"
        candidates.append((int(actor.actor_id), is_person, amodal[:2] - pad, amodal[2:] + pad, near, far))

    ys, xs = np.nonzero(np.isin(tag, DYNAMIC_TAGS))
    values, inverse, counts = np.unique(ids[ys, xs], return_inverse=True, return_counts=True)
    groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    matches = {}
    for value, group in zip(values, groups, strict=True):
        if len(group) < min_px:
            continue
        x, y = xs[group], ys[group]
        is_person = np.bincount(tag[y, x]).argmax() == PEDESTRIAN_TAG
        d = depth[y, x] if depth is not None else None
        scores = []
        for actor_id, person, (x1, y1), (x2, y2), near, far in candidates:
            if person != is_person:
                continue
            inside = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
            if d is not None:
                inside &= (d >= near) & (d <= far)
            scores.append((inside.mean(), actor_id))
        score, actor_id = max(scores, default=(0.0, None))
        matches[int(value)] = actor_id if score >= min_score else None
    return matches


def match_instances(frame_matches: Iterable[dict[int, int | None]], min_share: float = 0.5) -> dict[int, int]:
    """Recording-wide instance id -> actor id from every frame's match_frame.

    An id gets the actor it matched most often, if that happened in at least min_share of the frames the id
    was seen in; ids that mostly matched nothing (parked map cars) stay unmatched.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for matches in frame_matches:
        for value, actor_id in matches.items():
            votes[value][actor_id] += 1
    mapping = {}
    for value, counter in votes.items():
        matched = Counter({actor_id: n for actor_id, n in counter.items() if actor_id is not None})
        if matched:
            actor_id, n = matched.most_common(1)[0]
            if n >= min_share * counter.total():
                mapping[value] = actor_id
    return mapping


def actor_id_image(ids: np.ndarray, instance_to_actor: dict[int, int]) -> np.ndarray:
    """Per-pixel actor id (-1: none) from an instance id image and match_instances' mapping."""
    lookup = np.full(1 << 16, -1, np.int64)
    for value, actor_id in instance_to_actor.items():
        lookup[value] = actor_id
    return lookup[ids]


def ground_truth(
    tag: np.ndarray,
    actor_ids: np.ndarray,
    actors: pd.DataFrame,
    cam_to_world: np.ndarray,
    camera: CameraInfo,
    min_visible_px: int = 100,
) -> list[GroundTruthObject]:
    """2D ground truth for one frame from its per-pixel actor ids (actor_id_image) and the actors' 3D boxes.

    An actor's visible pixels are those with its id and a dynamic-object tag. A bicycle's or motorcycle's
    rider pixels become a separate "rider" object, as BDD100K (and so the detector) labels them; it shares
    the actor's id, amodal box, distance and velocity.
    """
    dynamic = np.isin(tag, DYNAMIC_TAGS)
    rider = tag == RIDER_TAG
    objects = []
    for actor in actors.itertuples(index=False):
        corners_cam = world_to_camera(np.asarray(actor.corners, np.float64).reshape(8, 3), cam_to_world)
        amodal = amodal_box(corners_cam, camera.K, camera.width, camera.height)
        if amodal is None:
            continue
        mask = dynamic & (actor_ids == actor.actor_id)
        for class_name, part in ((actor.class_name, mask & ~rider), ("rider", mask & rider)):
            visible_px = int(part.sum())
            if visible_px < min_visible_px:
                continue
            ys, xs = np.nonzero(part)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            objects.append(
                GroundTruthObject(
                    actor_id=int(actor.actor_id),
                    class_name=class_name,
                    box=box,
                    amodal_box=amodal,
                    visible_px=visible_px,
                    distance_m=float(corners_cam[:, 2].min()),
                    velocity=np.asarray(actor.velocity, np.float32),
                )
            )
    return objects


def unlabelled_boxes(tag: np.ndarray, ids: np.ndarray, known_ids, min_px: int = 20) -> np.ndarray:
    """(K, 4) xyxy boxes around vehicle/person pixels whose instance id is none of known_ids.

    One box per connected region of each instance id, so map objects that share an id (if CARLA gives them
    one) still get separate boxes.
    """
    mask = np.isin(tag, DYNAMIC_TAGS) & ~np.isin(ids, np.asarray(list(known_ids), np.uint16))
    boxes = []
    for value in np.unique(ids[mask]):
        region = (mask & (ids == value)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        for x, y, w, h, area in stats[1:]:  # row 0 is the background
            if area >= min_px:
                boxes.append([x, y, x + w, y + h])
    return np.array(boxes, np.float32).reshape(-1, 4)


@register("source", "carla_recording")
class CarlaRecording:
    """A recorded sequence: iterate it as a FrameSource, and ask for each frame's ground truth."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.frames = pd.read_parquet(self.root / "frames.parquet").set_index("index").sort_index()
        actors = pd.read_parquet(self.root / "actors.parquet")
        self._actors = dict(tuple(actors.groupby("index", sort=False)))
        self._no_actors = actors.iloc[:0]
        cam = self.meta["camera"]
        self.camera = CameraInfo.from_fov(cam["width"], cam["height"], cam["fov"], cam["mount"]["z"])

    def __len__(self) -> int:
        return len(self.frames)

    def _image(self, folder: str, index: int, ext: str) -> np.ndarray | None:
        path = self.root / folder / f"{FRAME_NAME.format(index)}.{ext}"
        return cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None

    def frame(self, index: int, with_depth: bool = True) -> Frame:
        row = self.frames.loc[index]
        depth = self._image(DEPTH_DIR, index, "png") if with_depth else None
        return Frame(
            image=self._image(RGB_DIR, index, "jpg"),
            timestamp=float(row.timestamp),
            frame_id=int(index),
            camera=self.camera,
            depth_gt=decode_depth(depth) if depth is not None else None,
        )

    def __iter__(self) -> Iterator[Frame]:
        for index in self.frames.index:
            yield self.frame(int(index))

    def instance(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(semantic tag, instance id) per pixel of a frame's instance segmentation image."""
        return decode_instance(self._image(INSTANCE_DIR, index, "png"))

    def actor_ids(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(semantic tag, actor id or -1) per pixel of a frame's instance segmentation image."""
        tag, ids = self.instance(index)
        return tag, actor_id_image(ids, self.instance_to_actor)

    @cached_property
    def instance_to_actor(self) -> dict[int, int]:
        """Instance id -> actor id for the whole recording; built on first use from every frame's images."""
        return match_instances(self._match_frame(int(index)) for index in self.frames.index)

    def _match_frame(self, index: int) -> dict[int, int | None]:
        tag, ids = self.instance(index)
        depth = self._image(DEPTH_DIR, index, "png")
        depth = decode_depth(depth) if depth is not None else None
        actors = self._actors.get(index, self._no_actors)
        return match_frame(tag, ids, depth, actors, self._cam_to_world(index), self.camera)

    def _cam_to_world(self, index: int) -> np.ndarray:
        return np.asarray(self.frames.loc[index, "cam_to_world"], np.float64).reshape(4, 4)

    def ground_truth(self, index: int, min_visible_px: int = 100) -> list[GroundTruthObject]:
        tag, actor_ids = self.actor_ids(index)
        actors = self._actors.get(index, self._no_actors)
        return ground_truth(tag, actor_ids, actors, self._cam_to_world(index), self.camera, min_visible_px)

    def unlabelled_boxes(self, index: int, min_px: int = 20) -> np.ndarray:
        """Vehicle/person regions without a recorded actor in this frame (actors move out of range too)."""
        tag, ids = self.instance(index)
        present = set(self._actors.get(index, self._no_actors).actor_id)
        known = [value for value, actor_id in self.instance_to_actor.items() if actor_id in present]
        return unlabelled_boxes(tag, ids, known, min_px)
