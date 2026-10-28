import itertools
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from perception.core.types import CameraInfo
from perception.io.carla_recording import (
    CarlaRecording,
    amodal_box,
    decode_depth,
    decode_instance,
    ground_truth,
    unlabelled_boxes,
    world_to_camera,
)

CAR, ROAD = 14, 1
W, H = 800, 600
# CARLA camera at the world origin looking along +x (CARLA camera axes: x forward, y right, z up)
CAM_TO_WORLD = np.eye(4)


@pytest.fixture
def camera():
    return CameraInfo.from_fov(W, H, 90.0, 1.6)  # focal length 400 px, centre (400, 300)


def corners(centre, half_extent) -> np.ndarray:
    """The 8 world corners of an axis-aligned 3D box."""
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    return np.asarray(centre, float) + signs * np.asarray(half_extent, float)


def actors(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"actor_id": i, "class_name": c, "corners": corners(centre, ext).ravel(), "velocity": np.zeros(3)}
            for i, c, centre, ext in rows
        ]
    )


def instance_image(*regions) -> np.ndarray:
    """BGR instance image with (actor_id, tag, (x1, y1, x2, y2)) regions painted in order."""
    image = np.zeros((H, W, 3), np.uint8)
    for actor_id, tag, (x1, y1, x2, y2) in regions:
        image[y1:y2, x1:x2] = (actor_id >> 8, actor_id & 0xFF, tag)
    return image


def test_world_to_camera_axes():
    np.testing.assert_allclose(
        world_to_camera(np.array([[10.0, 2.0, 1.0]]), CAM_TO_WORLD), [[2.0, -1.0, 10.0]]
    )


def test_decode_instance_and_depth():
    tag, ids = decode_instance(instance_image((0x1234, CAR, (0, 0, 1, 1))))
    assert tag[0, 0] == CAR and ids[0, 0] == 0x1234 and ids[1, 1] == 0

    far = np.full((1, 1, 3), 255, np.uint8)
    assert decode_depth(far)[0, 0] == pytest.approx(1000.0)
    assert decode_depth(np.zeros((1, 1, 3), np.uint8))[0, 0] == 0.0


def test_amodal_box(camera):
    # 2 m wide, 2 m tall box 10 m ahead, centred: 400 * 1 / 10 = 40 px on each side of the centre at its
    # near face (9 m: 44.4 px), so the box spans the near face
    box = amodal_box(world_to_camera(corners([10, 0, 0], [1, 1, 1]), CAM_TO_WORLD), camera.K, W, H)
    np.testing.assert_allclose(box, [400 - 400 / 9, 300 - 400 / 9, 400 + 400 / 9, 300 + 400 / 9], rtol=1e-5)

    behind = world_to_camera(corners([-10, 0, 0], [1, 1, 1]), CAM_TO_WORLD)
    assert amodal_box(behind, camera.K, W, H) is None
    off_left = world_to_camera(corners([10, -30, 0], [1, 1, 1]), CAM_TO_WORLD)
    assert amodal_box(off_left, camera.K, W, H) is None


def test_ground_truth_visible_box_and_occlusion(camera):
    scene = actors(
        (7, "car", [20, 0, 0], [2, 1, 1]),  # far car: amodal box about x 378-422, y 278-322
        (8, "truck", [10, -1, 0], [1, 1, 1]),  # near truck covering the far car's left half
    )
    tag, ids = decode_instance(
        instance_image(
            (7, CAR, (381, 281, 420, 320)),
            (8, CAR, (316, 256, 400, 345)),  # painted over the far car's left part
        )
    )

    gt = {g.actor_id: g for g in ground_truth(tag, ids, scene, CAM_TO_WORLD, camera, min_visible_px=50)}

    far = gt[7]
    np.testing.assert_array_equal(far.box, [400, 281, 420, 320])  # only the visible right part
    assert far.amodal_box[0] < 385 and far.amodal_box[2] > 415  # full extent
    assert far.visible_px == 20 * 39
    assert far.distance_m == pytest.approx(18.0)  # near face of a box centred 20 m ahead, 4 m long
    assert gt[8].class_name == "truck"


def test_ground_truth_drops_hidden_and_tiny_objects(camera):
    scene = actors((7, "car", [20, 0, 0], [2, 1, 1]), (9, "car", [30, 5, 0], [2, 1, 1]))
    tag, ids = decode_instance(instance_image((7, CAR, (390, 290, 395, 295))))  # 25 px; actor 9 not visible

    assert ground_truth(tag, ids, scene, CAM_TO_WORLD, camera, min_visible_px=100) == []


def test_ground_truth_ignores_pixels_outside_the_box_or_not_dynamic(camera):
    scene = actors((7, "car", [20, 0, 0], [2, 1, 1]))
    tag, ids = decode_instance(
        instance_image(
            (7, CAR, (390, 290, 410, 310)),
            (7, CAR, (0, 0, 100, 100)),  # same id far outside the actor's box
            (7, ROAD, (410, 290, 415, 310)),  # same id, but a road pixel
        )
    )

    (gt,) = ground_truth(tag, ids, scene, CAM_TO_WORLD, camera, min_visible_px=50)

    np.testing.assert_array_equal(gt.box, [390, 290, 410, 310])
    assert gt.visible_px == 400


def test_recording_round_trip(tmp_path, camera):
    for folder in ("rgb", "instance", "depth"):
        (tmp_path / folder).mkdir()
    for i in range(2):
        cv2.imwrite(str(tmp_path / "rgb" / f"{i:06d}.jpg"), np.full((H, W, 3), 100, np.uint8))
        cv2.imwrite(
            str(tmp_path / "instance" / f"{i:06d}.png"), instance_image((7, CAR, (390, 290, 410, 310)))
        )
        cv2.imwrite(str(tmp_path / "depth" / f"{i:06d}.png"), np.full((H, W, 3), 255, np.uint8))
    pd.DataFrame(
        [
            {
                "index": i,
                "carla_frame": 100 + i,
                "timestamp": i * 0.05,
                "cam_to_world": CAM_TO_WORLD.ravel(),
                "ego_velocity": np.zeros(3, np.float32),
            }
            for i in range(2)
        ]
    ).to_parquet(tmp_path / "frames.parquet", index=False)
    scene = actors((7, "car", [20, 0, 0], [2, 1, 1]))
    pd.concat([scene.assign(index=1)]).to_parquet(tmp_path / "actors.parquet", index=False)  # in frame 1 only
    mount = {"x": 1.5, "y": 0.0, "z": 1.6, "pitch": 0.0}
    meta = {"camera": {"width": W, "height": H, "fov": 90.0, "mount": mount}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    rec = CarlaRecording(tmp_path)
    frames = list(rec)

    assert len(rec) == 2 and [f.frame_id for f in frames] == [0, 1]
    assert frames[1].image.shape == (H, W, 3) and frames[1].timestamp == pytest.approx(0.05)
    np.testing.assert_allclose(frames[0].depth_gt, 1000.0, rtol=1e-6)
    np.testing.assert_allclose(rec.camera.K, camera.K)
    assert rec.ground_truth(0) == []
    assert [g.actor_id for g in rec.ground_truth(1)] == [7]
    assert rec.frame(1, with_depth=False).depth_gt is None
    np.testing.assert_array_equal(rec.unlabelled_boxes(0), [[390, 290, 410, 310]])  # no actor in frame 0
    assert len(rec.unlabelled_boxes(1)) == 0


def test_unlabelled_boxes_split_regions_and_skip_known_actors():
    tag, ids = decode_instance(
        instance_image(
            (7, CAR, (10, 10, 30, 30)),  # recorded actor
            (0, CAR, (100, 100, 140, 120)),  # two parked cars sharing id 0, apart
            (0, CAR, (300, 100, 340, 120)),
            (0, ROAD, (500, 500, 600, 600)),  # not a vehicle
            (5, CAR, (700, 10, 703, 13)),  # 9 px: too small
        )
    )

    boxes = unlabelled_boxes(tag, ids, known_ids=[7], min_px=20)

    np.testing.assert_array_equal(boxes, [[100, 100, 140, 120], [300, 100, 340, 120]])
