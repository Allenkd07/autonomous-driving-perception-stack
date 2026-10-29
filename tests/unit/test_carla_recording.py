import itertools
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from perception.core.types import CameraInfo
from perception.io.carla_recording import (
    CarlaRecording,
    actor_id_image,
    amodal_box,
    decode_depth,
    decode_instance,
    ground_truth,
    match_frame,
    match_instances,
    unlabelled_boxes,
    world_to_camera,
)

CAR, PEDESTRIAN, RIDER, BICYCLE, ROAD = 14, 12, 13, 19, 1
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
    """BGR instance image with (instance_id, tag, (x1, y1, x2, y2)) regions painted in order."""
    image = np.zeros((H, W, 3), np.uint8)
    for instance_id, tag, (x1, y1, x2, y2) in regions:
        image[y1:y2, x1:x2] = (instance_id >> 8, instance_id & 0xFF, tag)
    return image


def depth_image(metres: float) -> np.ndarray:
    """BGR CARLA depth image, the same depth everywhere."""
    value = round(metres / 1000.0 * (256**3 - 1))
    return np.full((H, W, 3), (value >> 16, (value >> 8) & 0xFF, value & 0xFF), np.uint8)


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


def test_match_frame_by_box_depth_and_kind(camera):
    scene = actors(
        (7, "car", [20, 0, 0], [2, 1, 1]),  # amodal box about x 378-422, y 278-322; depth 18-22 m
        (9, "person", [10, 3, 0], [0.3, 0.3, 0.9]),  # about x 511-536, y 263-337; depth 9.7-10.3 m
    )
    regions = [
        (1007, CAR, (385, 285, 415, 315), 20.0),
        (1500, CAR, (380, 280, 384, 320), 40.0),  # parked car further back, seen inside the car's box
        (1009, PEDESTRIAN, (515, 270, 530, 330), 10.0),
        (1600, CAR, (515, 330, 530, 337), 10.0),  # vehicle pixels inside the person's box
        (1700, CAR, (700, 10, 703, 13), 20.0),  # 9 px: too small to match
    ]
    tag, ids = decode_instance(instance_image(*(r[:3] for r in regions)))
    depth = np.full((H, W), 50.0, np.float32)
    for _, _, (x1, y1, x2, y2), d in regions:
        depth[y1:y2, x1:x2] = d

    matches = match_frame(tag, ids, depth, scene, CAM_TO_WORLD, camera)

    assert matches == {1007: 7, 1500: None, 1009: 9, 1600: None}


def test_match_instances_votes_over_frames():
    frames = [
        {1: 7, 2: None},
        {1: 7, 2: 8, 4: 5},
        {1: None, 2: None, 3: None, 4: None},
        {1: 7},
    ]

    # 1: actor 7 in 3 of 4 frames; 2: actor 8 in 1 of 3 (a chance match); 3: never; 4: 1 of 2 (min_share 0.5)
    assert match_instances(frames, min_share=0.5) == {1: 7, 4: 5}


def test_actor_id_image():
    ids = np.array([[1007, 0], [1500, 1007]], np.uint16)
    np.testing.assert_array_equal(actor_id_image(ids, {1007: 7}), [[7, -1], [-1, 7]])


def test_ground_truth_visible_box_and_occlusion(camera):
    scene = actors(
        (7, "car", [20, 0, 0], [2, 1, 1]),  # far car: amodal box about x 378-422, y 278-322
        (8, "truck", [10, -1, 0], [1, 1, 1]),  # near truck covering the far car's left half
    )
    tag, ids = decode_instance(
        instance_image(
            (1007, CAR, (381, 281, 420, 320)),
            (1008, CAR, (316, 256, 400, 345)),  # painted over the far car's left part
        )
    )
    actor_ids = actor_id_image(ids, {1007: 7, 1008: 8})

    gt = {g.actor_id: g for g in ground_truth(tag, actor_ids, scene, CAM_TO_WORLD, camera, min_visible_px=50)}

    far = gt[7]
    np.testing.assert_array_equal(far.box, [400, 281, 420, 320])  # only the visible right part
    assert far.amodal_box[0] < 385 and far.amodal_box[2] > 415  # full extent
    assert far.visible_px == 20 * 39
    assert far.distance_m == pytest.approx(18.0)  # near face of a box centred 20 m ahead, 4 m long
    assert gt[8].class_name == "truck"


def test_ground_truth_drops_hidden_and_tiny_objects(camera):
    scene = actors((7, "car", [20, 0, 0], [2, 1, 1]), (9, "car", [30, 5, 0], [2, 1, 1]))
    tag, ids = decode_instance(instance_image((1007, CAR, (390, 290, 395, 295))))  # 25 px; 9 not visible
    actor_ids = actor_id_image(ids, {1007: 7})

    assert ground_truth(tag, actor_ids, scene, CAM_TO_WORLD, camera, min_visible_px=100) == []


def test_ground_truth_counts_only_dynamic_pixels_of_the_actor(camera):
    scene = actors((7, "car", [20, 0, 0], [2, 1, 1]))
    tag, ids = decode_instance(
        instance_image(
            (1007, CAR, (390, 290, 410, 310)),
            (1500, CAR, (380, 290, 390, 310)),  # an unmatched id inside the actor's box
            (1007, ROAD, (410, 290, 415, 310)),  # the actor's id, but a road pixel
        )
    )
    actor_ids = actor_id_image(ids, {1007: 7})

    (gt,) = ground_truth(tag, actor_ids, scene, CAM_TO_WORLD, camera, min_visible_px=50)

    np.testing.assert_array_equal(gt.box, [390, 290, 410, 310])
    assert gt.visible_px == 400


def test_ground_truth_splits_rider_from_bicycle(camera):
    scene = actors((46, "bicycle", [10, 0, 0], [1, 0.3, 0.8]))  # amodal box about x 387-413, y 264-336
    tag, ids = decode_instance(
        instance_image((1046, BICYCLE, (390, 310, 410, 335)), (1046, RIDER, (392, 268, 408, 310)))
    )
    actor_ids = actor_id_image(ids, {1046: 46})

    bicycle, rider = ground_truth(tag, actor_ids, scene, CAM_TO_WORLD, camera, min_visible_px=50)

    assert (bicycle.class_name, bicycle.actor_id, bicycle.object_id) == ("bicycle", 46, 46)
    assert (rider.class_name, rider.actor_id, rider.object_id) == ("rider", 46, -46)
    np.testing.assert_array_equal(bicycle.box, [390, 310, 410, 335])
    np.testing.assert_array_equal(rider.box, [392, 268, 408, 310])
    np.testing.assert_array_equal(rider.amodal_box, bicycle.amodal_box)


def test_recording_round_trip(tmp_path, camera):
    for folder in ("rgb", "instance", "depth"):
        (tmp_path / folder).mkdir()
    for i in range(2):
        cv2.imwrite(str(tmp_path / "rgb" / f"{i:06d}.jpg"), np.full((H, W, 3), 100, np.uint8))
        cv2.imwrite(
            str(tmp_path / "instance" / f"{i:06d}.png"), instance_image((1007, CAR, (390, 290, 410, 310)))
        )
        cv2.imwrite(str(tmp_path / "depth" / f"{i:06d}.png"), depth_image(20.0))
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
    np.testing.assert_allclose(frames[0].depth_gt, 20.0, rtol=1e-5)
    np.testing.assert_allclose(rec.camera.K, camera.K)
    assert rec.instance_to_actor == {1007: 7}  # matched in frame 1, the only frame with the actor
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
