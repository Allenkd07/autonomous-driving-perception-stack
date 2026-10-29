import cv2
import numpy as np
import pytest

from perception.core import registry
from perception.core.interfaces import FrameSource, Sink
from perception.core.types import LaneResult, ObjectState, PerceptionOutput, Track
from perception.io import ImageFolderSource, VideoSink, VideoSource
from perception.io.carla_recording import GroundTruthObject
from perception.viz.overlay import DANGER, UNLABELLED, draw_ground_truth, draw_output, track_colour


def write_video(path, frames: int, fps: float = 10.0, size=(160, 120)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    writer.release()


def test_image_folder_source(tmp_path):
    for name in ("b.png", "a.png", "c.jpg", "notes.txt"):
        if name.endswith("txt"):
            (tmp_path / name).write_text("not an image")
        else:
            cv2.imwrite(str(tmp_path / name), np.zeros((60, 80, 3), np.uint8))

    source = ImageFolderSource(tmp_path, fov_deg=90.0, height_m=1.2, pitch_deg=2.0, fps=5.0)
    frames = list(source)

    assert isinstance(source, FrameSource) and len(source) == 3
    assert [f.frame_id for f in frames] == [0, 1, 2] and frames[2].timestamp == pytest.approx(0.4)
    assert frames[0].camera.width == 80 and frames[0].camera.height_m == 1.2
    assert frames[0].camera.pitch_rad == pytest.approx(np.radians(2.0))


def test_video_source(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path, frames=5, fps=10.0)

    source = VideoSource(path)
    frames = list(source)

    assert source.frame_rate == pytest.approx(10.0)
    assert len(frames) == 5 and frames[0].image.shape == (120, 160, 3)
    assert frames[4].timestamp == pytest.approx(0.4)
    with pytest.raises(FileNotFoundError):
        list(VideoSource(tmp_path / "missing.mp4"))


def output(make_frame, ttc=None, in_lane=True, with_lanes=True) -> PerceptionOutput:
    frame = make_frame()
    track = Track(
        box=np.array([300, 200, 400, 300], np.float32), score=0.9, class_id=0, class_name="car", track_id=3
    )
    lanes = None
    if with_lanes:
        mask = np.zeros(frame.image.shape[:2], bool)
        mask[400:, 300:500] = True
        lanes = LaneResult(
            left=np.array([0, 0, 300.0]), right=np.array([0, 0, 500.0]), ego_lane_mask=mask, confidence=1
        )
    state = ObjectState(track=track, distance_m=12.3, ttc_s=ttc, in_ego_lane=in_lane)
    return PerceptionOutput(
        frame=frame, objects=[state], lanes=lanes, timings_ms={"detect": 20.0, "track": 5.0}
    )


def test_overlay_draws_boxes_lanes_and_warnings(make_frame):
    calm = draw_output(output(make_frame, ttc=5.0))
    danger = draw_output(output(make_frame, ttc=1.0))

    assert calm.shape == (600, 800, 3)
    assert tuple(calm[250, 300]) == track_colour(3)  # box edge in the track's colour
    assert tuple(danger[250, 300]) == DANGER  # in lane and TTC below 2 s
    assert calm[500, 400].any()  # tinted ego lane on a black frame
    assert not draw_output(output(make_frame, with_lanes=False))[500, 400].any()


def test_ground_truth_overlay_tints_actor_and_unlabelled_pixels():
    image = np.zeros((200, 300, 3), np.uint8)
    actor_ids = np.full((200, 300), -1)
    actor_ids[60:80, 20:40] = 7  # the actor (its label goes above y = 60)
    dynamic = actor_ids >= 0
    dynamic[120:140, 200:240] = True  # a parked car: no actor
    actor = GroundTruthObject(
        actor_id=7,
        class_name="car",
        box=np.array([20, 60, 40, 80], np.float32),
        amodal_box=np.array([15, 55, 45, 85], np.float32),
        visible_px=400,
        distance_m=12.0,
        velocity=np.zeros(3, np.float32),
    )
    parked = np.array([[200, 120, 240, 140]], np.float32)

    drawn = draw_ground_truth(image, [actor], parked, dynamic, actor_ids)

    half = (np.array(track_colour(7)) * 0.5).astype(np.uint8)
    np.testing.assert_array_equal(drawn[70, 30], half)  # inside the actor: tinted in its colour
    np.testing.assert_array_equal(drawn[130, 220], (np.array(UNLABELLED) * 0.5).astype(np.uint8))
    assert tuple(drawn[70, 15]) == track_colour(7)  # amodal box edge
    assert tuple(drawn[130, 200]) == UNLABELLED  # unlabelled box edge
    assert not drawn[180, 150].any()


def test_video_sink_writes_overlays(tmp_path, make_frame):
    sink = VideoSink(tmp_path / "out" / "overlay.mp4", fps=10.0)
    for _ in range(3):
        sink.write(output(make_frame))
    sink.close()

    capture = cv2.VideoCapture(str(tmp_path / "out" / "overlay.mp4"))
    assert isinstance(sink, Sink) and sink.frames == 3
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()


def test_registered():
    assert registry.get("source", "video") is VideoSource
    assert registry.get("source", "images") is ImageFolderSource
    assert registry.get("sink", "video") is VideoSink
