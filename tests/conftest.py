import numpy as np
import pytest

from perception.core.types import CameraInfo, Frame


@pytest.fixture
def camera() -> CameraInfo:
    return CameraInfo.from_fov(width=800, height=600, fov_deg=90.0, height_m=1.6)


@pytest.fixture
def make_frame(camera):
    def _make(frame_id: int = 0) -> Frame:
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        return Frame(image=image, timestamp=frame_id * 0.05, frame_id=frame_id, camera=camera)

    return _make
