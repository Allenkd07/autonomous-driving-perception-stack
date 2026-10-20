import numpy as np

from perception.core.types import CameraInfo


def test_camera_from_fov_90_deg():
    cam = CameraInfo.from_fov(width=800, height=600, fov_deg=90.0, height_m=1.6)

    # tan(45 deg) = 1, so the focal length equals half the image width
    np.testing.assert_allclose(cam.K[0, 0], 400.0)
    np.testing.assert_allclose(cam.K[1, 1], 400.0)
    np.testing.assert_allclose(cam.K[:2, 2], [400.0, 300.0])
