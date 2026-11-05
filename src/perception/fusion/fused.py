"""Ground plane and calibrated depth combined, each weighted by how far it can be trusted.

The two fail differently. The ground plane needs the box's contact with the road: it has nothing to go on
when the bottom is cut off by the image edge, and a pixel of box-bottom or pitch error costs more the
farther the object. A depth network sees any surface, but predicts inverse depth, so a constant error in
1/z becomes an error growing with the square of the distance. Each gives a standard deviation:

    ground plane   sigma^2 = (ground_rel_sigma * z)^2 + (z^2 / (f_y * camera height) * pixel_sigma)^2
    depth          sigma   = depth_inv_sigma * z^2

and the result is their inverse-variance weighted mean, the minimum-variance combination of two independent
estimates. When only one exists it is used. The defaults are the robust spreads measured on CARLA clean
objects (scripts/eval_distance.py) with Depth Anything V2 Small calibrated on the road: the ground plane is
within ~1 px (2-5 %), the network's relative error grows from ~5 % at 10 m to ~19 % at 45 m (0.005 z).
"""

from __future__ import annotations

from perception.core.registry import register
from perception.core.types import CameraInfo, DepthMap, Frame, LaneResult, Track
from perception.fusion.base import Measurement, TrackDistanceEstimator
from perception.fusion.depth_scale import DepthScaleDistance
from perception.fusion.ground_plane import GroundPlaneDistance


@register("distance", "fused")
class FusedDistance(TrackDistanceEstimator):
    """DistanceEstimator combining GroundPlaneDistance and DepthScaleDistance.

    ground_plane       GroundPlaneDistance parameters (auto_pitch); its pitch also places the depth
                       calibration's road samples
    depth_scale        DepthScaleDistance parameters (reference, box_quantile, ...)
    pixel_sigma        ground-plane uncertainty of a box's bottom row, in pixels
    ground_rel_sigma   ground-plane relative uncertainty that doesn't shrink with distance (box statistics,
                       road not quite flat)
    depth_inv_sigma    calibrated depth's uncertainty per metre of distance, relative (sigma = this * z^2);
                       depends on the network: measure it for each (MiDaS Small's is several times larger)
    edge_px            a box whose bottom is this close to the image's bottom edge is cut off: no ground plane
    other parameters: see TrackDistanceEstimator
    """

    def __init__(
        self,
        ground_plane: dict | None = None,
        depth_scale: dict | None = None,
        pixel_sigma: float = 1.0,
        ground_rel_sigma: float = 0.04,
        depth_inv_sigma: float = 0.005,
        edge_px: float = 2.0,
        **base,
    ) -> None:
        super().__init__(**base)
        self.ground_plane = GroundPlaneDistance(**(ground_plane or {}))
        self.depth_scale = DepthScaleDistance(**(depth_scale or {}))
        self.pixel_sigma = pixel_sigma
        self.ground_rel_sigma = ground_rel_sigma
        self.depth_inv_sigma = depth_inv_sigma
        self.edge_px = edge_px

    def reset(self) -> None:
        super().reset()
        self.ground_plane.reset()
        self.depth_scale.reset()

    def camera(self, lanes: LaneResult | None, frame: Frame) -> CameraInfo:
        return self.ground_plane.camera(lanes, frame)

    def measure(
        self,
        tracks: list[Track],
        depth: DepthMap | None,
        lanes: LaneResult | None,
        frame: Frame,
        camera: CameraInfo,
    ) -> list[Measurement]:
        plane = self.ground_plane.measure(tracks, depth, lanes, frame, camera)
        scaled = self.depth_scale.measure(tracks, depth, lanes, frame, camera)
        height = frame.image.shape[0]
        out = []
        for track, (z_g, x_g), (z_d, x_d) in zip(tracks, plane, scaled, strict=True):
            if z_g is not None and (float(track.box[3]) >= height - self.edge_px or z_g <= 0):
                z_g = None
            if z_g is None or z_d is None:
                out.append((z_g, x_g) if z_d is None else (z_d, x_d))
                continue
            var_g = (self.ground_rel_sigma * z_g) ** 2 + (
                z_g**2 / (camera.K[1, 1] * camera.height_m) * self.pixel_sigma
            ) ** 2
            var_d = (self.depth_inv_sigma * z_d**2) ** 2
            w = var_d / (var_g + var_d)  # the ground plane's weight
            out.append((w * z_g + (1 - w) * z_d, w * x_g + (1 - w) * x_d))
        return out
