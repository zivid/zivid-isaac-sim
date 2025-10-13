from __future__ import annotations

import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils import prims as prim_utils
from isaacsim.sensors.camera import USD_CAMERA_TENTHS_TO_STAGE_UNIT, Camera
from isaacsim.zivid.resolution import Resolution
from isaacsim.zivid.transforms import Rotation, Transform
from isaacsim.zivid.utilities.usd import hide_camera_icon, lock_camera
from numpy.typing import NDArray
from pxr import Gf

from .lensmodel import LensModel
from .projection import unproject_from_camera


class RGBDSensor:

    def __init__(self, prim_path: str, lens_model: LensModel, pose: Transform, is_local_pose: bool = True) -> None:

        self._camera_sensor_prim_path = prim_path + "/camera_sensor"
        self._lens_model = lens_model

        self._prim = SingleXFormPrim(
            prim_path,
            name="rgbd_sensor",
            orientation=pose.rot.as_quat().tolist(),
            **{"translation" if is_local_pose else "position": pose.t.tolist()},
        )

        self._camera_sensor = self._create_camera_sensor(self._camera_sensor_prim_path, "camera_sensor")

    def _create_camera_sensor(self, prim_path: str, name: str) -> Camera:
        # This is a hacky way to avoid warnings because that the camera  init methods sets a resolution
        # that does not match the default aspect ratio of the camera in USD
        prim = prim_utils.get_prim_at_path(prim_path) if prim_utils.is_prim_path_valid(prim_path) else None
        if prim and prim.GetTypeName() == "Camera":
            # reset to default aspect ratio if camera already exists
            prim.GetAttribute("verticalAperture").Set(15.2908)
            prim.GetAttribute("horizontalAperture").Set(20.955)

        dummy_resolution = (825, 602)  # Random resolution that corresponds to the default aspect ratio

        camera = Camera(
            prim_path=prim_path,
            name=name,
            orientation=(Rotation.y_down_z_forward().inverse()).as_quat().tolist(),
            translation=[0, 0, 0],
            resolution=dummy_resolution,
        )

        return camera

    def _configure_sensor(self) -> None:

        assert self._camera_sensor is not None
        self._camera_sensor.set_resolution(self._lens_model.resolution.as_tuple(), maintain_square_pixels=False)
        self._camera_sensor.set_lens_aperture(0)
        self._camera_sensor.set_clipping_range(0.0000000000001, 10)
        lock_camera(self._camera_sensor)
        hide_camera_icon(self._camera_sensor)
        self._camera_sensor.set_focal_length(
            (self._lens_model.camera_matrix.fx + self._lens_model.camera_matrix.fy) / 2 * self._lens_model.pixel_size
        )

        self._camera_sensor.set_vertical_aperture(
            self._lens_model.resolution.height * self._lens_model.pixel_size, maintain_square_pixels=False
        )
        vertical_aperture_offset = (
            self._lens_model.camera_matrix.cy - self._lens_model.resolution.height / 2
        ) * self._lens_model.pixel_size
        self._camera_sensor.prim.GetAttribute("verticalApertureOffset").Set(
            vertical_aperture_offset * USD_CAMERA_TENTHS_TO_STAGE_UNIT
        )

        self._camera_sensor.set_horizontal_aperture(
            self._lens_model.resolution.width * self._lens_model.pixel_size, maintain_square_pixels=False
        )
        horizontal_aperture_offset = (
            self._lens_model.camera_matrix.cx - self._lens_model.resolution.width / 2
        ) * self._lens_model.pixel_size
        self._camera_sensor.prim.GetAttribute("horizontalApertureOffset").Set(
            horizontal_aperture_offset * USD_CAMERA_TENTHS_TO_STAGE_UNIT
        )

        self._camera_sensor.set_focus_distance(self._lens_model.focus_distance)

        self._camera_sensor.set_opencv_pinhole_properties(
            fx=self._lens_model.camera_matrix.fx,
            fy=self._lens_model.camera_matrix.fy,
            cx=self._lens_model.camera_matrix.cx,
            cy=self._lens_model.camera_matrix.cy,
            pinhole=self._lens_model.distortion_coefficients.to_array().tolist(),
        )

        self._camera_sensor.prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Set(
            Gf.Vec2i(self._lens_model.resolution.as_tuple())
        )

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        self._camera_sensor = None

    def initialize(self, physics_sim_view=None) -> None:
        assert self._camera_sensor is not None
        should_attach_annotators = False
        if "rgb" not in self._camera_sensor._custom_annotators.keys():  # pylint: disable=W0212
            should_attach_annotators = True
        if "distance_to_image_plane" not in self._camera_sensor._custom_annotators.keys():  # pylint: disable=W0212
            should_attach_annotators = True

        assert self._camera_sensor is not None

        self._prim.initialize(physics_sim_view=physics_sim_view)
        self._camera_sensor.initialize(physics_sim_view=physics_sim_view, attach_rgb_annotator=False)
        self._configure_sensor()

        if should_attach_annotators:
            self._camera_sensor.add_rgb_to_frame()
            self._camera_sensor.add_distance_to_image_plane_to_frame()

    def set_aperture(self, fnumber: float) -> None:
        assert self._camera_sensor is not None
        self._camera_sensor.set_lens_aperture(fnumber)

    def set_local_pose(self, pose: Transform) -> None:
        self._prim.set_local_pose(*pose.to_isaac_sim())

    def set_world_pose(self, pose: Transform) -> None:
        self._prim.set_world_pose(*pose.to_isaac_sim())

    def set_lens_model(self, lensmodel: LensModel) -> None:
        self._lens_model = lensmodel
        self.initialize()

    def get_world_pose(self) -> Transform:
        """Get the world pose of the RGBD sensor."""
        return Transform.from_isaac_sim(self._prim.get_world_pose())

    def get_camera_prim_path(self) -> str:
        return self._camera_sensor_prim_path

    def get_data_rgb(self) -> NDArray:
        assert self._camera_sensor is not None
        return self._camera_sensor.get_rgb()

    def get_data_depth(self) -> NDArray:
        assert self._camera_sensor is not None
        return self._camera_sensor.get_depth()

    def get_data_xyz(self) -> NDArray:
        depth = self.get_data_depth()
        valid_camera = np.isfinite(depth)
        xyz_camera = np.where(
            valid_camera[:, :, None], unproject_from_camera(np.where(valid_camera, depth, 0), self._lens_model), 1
        )

        return np.where(valid_camera[:, :, None], xyz_camera, np.nan)

    def get_lens_model(self) -> LensModel:
        return self._lens_model

    def get_resolution(self) -> Resolution:
        assert self._camera_sensor is not None
        res = self._camera_sensor.get_resolution()
        return Resolution(width=res[0], height=res[1])

    def get_camera_sensor(self) -> Camera:
        return self._camera_sensor
