from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np


import omni.usd
from omni.kit.app import get_app
from pxr import Sdf
from isaacsim.core.api.sensors.base_sensor import BaseSensor
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils import prims as prim_utils


from isaacsim.zivid.cameras.calibration import CameraCalibration
from isaacsim.zivid.cameras.models import ZividCameraModelName
from isaacsim.zivid.cameras.zivid_camera_model import ZividCameraModel
from isaacsim.zivid.cameras.resolution import Resolution, SamplingMode, get_camera_resolution, get_projector_resolution
from isaacsim.zivid.utilities.omniverse_camera_params import (
    apply_model_params_to_camera,
    lock_camera,
    hide_camera_icon,
)
from isaacsim.zivid.utilities.transforms import Rotation, Transform, transform_points


class ZividCamera(BaseSensor):

    def __init__(
        self, model_name: ZividCameraModelName, prim_path: str, sampling_mode: SamplingMode = SamplingMode.FULL
    ):

        self._camera_prim_path = prim_path
        self._camera_sensor_prim_path = self._camera_prim_path + "/camera_sensor"
        self._projector_prim_path = self._camera_prim_path + "/projector"
        self._sampling_mode = sampling_mode
        self._model_name = model_name
        self._aparture = _stops_to_f_number(4)

        self._case = SingleXFormPrim(prim_path)

        super().__init__(
            prim_path=self._camera_prim_path,
            name="zivid_camera",
        )

        self._setup_sensors()

    def _setup_sensors(self) -> None:

        self._camera_resolution = get_camera_resolution(self._sampling_mode, self._model_name)
        self._projector_resolution = get_projector_resolution(self._model_name)
        self._camera_model = ZividCameraModel.from_model_name(self._model_name, self._sampling_mode)
        self._camera_sensor = Camera(
            prim_path=self._camera_sensor_prim_path,
            name="camera_sensor",
            resolution=self._camera_resolution.as_tuple(),
            orientation=(self._camera_model.optical_frame.rot * Rotation.y_down_z_forward().inverse())
            .to_quat()
            .tolist(),
            translation=self._camera_model.optical_frame.t.tolist(),
        )

        self._projector = Camera(
            prim_path=self._projector_prim_path,
            name="projector",
            resolution=self._projector_resolution.as_tuple(),
            translation=self._camera_model.projector_frame().t.tolist(),
            orientation=(self._camera_model.projector_frame().rot * Rotation.y_down_z_forward().inverse())
            .to_quat()
            .tolist(),
        )

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize the camera."""
        super().initialize(physics_sim_view)
        self._camera_sensor.initialize(physics_sim_view)

        self._camera_sensor.set_resolution(self._camera_resolution.as_tuple())
        self._camera_sensor.set_lens_aperture(self._aparture)
        apply_model_params_to_camera(
            camera=self._camera_sensor,
            model_params=self._camera_model.camera_sensor_model,
        )
        self._camera_sensor.add_distance_to_image_plane_to_frame()
        self._camera_sensor.add_distance_to_camera_to_frame()
        lock_camera(self._camera_sensor)
        hide_camera_icon(self._camera_sensor)

        self._projector.initialize(physics_sim_view)

        self._projector.set_resolution(self._projector_resolution.as_tuple())
        self._camera_sensor.set_lens_aperture(0)  ## zero aperture for projector
        apply_model_params_to_camera(
            camera=self._projector,
            model_params=self._camera_model.projector_sensor_model,
        )
        self._projector.add_distance_to_image_plane_to_frame()
        self._projector.add_distance_to_camera_to_frame()
        lock_camera(self._projector)
        hide_camera_icon(self._projector)

    def set_camera_model(self, model_name: ZividCameraModelName) -> None:
        if model_name != self._model_name:
            self._model_name = model_name
            self._setup_sensors()
            self.initialize()

    def set_sampling_mode(self, sampling_mode: SamplingMode) -> None:
        self._sampling_mode = sampling_mode
        self._camera_resolution = get_camera_resolution(self._sampling_mode, self._model_name)
        self._projector_resolution = get_projector_resolution(self._model_name)
        self._camera_model = ZividCameraModel.from_model_name(
            model_name=self._model_name,
            sampling_mode=self._sampling_mode,
        )
        self._camera_sensor.set_resolution(self._camera_resolution.as_tuple())
        apply_model_params_to_camera(
            camera=self._camera_sensor,
            model_params=self._camera_model.camera_sensor_model,
        )
        self._projector.set_resolution(self._projector_resolution.as_tuple())
        apply_model_params_to_camera(
            camera=self._projector,
            model_params=self._camera_model.projector_sensor_model,
        )

    def set_lens_aperture(self, stops: int) -> None:
        """Set the lens aperture of the camera."""
        self._aparture = _stops_to_f_number(stops)
        self._camera_sensor.set_lens_aperture(self._aparture)

    def get_camera_resolution(self):
        return self._camera_sensor.get_resolution()

    def get_data_rgb(self) -> np.ndarray:
        """Get the RGB data from the camera."""
        return self._camera_sensor.get_rgb()

    def get_data_xyz(self) -> np.ndarray:
        """Get the XYZ data from the camera."""

        depth = self._camera_sensor.get_depth()
        if depth is None:
            return np.full(
                (self._camera_resolution.height, self._camera_resolution.width, 3),
                np.nan,
                dtype=np.float32,
            )
        camera_matrix = self._camera_model.calibration.camera_model.camera_matrix
        valid_camera = np.isfinite(depth)
        xyz_camera = np.where(valid_camera[:, :, None], _unproject_depth(depth, camera_matrix), 1)

        depth = self._projector.get_depth()
        camera_matrix = self._camera_model.calibration.projector_model.camera_matrix
        valid = np.isfinite(depth)
        xyz_projector = np.where(valid[:, :, None], _unproject_depth(depth, camera_matrix), 1)

        filtered_xyz = _filter_invisible_points(
            xyz_camera,
            xyz_projector,
            self._camera_model.calibration,
        )

        return np.where(valid_camera[:, :, None], filtered_xyz, np.nan)

    def get_case_world_pose(self) -> Transform:
        """Get the case world pose."""
        p, q = self._case.get_world_pose()
        return Transform(
            t=p,
            rot=Rotation.from_quat(q),
        )

    def get_sensor_world_pose(self) -> Transform:
        """Get the sensor world pose."""
        case_t = self.get_case_world_pose()
        return case_t * self._camera_model.optical_frame

    def _get_senor_world_pose(self) -> Transform:
        """Get the sensor world pose."""
        p, q = self._camera_sensor.get_world_pose()
        return Transform(
            t=p,
            rot=Rotation.from_quat(q) * Rotation.y_down_z_forward(),
        )

    def _get_projector_world_pose(self) -> Transform:
        """Get the projector world pose."""
        p, q = self._projector.get_world_pose()
        return Transform(
            t=p,
            rot=Rotation.from_quat(q) * Rotation.y_down_z_forward(),
        )


def _stops_to_f_number(stops: int) -> float:
    """Convert stops to f-number."""
    match stops:
        case -5:
            return 32
        case -4:
            return 22
        case -3:
            return 16
        case -2:
            return 11
        case -1:
            return 8
        case 0:
            return 5.6
        case 1:
            return 4.0
        case 2:
            return 2.8
        case 3:
            return 3
        case 4:
            return 1.4
        case _:
            raise ValueError("Stops must be integer between -5 and 4")


def _unproject_depth(depth: NDArray, camera_matrix: CameraCalibration.LensModel.CameraMatrix) -> NDArray:
    """Unproject depth to 3D points."""
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - camera_matrix.cx) / camera_matrix.fx
    y = (y - camera_matrix.cy) / camera_matrix.fy
    z = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    return np.stack((x * z, y * z, z), axis=-1)


def _project_to_camera(xyz: NDArray, camera_matrix: CameraCalibration.LensModel.CameraMatrix) -> NDArray:
    h, w = xyz.shape[:2]
    im = np.zeros((h, w, 2))
    im[..., 0] = xyz[..., 0] * camera_matrix.fx / xyz[..., 2] + camera_matrix.cx
    im[..., 1] = xyz[..., 1] * camera_matrix.fy / xyz[..., 2] + camera_matrix.cy
    return im


def _filter_invisible_points(xyz_camera: NDArray, xyz_projector: NDArray, calibration: CameraCalibration) -> NDArray:

    camera_points_in_projector_frame = transform_points(xyz_camera, calibration.extriniscs.inverse())
    camera_points_in_projector_plane = _project_to_camera(
        camera_points_in_projector_frame, calibration.projector_model.camera_matrix
    )
    h, w, _ = xyz_projector.shape

    out_of_fov = (
        (camera_points_in_projector_plane[..., 0] < 0)
        | (camera_points_in_projector_plane[..., 0] >= w)
        | (camera_points_in_projector_plane[..., 1] < 0)
        | (camera_points_in_projector_plane[..., 1] >= h)
    ) | (camera_points_in_projector_frame[:, :, 2] < 0)

    cliped_camera_points_in_projector_plane = camera_points_in_projector_plane.clip([0, 0], [w - 1, h - 1])

    dx = cliped_camera_points_in_projector_plane[..., 0] % 1
    dy = cliped_camera_points_in_projector_plane[..., 1] % 1
    cliped_camera_points_in_projector_plane_int = cliped_camera_points_in_projector_plane.astype(int)
    p11 = xyz_projector[
        cliped_camera_points_in_projector_plane_int[..., 1],
        cliped_camera_points_in_projector_plane_int[..., 0],
        2,
    ]
    p12 = xyz_projector[
        cliped_camera_points_in_projector_plane_int[..., 1],
        np.clip(cliped_camera_points_in_projector_plane_int[..., 0] + 1, a_min=0, a_max=w - 1),
        2,
    ]
    p21 = xyz_projector[
        np.clip(cliped_camera_points_in_projector_plane_int[..., 1] + 1, a_min=0, a_max=h - 1),
        cliped_camera_points_in_projector_plane_int[..., 0],
        2,
    ]
    p22 = xyz_projector[
        np.clip(cliped_camera_points_in_projector_plane_int[..., 1] + 1, a_min=0, a_max=h - 1),
        np.clip(cliped_camera_points_in_projector_plane_int[..., 0] + 1, a_min=0, a_max=w - 1),
        2,
    ]

    z1 = p11 * (1 - dx) + p12 * dx
    z2 = p21 * (1 - dx) + p22 * dx
    z_proj = z1 * (1 - dy) + z2 * dy
    diff = np.abs(camera_points_in_projector_frame[:, :, 2] - z_proj)
    projector_shadow = diff > 5e-3 * camera_points_in_projector_frame[:, :, 2]

    mask = out_of_fov | projector_shadow
    return np.where(mask[:, :, None], np.nan, xyz_camera)
