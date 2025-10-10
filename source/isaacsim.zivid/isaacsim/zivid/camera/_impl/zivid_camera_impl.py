from __future__ import annotations

import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.zivid.camera._internal import ZividCameraModel
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.sampling import SamplingMode
from isaacsim.zivid.frame import Frame
from isaacsim.zivid.resolution import Resolution
from isaacsim.zivid.transforms import Rotation, Transform, transform_points
from numpy.typing import NDArray
from pxr import Sdf

from .camera_calibration import CameraCalibration
from .lensmodel import CameraMatrix
from .projection import project_to_camera
from .rgbd_sensor import RGBDSensor


class ZividCameraImpl:

    def __init__(
        self, model_name: ZividCameraModelName, prim_path: str, sampling_mode: SamplingMode = SamplingMode.FULL
    ):

        self._camera_prim_path = prim_path
        self._camera_sensor_prim_path = self._camera_prim_path + "/camera"
        self._projector_prim_path = self._camera_prim_path + "/projector"
        self._sampling_mode = sampling_mode
        self._model_name = model_name
        self._aperture = _get_closest_valid_aperture(1.4)
        self._camera_model = ZividCameraModel.from_model_name(self._model_name, self._sampling_mode)

        self._prim = SingleXFormPrim(prim_path)
        self._prim.prim.CreateAttribute("is_zivid_camera", Sdf.ValueTypeNames.Bool).Set(True)
        self._prim.prim.CreateAttribute("zivid_camera_model", Sdf.ValueTypeNames.Int).Set(int(model_name.value))

        self._camera_model = ZividCameraModel.from_model_name(self._model_name, self._sampling_mode)

        self._camera_sensor = RGBDSensor(
            prim_path=self._camera_sensor_prim_path,
            lens_model=self._camera_model.calibration.camera_model,
            pose=self._camera_model.optical_frame,
        )

        self._projector = RGBDSensor(
            prim_path=self._projector_prim_path,
            lens_model=self._camera_model.calibration.projector_model,
            pose=self._camera_model.projector_frame(),
        )

    def __del__(self):
        self.destroy()

    def destroy(self) -> None:
        self._camera_sensor.destroy()
        self._projector.destroy()

    def _configure_sensors(self) -> None:
        self._camera_model = ZividCameraModel.from_model_name(
            model_name=self._model_name,
            sampling_mode=self._sampling_mode,
        )
        self._camera_sensor.set_lens_model(self._camera_model.calibration.camera_model)
        self._projector.set_lens_model(self._camera_model.calibration.projector_model)
        self._camera_sensor.set_aperture(self._aperture)
        self._projector.set_aperture(0)

    def initialize(self, physics_sim_view=None) -> None:
        self._prim.initialize(physics_sim_view=physics_sim_view)
        self._camera_sensor.initialize(physics_sim_view=physics_sim_view)
        self._projector.initialize(physics_sim_view=physics_sim_view)

    def set_world_pose(self, pose: Transform) -> None:
        self._prim.set_world_pose(*pose.to_isaac_sim())

    def set_local_pose(self, pose: Transform) -> None:
        self._prim.set_local_pose(*pose.to_isaac_sim())

    def set_camera_model(self, model_name: ZividCameraModelName) -> None:
        if model_name != self._model_name:
            self._model_name = model_name
            self._configure_sensors()
            self._prim.prim.GetAttribute("zivid_camera_model").Set(int(model_name.value))

    def set_sampling_mode(self, sampling_mode: SamplingMode) -> None:
        self._sampling_mode = sampling_mode
        self._configure_sensors()

    def get_prim_path(self) -> str:
        return self._camera_prim_path

    def get_camera_sensor_prim_path(self) -> str:

        return self._camera_sensor.get_camera_prim_path()

    def get_projector_prim_path(self) -> str:

        return self._projector.get_camera_prim_path()

    def set_lens_aperture(self, fnumber: float) -> None:
        self._aperture = _get_closest_valid_aperture(fnumber)
        self._camera_sensor.set_aperture(self._aperture)

    def get_calibration(self) -> CameraCalibration:
        return self._camera_model.calibration

    def get_camera_model(self) -> ZividCameraModel:
        return self._camera_model

    def get_camera_resolution(self) -> Resolution:
        return self._camera_sensor.get_resolution()

    def get_frame(self) -> Frame:
        rgb = self._camera_sensor.get_data_rgb()
        xyz_camera = self._camera_sensor.get_data_xyz()
        xyz_projector = self._projector.get_data_xyz()
        calibration = self.get_calibration()

        mask = _points_visible(xyz_camera, xyz_projector, calibration)
        xyz = np.where(mask[..., None], xyz_camera, np.nan)

        return Frame(rgb=rgb, xyz=xyz)

    def get_world_pose(self) -> Transform:
        p, q = self._prim.get_world_pose()
        return Transform(
            t=p,
            rot=Rotation.from_quat(q),
        )

    def get_local_pose(self) -> Transform:
        p, q = self._prim.get_local_pose()
        return Transform(
            t=p,
            rot=Rotation.from_quat(q),
        )

    def get_camera_sensor_world_pose(self) -> Transform:
        return self._camera_sensor.get_world_pose()

    def get_projector_world_pose(self) -> Transform:
        return self._projector.get_world_pose()

    def get_camera_sensor(self) -> RGBDSensor:
        return self._camera_sensor

    def get_projector(self) -> RGBDSensor:
        return self._projector

    def get_focus_distance(self) -> float:
        return self._camera_sensor.get_lens_model().focus_distance

    def project_points(self, points: NDArray) -> NDArray:
        return project_to_camera(
            points,
            self._camera_model.calibration.camera_model.camera_matrix,
            self._camera_model.calibration.camera_model.distortion_coefficients,
        )

    @property
    def prim_path(self) -> str:
        return self._camera_prim_path

    @property
    def model_name(self) -> ZividCameraModelName:
        return self._camera_model.name


def _get_closest_valid_aperture(aperture: float) -> float:
    allowed_f_numbers = [1.4, 3.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0]
    closest_idx = min(range(len(allowed_f_numbers)), key=lambda i: abs(allowed_f_numbers[i] - aperture))
    return allowed_f_numbers[closest_idx]


def _project_to_camera(xyz: NDArray, camera_matrix: CameraMatrix) -> NDArray:
    h, w = xyz.shape[:2]
    im = np.zeros((h, w, 2))
    im[..., 0] = xyz[..., 0] * camera_matrix.fx / xyz[..., 2] + camera_matrix.cx
    im[..., 1] = xyz[..., 1] * camera_matrix.fy / xyz[..., 2] + camera_matrix.cy
    return im


def _points_visible(xyz_camera: NDArray, xyz_projector: NDArray, calibration: CameraCalibration) -> NDArray:

    camera_points_in_projector_frame = transform_points(xyz_camera, calibration.extriniscs.inverse())
    camera_points_in_projector_plane = _project_to_camera(
        camera_points_in_projector_frame, calibration.projector_model.camera_matrix
    )  # the point cloud is undistorted
    h, w, _ = xyz_projector.shape

    nan_mask = np.isnan(xyz_camera).any(axis=-1)

    out_of_fov = (
        (camera_points_in_projector_plane[..., 0] < 0)
        | (camera_points_in_projector_plane[..., 0] >= w)
        | (camera_points_in_projector_plane[..., 1] < 0)
        | (camera_points_in_projector_plane[..., 1] >= h)
        | (camera_points_in_projector_frame[:, :, 2] < 0)
    )

    camera_points_in_projector_plane[nan_mask] = 0  # ensure valid values, we discard them anyway

    cliped_camera_points_in_projector_plane = camera_points_in_projector_plane.clip([0, 0], [w - 1, h - 1])

    dx = cliped_camera_points_in_projector_plane[..., 0] % 1
    dy = cliped_camera_points_in_projector_plane[..., 1] % 1
    cliped_camera_points_in_projector_plane_int = cliped_camera_points_in_projector_plane.astype(np.int32)
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

    nan_mask |= np.isnan(z_proj)

    mask = out_of_fov | projector_shadow | nan_mask
    return ~mask
