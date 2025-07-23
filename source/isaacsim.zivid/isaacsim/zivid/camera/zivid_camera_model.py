from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from isaacsim.zivid.camera.camera_calibration import CameraCalibration
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.resolution import SamplingMode, get_camera_resolution, get_projector_resolution
from isaacsim.zivid.utilities.omniverse_camera_params import OmniverseCameraParams
from isaacsim.zivid.utilities.transforms import Rotation, Transform


@dataclass
class ZividCameraModel:
    name: ZividCameraModelName
    camera_sensor_model: OmniverseCameraParams
    projector_sensor_model: OmniverseCameraParams
    optical_frame: Transform
    calibration: CameraCalibration

    @classmethod
    def from_model_name(cls, model_name: ZividCameraModelName, sampling_mode: SamplingMode) -> ZividCameraModel:
        """Create a Zivid camera model from the given model name."""

        calibration = CameraCalibration.from_model_name(
            model_name=model_name,
            sampling_mode=sampling_mode,
        )
        camera_res = get_camera_resolution(sampling_mode, model_name=model_name)
        projector_res = get_projector_resolution(model_name=model_name)
        focus_distance = _focus_distance_from_model_name(model_name)

        return cls(
            name=model_name,
            camera_sensor_model=OmniverseCameraParams.from_intrinsic_matrix(
                intrinsic_matrix=calibration.camera_model.camera_matrix.as_array().flatten().tolist(),
                pixel_size=calibration.camera_model.pixel_size,
                width=camera_res.width,
                height=camera_res.height,
                clipping_range=(0.00, 10.0),
                focus_distance=focus_distance,
            ),
            projector_sensor_model=OmniverseCameraParams.from_intrinsic_matrix(
                intrinsic_matrix=calibration.projector_model.camera_matrix.as_array().flatten().tolist(),
                pixel_size=calibration.projector_model.pixel_size,
                width=projector_res.width,
                height=projector_res.height,
                clipping_range=(0.00, 10.0),
                focus_distance=focus_distance,
            ),
            optical_frame=cls.optical_frame_from_model_name(model_name),
            calibration=calibration,
        )

    @staticmethod
    def optical_frame_from_model_name(model_name: ZividCameraModelName) -> Transform:
        return Transform(
            _optical_center_from_model_name(model_name),
            _sensor_orientation_from_model_name(model_name),
        )

    def projector_frame(self) -> Transform:
        return self.optical_frame * self.calibration.extriniscs


def _optical_center_from_model_name(model_name: ZividCameraModelName) -> NDArray[np.float64]:
    """Get the optical center from the given model name."""
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return np.array([0.04802, 0.03084, 0.0295])
        case ZividCameraModelName.ZIVID_2_L100:
            return np.array([0.04802, 0.03084, 0.0295])
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return np.array([0.049, 0.03202, 0.0295])
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return np.array([0.049, 0.03202, 0.0295])
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return np.array([0.040, 0.031, 0.0295])
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return np.array([0.049, 0.03202, 0.0295])
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return np.array([0.049, 0.03202, 0.0295])
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return np.array([0.040, 0.031, 0.0295])
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")


def _sensor_orientation_from_model_name(model_name: ZividCameraModelName) -> Rotation:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            cam_yaw_deg = -3.0
        case ZividCameraModelName.ZIVID_2_L100:
            cam_yaw_deg = -3.0
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            cam_yaw_deg = -2.5
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            cam_yaw_deg = -2.5
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            cam_yaw_deg = 0.0
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            cam_yaw_deg = -2.5
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            cam_yaw_deg = -2.5
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            cam_yaw_deg = 0.0
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")

    cam_yaw_rad = cam_yaw_deg * np.pi / 180.0
    return (
        Rotation.from_quat(
            np.array(
                [np.cos(cam_yaw_rad / 2), 0.0, 0.0, np.sin(cam_yaw_rad / 2)],
            )
        )
        * Rotation.y_down_z_forward()
    )


def _focus_distance_from_model_name(model_name: ZividCameraModelName) -> float:
    """Get the focus distance from the given model name."""
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return 0.7
        case ZividCameraModelName.ZIVID_2_L100:
            return 1.0
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 0.6
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 1.1
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 1.3
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 0.6
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 1.1
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 1.3
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")
