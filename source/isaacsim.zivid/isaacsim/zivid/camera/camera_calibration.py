from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.resolution import SamplingMode, get_camera_resolution, get_projector_resolution
from isaacsim.zivid.utilities.transforms import Rotation, Transform
from numpy.typing import NDArray


@dataclass
class CameraCalibration:
    @dataclass
    class LensModel:
        @dataclass
        class CameraMatrix:
            fx: float
            fy: float
            cx: float
            cy: float

            @classmethod
            def from_array(cls, array: NDArray):
                return cls(
                    fx=array[0, 0],
                    fy=array[1, 1],
                    cx=array[0, 2],
                    cy=array[1, 2],
                )

            def as_array(self) -> NDArray:
                return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        camera_matrix: CameraMatrix
        pixel_size: float

    camera_model: LensModel
    projector_model: LensModel
    extriniscs: Transform

    @classmethod
    def from_model_name(cls, model_name: ZividCameraModelName, sampling_mode: SamplingMode) -> CameraCalibration:
        return cls(
            camera_model=_get_camera_lens_model(model_name, sampling_mode),
            projector_model=_get_projector_lens_model(model_name),
            extriniscs=_get_extrinsics(model_name),
        )


def _get_extrinsics(model_name: ZividCameraModelName) -> Transform:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            aa = [0, -0.113, 0]
            t = [111.2, 0, 9.7]
        case ZividCameraModelName.ZIVID_2_L100:
            aa = [0, -0.112, 0]
            t = [111.3, 0, 10.3]
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            aa = [0, -0.085, 0]
            t = [111.2, 0, 7.9]
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            aa = [0, -0.087, 0]
            t = [111.6, 0, 8.6]
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            aa = [0, -0.069, 0]
            t = [111.9, 0, 2.8]
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            aa = [0, -0.085, 0]
            t = [111.2, 0, 7.9]
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            aa = [0, -0.087, 0]
            t = [111.6, 0, 8.6]
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            aa = [0, -0.069, 0]
            t = [111.9, 0, 2.8]
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")
    rotation = Rotation.from_axis_angle(np.array(aa))
    translation = np.array(t) * 1e-3  # Convert mm to meters
    return Transform(t=translation, rot=rotation)


def _get_camera_lens_model(
    model_name: ZividCameraModelName, sampling_mode: SamplingMode
) -> CameraCalibration.LensModel:
    return _adjust_camera_lens_model(_get_full_res_camera_lens_model(model_name), sampling_mode)


def _get_projector_lens_model(model_name: ZividCameraModelName) -> CameraCalibration.LensModel:
    res = get_projector_resolution(model_name)
    f = _get_projector_focal_length_in_pixels(model_name)
    camera_mat = CameraCalibration.LensModel.CameraMatrix(
        fx=f,
        fy=f,
        cx=int(res.width / 2),
        cy=int(res.height / 2),
    )
    return _scale_lens_model(
        CameraCalibration.LensModel(
            pixel_size=_get_projector_pixel_size(model_name),
            camera_matrix=camera_mat,
        ),
        1,
    )


def _get_projector_pixel_size(model_name: ZividCameraModelName) -> float:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_L100:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 5.4 * 1e-6 * 4
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 5.4 * 1e-6 * 4
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")


def _get_camera_pixel_size(model_name: ZividCameraModelName) -> float:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_L100:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 2.74 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 2.74 * 1e-6
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")


def _get_full_res_camera_lens_model(model_name: ZividCameraModelName) -> CameraCalibration.LensModel:
    full_res = get_camera_resolution(SamplingMode.FULL, model_name)
    f = _get_camera_focal_length_in_pixels(model_name)
    return CameraCalibration.LensModel(
        pixel_size=_get_camera_pixel_size(model_name),
        camera_matrix=CameraCalibration.LensModel.CameraMatrix(
            fx=f, fy=f, cx=int(full_res.width / 2), cy=int(full_res.height / 2)
        ),
    )


def _get_camera_focal_length_in_pixels(model_name: ZividCameraModelName) -> float:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return 1785
        case ZividCameraModelName.ZIVID_2_L100:
            return 1784
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 2481
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 2479
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 4028
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 2481
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 2479
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 4028
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")


def _get_projector_focal_length_in_pixels(model_name: ZividCameraModelName) -> float:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return 1170 // 4
        case ZividCameraModelName.ZIVID_2_L100:
            return 1170 // 4
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 932 // 4
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 931 // 4
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 1407 // 4
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 932 // 4
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 931 // 4
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 1407 // 4
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")


def _adjust_camera_lens_model(
    lens_model: CameraCalibration.LensModel, sampling_mode: SamplingMode
) -> CameraCalibration.LensModel:
    match sampling_mode:
        case SamplingMode.FULL:
            scale = 1.0
        case SamplingMode.DOWNSAMPLE2X2:
            scale = 1 / 2
        case SamplingMode.DOWNSAMPLE4X4:
            scale = 1 / 4
        case _:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    return _scale_lens_model(lens_model, scale)


def _scale_lens_model(lens_model: CameraCalibration.LensModel, scale: float) -> CameraCalibration.LensModel:

    adjusted_lens_model = CameraCalibration.LensModel(
        pixel_size=lens_model.pixel_size / scale,
        camera_matrix=CameraCalibration.LensModel.CameraMatrix(
            lens_model.camera_matrix.fx * scale,
            lens_model.camera_matrix.fy * scale,
            lens_model.camera_matrix.cx * scale,
            lens_model.camera_matrix.cy * scale,
        ),
    )
    return adjusted_lens_model
