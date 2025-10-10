from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.sampling import SamplingMode
from isaacsim.zivid.transforms import Rotation, Transform

from .lensmodel import CameraMatrix, DistortionCoefficients, LensModel, scale_lens_model
from .resolutions import get_camera_resolution, get_projector_resolution


@dataclass
class CameraCalibration:

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


def _get_camera_lens_model(model_name: ZividCameraModelName, sampling_mode: SamplingMode) -> LensModel:
    full_res = get_camera_resolution(SamplingMode.FULL, model_name)
    current_res = get_camera_resolution(sampling_mode, model_name)
    scale_w = current_res.width / full_res.width
    scale_h = current_res.height / full_res.height
    assert scale_w == scale_h, "Down sampling should conserve aspect ratio"
    scale = scale_w
    return scale_lens_model(_get_full_res_camera_lens_model(model_name), scale)


def _get_projector_lens_model(model_name: ZividCameraModelName) -> LensModel:
    res = get_projector_resolution(model_name)
    f = _get_projector_focal_length_in_pixels(model_name)
    camera_mat = CameraMatrix(
        fx=f,
        fy=f,
        cx=int(res.width / 2),
        cy=int(res.height / 2),
    )
    return scale_lens_model(
        LensModel(
            pixel_size=_get_projector_pixel_size(model_name),
            camera_matrix=camera_mat,
            distortion_coefficients=_get_projector_distortion_coefficients(model_name),
            resolution=res,
            focus_distance=_focus_distance_from_model_name(model_name),
        ),
        0.5,  # Down sample for increased render performance
    )


def _get_projector_pixel_size(model_name: ZividCameraModelName) -> float:
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_L100:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 5.4 * 1e-6
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 5.4 * 1e-6
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


def _get_full_res_camera_lens_model(model_name: ZividCameraModelName) -> LensModel:
    full_res = get_camera_resolution(SamplingMode.FULL, model_name)
    f = _get_camera_focal_length_in_pixels(model_name)
    return LensModel(
        pixel_size=_get_camera_pixel_size(model_name),
        camera_matrix=CameraMatrix(fx=f, fy=f, cx=int(full_res.width / 2), cy=int(full_res.height / 2)),
        distortion_coefficients=_get_camera_distortion_coefficients(model_name),
        resolution=full_res,
        focus_distance=_focus_distance_from_model_name(model_name),
    )


def _get_camera_distortion_coefficients(
    model_name: ZividCameraModelName,
) -> DistortionCoefficients:

    params = np.zeros(12)
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            params[:5] = [-0.094, 0.136, 0, 0, -0.071]
        case ZividCameraModelName.ZIVID_2_L100:
            params[:5] = [-0.089, 0.125, 0, 0, -0.048]
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            params[:5] = [0.044, -0.005, 0, 0, -0.276]
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            params[:5] = [0.056, -0.057, 0, 0, -0.186]
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            params[:5] = [-0.09, 0.098, 0, 0, -0.128]
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            params[:5] = [0.044, -0.005, 0, 0, -0.276]
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            params[:5] = [0.056, -0.057, 0, 0, -0.186]
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            params[:5] = [-0.09, 0.098, 0, 0, -0.128]
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")
    return DistortionCoefficients.from_array(params)


def _get_projector_distortion_coefficients(
    model_name: ZividCameraModelName,
) -> DistortionCoefficients:

    params = np.zeros(12)
    match model_name:
        case ZividCameraModelName.ZIVID_2_M70:
            params[:5] = [0.003, -0.007, 0, 0, -0.085]
        case ZividCameraModelName.ZIVID_2_L100:
            params[:5] = [0.006, -0.018, 0, 0, -0.07]
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            params[:5] = [0.012, -0.118, 0, 0, 0.063]
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            params[:5] = [0.016, -0.126, 0, 0, 0.071]
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            params[:5] = [0.017, -0.002, 0, 0, -0.088]
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            params[:5] = [0.012, -0.118, 0, 0, 0.063]
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            params[:5] = [0.016, -0.126, 0, 0, 0.071]
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            params[:5] = [0.017, -0.002, 0, 0, -0.088]
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")
    return DistortionCoefficients.from_array(params)


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
            return 1170
        case ZividCameraModelName.ZIVID_2_L100:
            return 1170
        case ZividCameraModelName.ZIVID_2_PLUS_M60:
            return 932
        case ZividCameraModelName.ZIVID_2_PLUS_L110:
            return 931
        case ZividCameraModelName.ZIVID_2_PLUS_M130:
            return 1407
        case ZividCameraModelName.ZIVID_2_PLUS_MR60:
            return 932
        case ZividCameraModelName.ZIVID_2_PLUS_LR110:
            return 931
        case ZividCameraModelName.ZIVID_2_PLUS_MR130:
            return 1407
        case _:
            raise ValueError(f"Unknown Zivid camera model name: {model_name}")


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
