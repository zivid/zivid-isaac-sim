import cv2
import numpy as np
from isaacsim.zivid.camera._impl.lensmodel import CameraMatrix, DistortionCoefficients, LensModel
from numpy.typing import NDArray


def project_to_camera(
    xyz: NDArray, camera_matrix: CameraMatrix, distortion_coeffs: DistortionCoefficients | None = None
) -> NDArray:
    h, w = xyz.shape[:2]
    im = np.zeros((h, w, 2))
    im[..., 0] = xyz[..., 0] * camera_matrix.fx / xyz[..., 2] + camera_matrix.cx
    im[..., 1] = xyz[..., 1] * camera_matrix.fy / xyz[..., 2] + camera_matrix.cy

    points_d = xyz[..., :2] / xyz[..., 2:3]
    if distortion_coeffs is not None:
        points_d = _apply_distortion(points_d, distortion_coeffs)
    im[..., 0] = points_d[..., 0] * camera_matrix.fx + camera_matrix.cx
    im[..., 1] = points_d[..., 1] * camera_matrix.fy + camera_matrix.cy

    return im


def unproject_from_camera(depth: NDArray, lens_model: LensModel) -> NDArray:
    if depth.shape != (lens_model.resolution.height, lens_model.resolution.width):
        raise ValueError(
            f"Depth shape {depth.shape} does not match lens model resolution "
            f"({lens_model.resolution.height}, {lens_model.resolution.width})"
        )
    uv_undistorted = get_undistorted_normalized_pixels(lens_model)

    x = uv_undistorted[..., 0] * depth
    y = uv_undistorted[..., 1] * depth
    z = depth
    return np.stack((x, y, z), axis=-1)


def get_undistorted_normalized_pixels(lens_model: LensModel) -> NDArray:
    h, w = lens_model.resolution.height, lens_model.resolution.width
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    uv = np.stack((x, y), axis=-1, dtype=np.float32)
    uv_undistorted = cv2.undistortPoints(
        uv.reshape(-1, 2), lens_model.camera_matrix.to_array(), lens_model.distortion_coefficients.to_array()
    ).reshape(h, w, 2)

    return uv_undistorted


def _apply_distortion(points: NDArray, distortion_coeffs: DistortionCoefficients) -> NDArray:
    r2 = points[..., 0] ** 2 + points[..., 1] ** 2
    r4 = r2 * r2
    r6 = r4 * r2
    x2 = points[..., 0] ** 2
    y2 = points[..., 1] ** 2
    xy = points[..., 0] * points[..., 1]

    theta_radial = (1 + distortion_coeffs.k1 * r2 + distortion_coeffs.k2 * r4 + distortion_coeffs.k3 * r6) / (
        1 + distortion_coeffs.k4 * r2 + distortion_coeffs.k5 * r4 + distortion_coeffs.k6 * r6
    )

    distorted = np.zeros_like(points)

    distorted[..., 0] = points[..., 0] * theta_radial
    distorted[..., 1] = points[..., 1] * theta_radial

    distorted[..., 0] += 2 * distortion_coeffs.p1 * xy + distortion_coeffs.p2 * (r2 + 2 * x2)
    distorted[..., 1] += distortion_coeffs.p1 * (r2 + 2 * y2) + 2 * distortion_coeffs.p2 * xy

    distorted[..., 0] += distortion_coeffs.s1 * r2 + distortion_coeffs.s2 * r4
    distorted[..., 1] += distortion_coeffs.s3 * r2 + distortion_coeffs.s4 * r4

    return distorted
