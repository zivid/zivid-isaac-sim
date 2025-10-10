from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from isaacsim.zivid.resolution import Resolution
from numpy.typing import NDArray


@dataclass(frozen=True)
class LensModel:
    camera_matrix: CameraMatrix
    distortion_coefficients: DistortionCoefficients
    pixel_size: float
    focus_distance: float
    resolution: Resolution


@dataclass
class CameraMatrix:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_array(cls, array: NDArray):
        assert array.shape == (3, 3), f"Expected array of shape (3, 3), got {array.shape}"
        return cls(
            fx=array[0, 0],
            fy=array[1, 1],
            cx=array[0, 2],
            cy=array[1, 2],
        )

    def to_array(self) -> NDArray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class DistortionCoefficients:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0

    @classmethod
    def from_array(cls, array: NDArray):
        assert array.size == 12, f"Expected array of size 12, got {array.size}"
        return cls(*array.flatten().tolist())

    def to_array(self) -> NDArray:
        return np.array(
            [
                self.k1,
                self.k2,
                self.p1,
                self.p2,
                self.k3,
                self.k4,
                self.k5,
                self.k6,
                self.s1,
                self.s2,
                self.s3,
                self.s4,
            ]
        )


def scale_lens_model(lens_model: LensModel, scale: float) -> LensModel:

    return replace(
        lens_model,
        pixel_size=lens_model.pixel_size / scale,
        camera_matrix=_scale_camera_matrix(lens_model.camera_matrix, scale),
        resolution=_scale_resolution(lens_model.resolution, scale),
    )


def _scale_camera_matrix(camera_matrix: CameraMatrix, scale: float) -> CameraMatrix:
    return CameraMatrix(
        fx=camera_matrix.fx * scale,
        fy=camera_matrix.fy * scale,
        cx=camera_matrix.cx * scale,
        cy=camera_matrix.cy * scale,
    )


def _scale_resolution(resolution: Resolution, scale: float) -> Resolution:
    return Resolution(
        width=int(resolution.width * scale),
        height=int(resolution.height * scale),
    )
