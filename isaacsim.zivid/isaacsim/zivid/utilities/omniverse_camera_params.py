from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

import omni.kit.commands
from pxr import Sdf
from isaacsim.sensors.camera import Camera


@dataclass
class OmniverseCameraParams:
    clipping_range: tuple[float, float] = (0.01, 1e6)
    focal_length: float = 24.0
    focus_distance: float = 1.0
    f_stop: float = 0.0
    horizontal_aperture: float = 20.955
    vertical_aperture: float | None = None
    horizontal_aperture_offset: float = 0.0
    vertical_aperture_offset: float = 0.0

    @staticmethod
    def from_intrinsic_matrix(
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        clipping_range: tuple[float, float] = (0.01, 1e6),
        pixel_size: float = 3 * 1e-6,
        focus_distance: float = 400.0,
        f_stop: float = 0.0,
    ) -> OmniverseCameraParams:
        f_x = intrinsic_matrix[0]
        c_x = intrinsic_matrix[2]
        f_y = intrinsic_matrix[4]
        c_y = intrinsic_matrix[5]

        focal_length = f_x * pixel_size
        horizontal_aperture = width * pixel_size
        vertical_aperture = height * pixel_size
        horizontal_aperture_offset = (c_x - width / 2) / f_x
        vertical_aperture_offset = (c_y - height / 2) / f_y

        return OmniverseCameraParams(
            clipping_range=clipping_range,
            focal_length=focal_length,
            focus_distance=focus_distance,
            f_stop=f_stop,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=horizontal_aperture_offset,
            vertical_aperture_offset=vertical_aperture_offset,
        )


def apply_model_params_to_camera(
    camera: Camera,
    model_params: OmniverseCameraParams,
) -> None:
    camera.set_clipping_range(0.0000000000001, 10)
    camera.set_focal_length(model_params.focal_length * 10)
    camera.set_focus_distance(1.0)
    camera.set_lens_aperture(model_params.f_stop)
    camera.set_horizontal_aperture(model_params.horizontal_aperture * 10)
    camera.set_projection_type("pinhole")


def lock_camera(camera: Camera, lock: bool = True) -> None:
    omni.kit.commands.execute(
        "ChangePropertyCommand",
        prop_path=Sdf.Path(f"{camera.prim_path}.omni:kit:cameraLock"),
        value=lock,
        prev=None,
        type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
    )


def hide_camera_icon(camera: Camera, lock: bool = True) -> None:
    omni.kit.commands.execute(
        "ToggleVisibilitySelectedPrims",
        selected_paths=[camera.prim_path],
        stage=omni.usd.get_context().get_stage(),
        visible=False,
    )
