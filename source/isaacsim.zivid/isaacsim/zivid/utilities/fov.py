import carb
import numpy as np
from isaacsim.util.debug_draw._debug_draw import acquire_debug_draw_interface
from isaacsim.zivid.camera.zivid_camera import ZividCamera
from isaacsim.zivid.utilities.transforms import transform_points


def _filter_nan_points(points: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(points).all(axis=-1)
    return points[finite_mask]


def draw_fov(zivid_camera: ZividCamera):
    xyz = zivid_camera.get_data_xyz()[::100, ::100]  # downsample for performance

    xyz = transform_points(
        xyz,
        zivid_camera.get_sensor_world_pose(),
    )

    # Flatten and filter
    points = xyz.reshape(-1, 3)
    valid_points = _filter_nan_points(points)

    points = []
    colors = []
    sizes = []

    for point in valid_points:
        points.append(carb.Float3(*tuple(point.tolist())))
        colors.append(carb.ColorRgba(1, 0, 0, 1))
        sizes.append(10.0)

    draw = acquire_debug_draw_interface()

    draw.clear_points()
    draw.draw_points(
        points,
        colors,
        sizes,
    )


def clear_fov():
    acquire_debug_draw_interface().clear_points()
