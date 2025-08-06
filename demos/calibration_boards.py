# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# flake8: noqa: E402
# pylint: disable=C0413,C0103

import argparse
import math

import carb
import numpy as np
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Example script with --headless flag")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num-frames", type=int, default=-1, help="Number of frames to run, -1 for infinite")
args = parser.parse_args()

simulation_app = SimulationApp(
    {"headless": args.headless, "enable_cameras": True},
)

from utilities import enable_zivid_extension

enable_zivid_extension()

from isaacsim.core.api import World  # pylint: disable=C0412
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.util.debug_draw._debug_draw import acquire_debug_draw_interface
from isaacsim.zivid.calibration.calibration_board import BoardID
from isaacsim.zivid.calibration.detector import detect_calibration_board
from isaacsim.zivid.calibration.spawn import spawn_calibration_board
from isaacsim.zivid.camera.spawn import spawn_zivid_casing
from isaacsim.zivid.camera.zivid_camera import ZividCamera
from isaacsim.zivid.camera.zivid_camera_model import ZividCameraModelName
from isaacsim.zivid.utilities.transforms import Rotation, Transform, transform_points

draw = acquire_debug_draw_interface()


def visualize_points_detection(points_2d: np.ndarray, xyz: np.ndarray, pose: Transform):  # pylint: disable=W0621
    points_3d = []
    xyz = transform_points(xyz, pose)

    for point in points_2d:
        x = point[0, 0]
        y = point[0, 1]

        xl = math.floor(x)
        xu = math.ceil(x)
        yl = math.floor(y)
        yu = math.ceil(y)

        alpha = x - xl
        beta = y - yl

        p00 = xyz[yl, xl]
        p01 = xyz[yl, xu]
        p10 = xyz[yu, xl]
        p11 = xyz[yu, xu]

        p0 = p00 * (1 - alpha) + p01 * alpha
        p1 = p10 * (1 - alpha) + p11 * alpha
        p = p0 * (1 - beta) + p1 * beta
        points_3d.append(carb.Float3(p.tolist()))

    draw.clear_points()
    draw.draw_points(
        points_3d,
        [carb.ColorRgba(1, 0, 0, 1)] * len(points_3d),
        [10.0] * len(points_3d),
    )


my_world = World(stage_units_in_meters=1.0)
assert isinstance(my_world, World), "World instance should be created successfully"
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view


# Create a Zivid camera


model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
zivid_pose = Transform(t=np.array([0.0, 0.0, 1.8]), rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])))
zivid_prim_path = "/World/ZividCamera"


zivid_prim = spawn_zivid_casing(
    model_name=model_name,
    prim_path=zivid_prim_path,
    world_pose=zivid_pose,
    make_rigid_body=False,
)


board_pose = Transform(t=np.array([0.0, 0.0, 0.1]), rot=Rotation.from_axis_angle(np.array([np.pi, 0, 0.0])))

cb01 = spawn_calibration_board(
    BoardID.ZVDCB01,
    prim_path="/World/CB01",
    world_pose=board_pose,
)
cb02 = spawn_calibration_board(
    BoardID.ZVDCB02,
    prim_path="/World/CB02",
    world_pose=board_pose,
)

cb02.set_visibility(False)  # hide the board for now


zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim.prim_path)

# initialize the world
my_world.reset()
zivid_camera.initialize()

for _ in range(10):
    my_world.step(render=True)  # step the world

pose = zivid_camera.get_sensor_world_pose()
rgb = zivid_camera.get_data_rgb()
xyz = zivid_camera.get_data_xyz()


corners = detect_calibration_board(rgb)

visualize_points_detection(corners, xyz, pose)

for _ in range(50):
    my_world.step(render=True)  # step the world

draw.clear_points()

cb01.set_visibility(False)
cb02.set_visibility(True)  # show the second board

my_world.step(render=True)  # step the world

pose = zivid_camera.get_sensor_world_pose()
rgb = zivid_camera.get_data_rgb()
xyz = zivid_camera.get_data_xyz()

corners = detect_calibration_board(rgb)

visualize_points_detection(corners, xyz, pose)

i = 0
while (i < args.num_frames or args.num_frames == -1) and simulation_app.is_running():
    my_world.step(render=True)  # step the world
    i += 1

simulation_app.close()  # close the simulation app
