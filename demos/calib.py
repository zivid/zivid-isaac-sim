# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False, "enable_cameras": True},
)  # start the simulation app, with GUI open


import numpy as np
import matplotlib.pyplot as plt
import cv2

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.zivid.assembler.assemble import assemble_zivid
from isaacsim.zivid.assembler.calibration_board import detect_calibration_board, add_calibration_board_to_stage, BoardID

from isaacsim.zivid.cameras.models import ZividCameraModelName
from isaacsim.zivid.cameras.zivid_camera import ZividCamera
from isaacsim.zivid.cameras.resolution import Resolution, SamplingMode
from isaacsim.zivid.utilities.transforms import Transform, Rotation, transform_points

from isaacsim.util.debug_draw._debug_draw import acquire_debug_draw_interface
import carb

draw = acquire_debug_draw_interface()


def depthmap_from_xyz(xyz, min_depth=0.5, max_depth=4.0):
    """Converts structured point cloud to a color depth map with NaN handling."""
    depth = xyz[:, :, 2]  # Z-channel

    # Create a mask for valid depth values
    valid_mask = np.isfinite(depth)
    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_normalized = np.zeros_like(depth_clipped, dtype=np.float32)
    depth_normalized[valid_mask] = (depth_clipped[valid_mask] - min_depth) / (max_depth - min_depth)

    depth_8bit = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

    # Color invalid pixels (NaNs) as black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored


my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view


# Create a Zivid camera


model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
zivid_pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])))
zivid_prim_path = "/World/ZividCamera"


zivid_prim_path = assemble_zivid(
    model_name=model_name,
    prim_path=zivid_prim_path,
    world_pose=zivid_pose,
    make_rigid_body=False,
)


board = SingleXFormPrim(prim_path="/World/CB02", name="board")
board.set_world_pose([0.0, 0, 0.1], [0, 1.0, 0.0, 0.0])

board_pose = Transform(t=np.array([0.0, 0.0, 0.1]), rot=Rotation.from_axis_angle(np.array([np.pi / 2, 0, 0.0])))

add_calibration_board_to_stage(
    BoardID.ZVDCB02,
    prim_path="/World/CB02",
    world_pose=board_pose,
)


zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)

# initialize the world
my_world.reset()
zivid_camera.initialize()

# zivid_camera.set_sampling_mode(SamplingMode.DOWNSAMPLE4X4)


my_world.step(render=True)  # step the world

my_world.step(render=True)  # step the world
rgb = zivid_camera.get_data_rgb()


plt.imsave("rgb.png", rgb)

corners = detect_calibration_board(rgb)

if corners is not None:
    print(f"Detected corners: {corners}")

while simulation_app.is_running():
    my_world.step(render=True)  # step the world
