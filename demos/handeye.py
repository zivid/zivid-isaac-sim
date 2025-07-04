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

import sys

import carb
import numpy as np
import matplotlib.pyplot as plt
import cv2
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim, GeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils import prims as prim_utils
from isaacsim.core.utils.rotations import rot_matrix_to_quat, quat_to_rot_matrix
from pxr import UsdPhysics, Gf
from isaacsim.core.api.materials.preview_surface import PreviewSurface

from isaacsim.zivid.cameras.zivid_camera import ZividCamera, ZividCameraModelName
from isaacsim.zivid.assembler.assemble import assemble_zivid_on_robot
from isaacsim.zivid.assembler.mounts import Mount, Mount, attach_mount_to_robot
from isaacsim.zivid.utilities.transforms import Transform, Rotation


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


# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"


add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")  # add robot to stage
arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")  # create an articulation object


# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())

# Create a Zivid camera


mount = Mount.OA121

hand_eye = Transform(
    rot=Rotation.from_matrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])), t=np.array([0.05, -0.05, 0.02])
)

model_name = ZividCameraModelName.ZIVID_2_PLUS_M130

zivid_prim_path = assemble_zivid_on_robot(
    model_name=model_name,
    attach_prim_path="/World/Arm/panda_hand",
    robot_prim_path="/World/Arm",
    mount=mount,
    handeye_calibration=hand_eye,
)  # mount.zivid_mount_frame.R.as_quat().tolist(),


zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)
zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)

cube = FixedCuboid(
    prim_path="/World/cube",
    name="cube",
    translation=[0.8, 1.0, 0.18],
    size=0.05,
)

cube.set_visibility(False)

# initialize the world
my_world.reset()
zivid_camera.initialize()


i = 0


while simulation_app.is_running():
    my_world.step(render=True)  # step the world
    # arm.set_joint_positions(arm.get_joints_default_state().positions)
    rgb = zivid_camera.get_data_rgb()
    if i > 0 and i % 1000000 == 0:
        xyz = zivid_camera.get_data_xyz()
        depth_map = depthmap_from_xyz(xyz, min_depth=0, max_depth=2.5)
        plt.imsave("depth.png", depth_map)
        # print("saved depth map")

    i += 1


simulation_app.close()
