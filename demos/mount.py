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
import sys

import carb
import cv2
import numpy as np
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Example script with --headless flag")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num-frames", type=int, default=-1, help="Number of frames to run, -1 for infinite")
args = parser.parse_args()

simulation_app = SimulationApp(
    {"headless": args.headless, "enable_cameras": True},
)

print("Zivid extension enabled.")
from utilities import enable_zivid_extension

enable_zivid_extension()  # enable the Zivid extension

from isaacsim.core.api import World  # pylint: disable=C0412
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.zivid.assembler import assemble_zivid_casing_on_robot
from isaacsim.zivid.camera import ZividCameraModelName
from isaacsim.zivid.mounts import Mount
from isaacsim.zivid.utilities.transforms import Rotation, Transform


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
assert isinstance(my_world, World)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"


add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")  # add robot to stage
arm = SingleArticulation("/World/Arm", name="my_arm")  # create an articulation object


# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
arm.set_world_pose(position=np.array([0.0, 1.0, 0.0]) / get_stage_units())

my_world.scene.add(arm)  # add the arm to the world
# Create a Zivid camera


mount = Mount.OA121

mount_local_pose = Transform(
    rot=Rotation.from_matrix(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])), t=np.array([0.0, 0.0, 0.0])
)

model_name = ZividCameraModelName.ZIVID_2_PLUS_MR130


# initialize the world
my_world.reset()

ass = assemble_zivid_casing_on_robot(
    model_name=model_name,
    attach_prim_path="/World/Arm/panda_hand",
    robot_prim_path="/World/Arm",
    mount=mount,
    mount_local_pose=mount_local_pose,
)  # mount.zivid_mount_frame.R.as_quat().tolist(),

my_world.reset()  # reset the world to apply the changes
# arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")  # re-create articulation object

arm = SingleArticulation("/World/Arm", name="my_arm")  # create an articulation object

i = 0

attach_prim = SingleRigidPrim("/World/Arm/panda_hand", name="attach_prim")
mount_prim = ass.mount_prim


while (i < args.num_frames or args.num_frames == -1) and simulation_app.is_running():
    my_world.step(render=True)  # step the world
    attach_pose = Transform.from_isaac_sim(attach_prim.get_world_pose())
    mount_pose = Transform.from_isaac_sim(mount_prim.get_world_pose())
    zivid_pose = Transform.from_isaac_sim(ass.zivid_prim.get_world_pose())
    i += 1

simulation_app.close()
