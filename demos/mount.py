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

import numpy as np
from isaacsim import SimulationApp  # pyright: ignore

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

import carb
from isaacsim.core.api import World  # pylint: disable=C0412
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.zivid.assembler import assemble_zivid_casing_on_robot
from isaacsim.zivid.camera import SamplingMode, ZividCamera, ZividCameraModelName
from isaacsim.zivid.mounts import Mount
from isaacsim.zivid.transforms import Rotation, Transform
from isaacsim.zivid.utilities.assets import AssetPaths

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

add_reference_to_stage(usd_path=str(AssetPaths.franka), prim_path="/World/Arm")  # add robot to stage

mount = Mount.OA121
mount_local_pose = Transform(
    rot=Rotation.from_matrix(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])), t=np.array([0.0, 0.0, 0.0])
)
ass = assemble_zivid_casing_on_robot(
    attach_prim_path="/World/Arm/panda_hand",
    robot_prim_path="/World/Arm",
    mount=mount,
    mount_local_pose=mount_local_pose,
)

my_world.reset()  # reset the world to apply the changes
model_name = ZividCameraModelName.ZIVID_2_PLUS_MR130
arm = SingleArticulation("/World/Arm", name="my_arm")  # create an articulation object
zivid_camera = ZividCamera(model_name, ass.zivid_prim.prim_path, SamplingMode.DOWNSAMPLE4X4)
zivid_camera.initialize()

i = 0
while (i < args.num_frames or args.num_frames == -1) and simulation_app.is_running():
    my_world.step(render=True)  # step the world
    i += 1

simulation_app.close()
