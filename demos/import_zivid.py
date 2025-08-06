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

enable_zivid_extension()  # enable the Zivid extension

from isaacsim.core.api import World  # pylint: disable=C0412
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.camera import SamplingMode, ZividCamera, ZividCameraModelName, spawn_zivid_casing
from isaacsim.zivid.utilities.transforms import Rotation, Transform

my_world = World(stage_units_in_meters=1.0)
assert isinstance(my_world, World), "World instance should be created successfully"
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
zivid_pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.identity())
zivid_prim_path = "/World/ZividCamera"


spawn_zivid_casing(
    model_name=model_name,
    prim_path=zivid_prim_path,
    world_pose=zivid_pose,
    make_rigid_body=False,
)

zivid_prim_path = "/World/ZividCamera1"

spawn_zivid_casing(
    model_name=model_name,
    prim_path=zivid_prim_path,
    world_pose=zivid_pose,
    make_rigid_body=True,
)

zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)

my_world.reset()
zivid_camera.initialize()

zivid_camera.set_sampling_mode(SamplingMode.DOWNSAMPLE4X4)

i = 0
while (i < args.num_frames or args.num_frames == -1) and simulation_app.is_running():
    my_world.step(render=True)  # step the world
    i += 1

simulation_app.close()
