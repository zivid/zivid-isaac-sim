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
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.util.debug_draw._debug_draw import acquire_debug_draw_interface
import carb

from isaacsim.zivid.assembler.assemble import assemble_zivid
from isaacsim.zivid.cameras.models import ZividCameraModelName
from isaacsim.zivid.cameras.zivid_camera import ZividCamera
from isaacsim.zivid.utilities.transforms import Transform, Rotation, transform_points
from isaacsim.zivid.utilities.fov import draw_fov


my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view


# Create a Zivid camera


model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
zivid_pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.identity())
zivid_prim_path = "/World/ZividCamera"


zivid_prim_path = assemble_zivid(
    model_name=model_name,
    prim_path=zivid_prim_path,
    world_pose=zivid_pose,
    make_rigid_body=False,
)


zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)

# initialize the world
my_world.reset()
zivid_camera.initialize()


i = 0


while simulation_app.is_running():
    my_world.step(render=True)  # step the world
    # arm.set_joint_positions(arm.get_joints_default_state().positions)
    rgb = zivid_camera.get_data_rgb()
    if i > 0 and i % 10 == 0:
        draw_fov(zivid_camera)

    i += 1


simulation_app.close()
