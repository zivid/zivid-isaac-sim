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

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False, "enable_cameras": True},
)  # start the simulation app, with GUI open

from utilities import enable_zivid_extension

enable_zivid_extension()  # enable the Zivid extension


from isaacsim.core.api import World  # pylint: disable=C0412
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.camera import ZividCamera, ZividCameraModelName, spawn_zivid_caseing
from isaacsim.zivid.utilities.fov import draw_fov
from isaacsim.zivid.utilities.transforms import Rotation, Transform

my_world = World(stage_units_in_meters=1.0)
assert isinstance(my_world, World), "World instance should be created successfully"
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view


# Create a Zivid camera


model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
zivid_pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])))
zivid_prim_path = "/World/ZividCamera"


zivid_prim = spawn_zivid_caseing(
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
models = list(ZividCameraModelName)
num_models = len(models)
model_counter = 0
zivid_camera.set_camera_model(models[model_counter])  # Set the initial model

while simulation_app.is_running():
    my_world.step(render=True)  # step the world
    # arm.set_joint_positions(arm.get_joints_default_state().positions)
    rgb = zivid_camera.get_data_rgb()
    if i > 0 and i % 20 == 0 and model_counter < num_models:
        print(f"Switched to model: {models[model_counter].name}")

        draw_fov(zivid_camera)
        model_counter += 1
        if model_counter >= num_models:
            model_counter = 0
        zivid_camera.set_camera_model(models[model_counter])

    i += 1


simulation_app.close()
