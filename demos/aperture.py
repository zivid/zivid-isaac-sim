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

enable_zivid_extension()

from isaacsim.core.api import World  # pylint: disable=C0412
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.calibration import BoardID, spawn_calibration_board
from isaacsim.zivid.camera import ZividCamera, ZividCameraModelName, spawn_zivid_caseing
from isaacsim.zivid.utilities.transforms import Rotation, Transform
from omni.kit.viewport.utility import get_active_viewport

my_world =  World(stage_units_in_meters=1.0)
assert isinstance(my_world, World), "World instance should be created successfully"

my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
zivid_pose = Transform(t=np.array([0.0, 0.0, 1.5]), rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])))
zivid_prim_path = "/World/ZividCamera"


spawn_zivid_caseing(
    model_name=model_name,
    prim_path=zivid_prim_path,
    world_pose=zivid_pose,
    make_rigid_body=False,
)
zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)

spawn_calibration_board(
    BoardID.ZVDCB02,
    "/World/CB02",
    Transform(t=np.array([0.0, 0.02, 1.0]), rot=Rotation.from_axis_angle(np.array([np.pi, 0, 0]))),
)

spawn_calibration_board(
    BoardID.ZVDCB01,
    "/World/CB01",
    Transform(t=np.array([0.0, 0.5, 0]), rot=Rotation.from_axis_angle(np.array([np.pi, 0, 0]))),
)


# initialize the world
my_world.reset()
zivid_camera.initialize()


i = 0


stops = list(range(-5, 5))
num_stops = len(stops)

stop_counter = 0
zivid_camera.set_lens_aperture(stops[stop_counter])  # set initial aperture
viewport = get_active_viewport()
viewport.camera_path = zivid_camera.get_camera_sensor().prim_path  # set the viewport camera to the Zivid camera


while simulation_app.is_running():
    my_world.step(render=True)  # step the world
    rgb = zivid_camera.get_data_rgb()
    if i > 0 and i % 20 == 0 and stop_counter < num_stops:
        print(f"Stop: {stops[stop_counter]}")
        stop_counter += 1
        if stop_counter >= num_stops:
            stop_counter = 0
        zivid_camera.set_lens_aperture(stops[stop_counter])  # set initial aperture

    i += 1


simulation_app.close()
