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
import carb
from isaacsim.util.debug_draw._debug_draw import acquire_debug_draw_interface

from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view

from omni.kit.viewport.utility import get_active_viewport


from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path

from isaacsim.zivid.assembler.assemble import assemble_zivid
from isaacsim.zivid.cameras.models import ZividCameraModelName
from isaacsim.zivid.cameras.zivid_camera import ZividCamera
from isaacsim.zivid.utilities.transforms import Transform, Rotation, transform_points
from isaacsim.zivid.utilities.assets import get_assets_path


def filter_nan_points(points: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(points).all(axis=-1)
    return points[finite_mask]


draw = acquire_debug_draw_interface()


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


zivid_camera = ZividCamera(model_name=model_name, prim_path=zivid_prim_path)

add_reference_to_stage(
    usd_path=str(get_assets_path() / "CB02.usd"),
    prim_path="/World/CB02",
)  # add robot to stage

board = SingleXFormPrim(prim_path="/World/CB02", name="board")
board.set_world_pose([0, 0, 0], [0.5 for _ in range(4)])  # set board pose

# initialize the world
my_world.reset()
zivid_camera.initialize()


i = 0


stops = list(range(-5, 5))
num_stops = len(stops)
cmap = plt.get_cmap("tab20")

stop_counter = 0

# Generate colors from a colormap
cmap = plt.get_cmap("tab20")  # tab20 has 20 distinct colors

# Assign a color to each model by mapping indices to colors
color_map = [carb.ColorRgba(cmap(i / len(stops))) for i in stops]

zivid_camera.set_lens_aperture(stops[stop_counter])  # set initial aperture

viewport = get_active_viewport()
viewport.camera_path = zivid_camera._camera_sensor.prim_path  # set the viewport camera to the Zivid camera


while simulation_app.is_running():
    my_world.step(render=True)  # step the world
    # arm.set_joint_positions(arm.get_joints_default_state().positions)
    rgb = zivid_camera.get_data_rgb()
    if i > 0 and i % 20 == 0 and stop_counter < num_stops:

        print(f"Stop: {stops[stop_counter]}")

        stop_counter += 1
        if stop_counter >= num_stops:
            stop_counter = 0

        zivid_camera.set_lens_aperture(stops[stop_counter])  # set initial aperture

    i += 1


simulation_app.close()
