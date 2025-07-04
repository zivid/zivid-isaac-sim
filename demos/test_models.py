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

# initialize the world
my_world.reset()
zivid_camera.initialize()


i = 0

models = list(ZividCameraModelName)
num_models = len(models)
model_counter = 0

# Generate colors from a colormap
cmap = plt.get_cmap("tab20")  # tab20 has 20 distinct colors

# Assign a color to each model by mapping indices to colors
color_map = [carb.ColorRgba(cmap(i / len(models))) for i in models]

zivid_camera.set_camera_model(models[model_counter])  # Set the initial model


while simulation_app.is_running():
    my_world.step(render=True)  # step the world
    # arm.set_joint_positions(arm.get_joints_default_state().positions)
    rgb = zivid_camera.get_data_rgb()
    if i > 0 and i % 20 == 0 and model_counter < num_models:
        print(f"Switched to model: {models[model_counter].name}")

        xyz = zivid_camera.get_data_xyz()[::100, ::100]  # downsample for performance

        xyz = transform_points(
            xyz,
            zivid_camera.get_sensor_world_pose(),
        )

        # Flatten and filter
        points = xyz.reshape(-1, 3)
        valid_points = filter_nan_points(points)

        points = []
        colors = []
        sizes = []

        for point in valid_points:
            points.append(carb.Float3(*tuple(point.tolist())))
            colors.append(color_map[model_counter])
            sizes.append(10.0)

        print(f"Valid points: {len(valid_points)}")
        # Draw as spheres or points
        # draw.clear_points()
        draw.draw_points(
            points,
            colors,
            sizes,
        )
        model_counter += 1
        if model_counter >= num_models:
            model_counter = 0
        zivid_camera.set_camera_model(models[model_counter])

    i += 1


simulation_app.close()
