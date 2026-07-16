# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# How to run from inside your Isaac Sim installation, with the python.sh script it contains:
# ./python.sh <path-to-zivid-isaac-sim-repo>/demos/capture_with_noise.py [--headless]
#             --output <path> --model <MR60|MR130>

# flake8: noqa: E402
# pylint: disable=C0413,C0415,C0103
import argparse
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp  # pyright: ignore
from noise_profiles import ModelName, apply_point_precision_noise

_LUX_VALUES = [0, 150, 300, 450, 600, 750]
_ZIVID_PRIM_PATH = "/World/ZividCamera"


def _parse_args():
    parser = argparse.ArgumentParser(description="Demonstrate point precision noise on Isaac Sim captures")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--output", type=str, required=True, help="Output directory for PLY files")
    parser.add_argument(
        "--model",
        choices=["MR60", "MR130"],
        required=True,
        help="Camera model to simulate",
    )
    return parser.parse_args()


def _save_point_cloud(frame, directory, label):
    frame.save_ply(directory / f"point_cloud_{label}.ply")


def main():
    args = _parse_args()
    simulation_app = SimulationApp({"headless": args.headless, "enable_cameras": True})
    try:
        from isaacsim.core.api import World  # pylint: disable=C0412
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, UsdGeom
        from utilities import enable_zivid_extension

        def _set_prim_transform(stage, prim_path, translate, rotate_xyz=None, scale=None):
            precision = UsdGeom.XformOp.PrecisionDouble
            xformable = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp(precision=precision).Set(Gf.Vec3d(*translate))
            if rotate_xyz is not None:
                xformable.AddRotateXYZOp(precision=precision).Set(Gf.Vec3d(*rotate_xyz))
            if scale is not None:
                xformable.AddScaleOp(precision=precision).Set(Gf.Vec3d(*scale))

        enable_zivid_extension()
        from isaacsim.zivid.camera import ZividCamera, ZividCameraModelName, spawn_zivid_casing
        from isaacsim.zivid.frame import Frame
        from isaacsim.zivid.transforms import Rotation, Transform

        camera_config = {
            "MR60": (
                ZividCameraModelName.ZIVID_2_PLUS_MR60,
                Transform(t=np.array([0.0, 0.0, 0.8]), rot=Rotation.from_euler(np.array([0.0, np.pi / 2, 0.0]))),
            ),
            "MR130": (
                ZividCameraModelName.ZIVID_2_PLUS_MR130,
                Transform(t=np.array([0.0, 0.0, 1.2]), rot=Rotation.from_euler(np.array([0.0, np.pi / 2, 0.0]))),
            ),
        }
        noise_model_name_map = {
            "MR60": ModelName.MR60,
            "MR130": ModelName.MR130,
        }

        my_world = World(stage_units_in_meters=1.0)
        my_world.scene.add_default_ground_plane()
        set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0], camera_prim_path="/OmniverseKit_Persp")

        assets_root = get_assets_root_path()
        add_reference_to_stage(usd_path=assets_root + "/Isaac/Props/KLT_Bin/small_KLT.usd", prim_path="/World/KLT")
        add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Props/Food/mac_n_cheese_centered.usd",
            prim_path="/World/MacNCheese1",
        )
        add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Props/Food/mac_n_cheese_centered.usd",
            prim_path="/World/MacNCheese2",
        )
        add_reference_to_stage(usd_path=assets_root + "/Isaac/Props/Shapes/cylinder.usd", prim_path="/World/Cylinder")

        world_stage = my_world.stage
        _set_prim_transform(world_stage, "/World/KLT", translate=(0.0, 0.0, 0.1))
        UsdGeom.Xformable(world_stage.GetPrimAtPath("/World/KLT")).AddScaleOp().Set(Gf.Vec3d(1.5, 1.5, 1.5))
        _set_prim_transform(
            world_stage, "/World/MacNCheese1", translate=(0.05, -0.05, 0.05), rotate_xyz=(0.0, 0.0, 15.0)
        )
        _set_prim_transform(
            world_stage, "/World/MacNCheese2", translate=(-0.08, 0.02, 0.05), rotate_xyz=(0.0, 0.0, 0.0)
        )
        _set_prim_transform(world_stage, "/World/Cylinder", translate=(0.1, 0.13, 0.05), scale=(0.05, 0.05, 0.05))

        model_name, zivid_pose = camera_config[args.model]

        spawn_zivid_casing(
            prim_path=_ZIVID_PRIM_PATH,
            world_pose=zivid_pose,
            make_rigid_body=False,
        )

        zivid_camera = ZividCamera(model_name=model_name, prim_path=_ZIVID_PRIM_PATH)

        my_world.reset()
        zivid_camera.initialize()

        # Advance a few frames so scene updates and rendering stabilize before capture.
        for _ in range(30):
            my_world.step(render=True)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        captured_frame = zivid_camera.get_frame()
        clean_xyz = captured_frame.get_data_xyz(clone=True)
        rgb = captured_frame.get_data_rgb()

        _save_point_cloud(captured_frame, output_dir, "point__clean")
        print(f"Saved clean point cloud to {output_dir / 'point_cloud_point__clean.ply'}")

        valid_mask = np.isfinite(clean_xyz[:, :, 2])

        noise_model_name = noise_model_name_map[args.model]
        for lux in _LUX_VALUES:
            print(f"Applying point precision noise for lux={lux:.0f}")
            noisy_xyz = apply_point_precision_noise(clean_xyz, noise_model_name, lux)
            z_noise = clean_xyz[:, :, 2] - noisy_xyz[:, :, 2]
            noise_std = np.nanstd(z_noise[valid_mask])
            noisy_frame = Frame(rgb, noisy_xyz)
            file_label = f"point__precision_lux_{lux:.0f}"
            _save_point_cloud(noisy_frame, output_dir, file_label)
            print(f"  lux={lux:.0f}: noise (Z) std ~{noise_std * 1e6:.1f} um")
    finally:
        simulation_app.close()


if __name__ == "__main__":  # NOLINT: standalone demo entry point
    main()
