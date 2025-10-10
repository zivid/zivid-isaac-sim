import asyncio

import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.calibration import (
    BoardID,
    detect_calibration_board,
    get_ground_truth_feature_points_3d,
    spawn_calibration_board,
)
from isaacsim.zivid.camera import SamplingMode, ZividCamera, ZividCameraModelName, spawn_zivid_casing
from isaacsim.zivid.transforms import Rotation, Transform
from isaacsim.zivid.utilities.test_utils import BaseTest
from pxr import UsdLux


class Test(BaseTest):
    async def setUp(self):  # pylint: disable=invalid-name
        await create_new_stage_async()
        self.my_world = World(stage_units_in_meters=1.0)
        assert isinstance(self.my_world, World)

        await self.my_world.initialize_simulation_context_async()
        await update_stage_async()
        self.my_world.scene.add_default_ground_plane()
        self.zivid_prim_path = "/World/ZividCamera"
        zivid_pose = Transform(
            t=np.array([0.0, 0.0, 1.5]),
            rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])),
        )
        self.zivid_prim = spawn_zivid_casing(
            prim_path=self.zivid_prim_path,
            world_pose=zivid_pose,
            make_rigid_body=False,
        )

        self.zivid_camera = ZividCamera(
            prim_path=self.zivid_prim_path,
            model_name=ZividCameraModelName.ZIVID_2_PLUS_M130,
            sampling_mode=SamplingMode.FULL,
        )

        self.cb01_pose = Transform(
            t=np.array([0.5, 0.0, 0.1]),
            rot=Rotation.from_euler(np.array([0.0, 180.0, 0.0]), degrees=True),
        )

        self.cb02_pose = Transform(
            t=np.array([-0.5, 0.0, 0.1]),
            rot=Rotation.from_euler(np.array([0.0, 180.0, 0.0]), degrees=True),
        )

        self.boards = {}

        self.cb01 = spawn_calibration_board(BoardID.ZVDCB01, prim_path="/World/CB01", world_pose=self.cb01_pose)
        self.cb02 = spawn_calibration_board(BoardID.ZVDCB02, prim_path="/World/CB02", world_pose=self.cb02_pose)

        self.boards[BoardID.ZVDCB01] = self.cb01
        self.boards[BoardID.ZVDCB02] = self.cb02

        stage = omni.usd.get_context().get_stage()

        # Define the DistantLight
        light_path = "/World/Light"
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.GetIntensityAttr().Set(100.0)
        set_camera_view(
            eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
        )  # set camera view

        await update_stage_async()
        await update_stage_async()
        await self.my_world.reset_async()
        await update_stage_async()
        await update_stage_async()
        return

    # After running each test
    async def tearDown(self):  # pylint: disable=invalid-name
        self.my_world.stop()
        await omni.kit.app.get_app().next_update_async()
        self.my_world.clear_instance()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        return

    async def test_capture_calibration_board(self):
        self.zivid_camera.initialize()
        self.cb01.set_visibility(False)
        self.cb02.set_visibility(False)

        for camera_model in ZividCameraModelName:
            self.zivid_camera = ZividCamera(
                camera_model, prim_path=self.zivid_prim_path, sampling_mode=SamplingMode.FULL
            )
            self.zivid_camera.initialize()
            focus_distance = self.zivid_camera.get_focus_distance()
            for cb_id in BoardID:
                cb = self.boards[cb_id]
                cb.set_visibility(True)

                if cb_id == BoardID.ZVDCB02:
                    cob = Transform(rot=Rotation.identity(), t=np.array([40, 40, 0.0]) * 1e-3)

                else:
                    cob = Transform(rot=Rotation.identity(), t=np.array([90, 90, 0.0]) * 1e-3)

                t_oc = (
                    Transform(
                        t=np.array([0.0, 0, focus_distance]),
                        rot=Rotation.from_euler(np.array([0.0, 0.0, 0.0]), degrees=True),
                    )
                    * cob.inverse()
                )
                cb_pose = self.zivid_camera.get_camera_sensor_world_pose() * t_oc
                cb.set_world_pose(*cb_pose.to_isaac_sim())

                await self.wait_n_updates(10)

                frame = self.zivid_camera.get_frame()
                detection_result = detect_calibration_board(frame)

                self.assertTrue(
                    detection_result.valid(),
                    f"Detection failed for {camera_model.name} and {cb.prim.GetName()}"
                    f"with status {detection_result.status.name}",
                )

                ground_truth_cornes = get_ground_truth_feature_points_3d(cb_id, t_oc)
                self.assertArraysAlmostEqual(
                    detection_result.feature_points_3d,
                    ground_truth_cornes,
                    places=2,
                    quantile_thresholds={50: 1e-3},
                    msg=f"3D corners do not match for {camera_model.name} and {cb.prim.GetName()}",
                )
                projected_corners = (self.zivid_camera.project_points(ground_truth_cornes)).astype(np.float32)
                self.assertArraysAlmostEqual(
                    detection_result.feature_points_2d,
                    projected_corners,
                    places=None,
                    msg=f"2D corners do not match for {camera_model.name} and {cb.prim.GetName()}",
                    quantile_thresholds={99: 2.0, 75: 1.5, 50: 8e-1},
                )
                cb.set_visibility(False)
