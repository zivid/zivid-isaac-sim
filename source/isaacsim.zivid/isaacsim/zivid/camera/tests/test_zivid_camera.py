import asyncio

import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils import prims as prim_utils
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.camera._impl.resolutions import get_camera_resolution
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.sampling import SamplingMode
from isaacsim.zivid.camera.spawn import spawn_zivid_casing
from isaacsim.zivid.camera.zivid_camera import ZividCamera
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
            t=np.array([0.0, 0.0, 1.0]),
            rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])),
        )
        self.zivid_prim = spawn_zivid_casing(
            prim_path=self.zivid_prim_path,
            world_pose=zivid_pose,
            make_rigid_body=False,
        )

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

    async def test_all_models(self):
        for model_name in ZividCameraModelName:
            ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
            self.assertTrue(self.zivid_prim.prim.GetAttribute("is_zivid_camera").Get())
            self.assertEqual(self.zivid_prim.prim.GetAttribute("zivid_camera_model").Get(), model_name.value)

    async def test_capture(self):
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        sampling_mode = SamplingMode.DOWNSAMPLE4X4
        resolution = get_camera_resolution(sampling_mode, model_name)
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path, sampling_mode=sampling_mode)
        zivid_camera.initialize()
        await self.wait_n_updates(10)  # Wait rendering
        frame = zivid_camera.get_frame()
        rgb = frame.get_data_rgb()
        self.assertIsNotNone(rgb, "RGB data should not be None")
        self.assertEqual(rgb.shape, (resolution.height, resolution.width, 3), "RGB data shape mismatch")
        xyz = frame.get_data_xyz()
        self.assertIsNotNone(xyz, "XYZ data should not be None")
        self.assertEqual(xyz.shape, (resolution.height, resolution.width, 3), "XYZ data shape mismatch")

    async def test_capture_infinite_depth(self):
        pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.from_axis_angle(np.array([0.0, -np.pi / 2, 0.0])))
        for model_name in ZividCameraModelName:
            cam = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
            cam.initialize()
            cam.set_world_pose(pose)
            await self.wait_n_updates(10)  # Wait rendering
            frame = cam.get_frame()
            xyz = frame.get_data_xyz()
            self.assertIsNotNone(xyz, "XYZ data should not be None")
            self.assertTrue(np.all(np.isnan(xyz)), f"XYZ data should be all NaN for {model_name.name}")
            rgb = frame.get_data_rgb()
            self.assertIsNotNone(rgb, "RGB data should not be None")

    async def test_change_sampling_mode(self):
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()
        for sampling_mode in SamplingMode:
            zivid_camera.set_sampling_mode(sampling_mode)
            await self.wait_n_updates(1)
            current_resolution = zivid_camera.get_camera_resolution()
            expected_resolution = get_camera_resolution(sampling_mode, model_name)
            self.assertEqual(
                current_resolution,
                expected_resolution,
                f"Simulated camera resolution should match expected resolution for {sampling_mode.name}",
            )

    async def test_change_model(self):
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()
        for model_name in ZividCameraModelName:
            zivid_camera.set_camera_model(model_name)
            self.assertEqual(self.zivid_prim.prim.GetAttribute("zivid_camera_model").Get(), model_name.value)

    async def test_set_world_pose(self):
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()
        new_pose = Transform(
            t=np.array([0.5, 0.0, 1.0]),
            rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])),
        )
        zivid_camera.set_world_pose(new_pose)
        await self.wait_n_updates(1)
        prim_pose = Transform.from_isaac_sim(self.zivid_prim.get_world_pose())
        self.assertTranformsAlmostEqual(prim_pose, new_pose, places=6)
        self.assertTranformsAlmostEqual(new_pose, zivid_camera.get_world_pose(), places=6)

    async def test_set_local_pose(self):
        prim_utils.create_prim("/World/Parent", "Xform")
        parent_prim = SingleXFormPrim(prim_path="/World/Parent")
        parent_pose = Transform(
            t=np.array([0.5, 0.5, 0.5]),
            rot=Rotation.from_euler(np.array([10.0, 10.0, 10.0]), degrees=True),
        )
        parent_prim.set_world_pose(*parent_pose.to_isaac_sim())
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=parent_prim.prim_path + "/ZividCamera")
        zivid_camera.initialize()
        local_pose = Transform(
            t=np.array([0.2, 0.3, 0.2]), rot=Rotation.from_euler(np.array([20.0, 30.0, 40.0]), degrees=True)
        )
        zivid_camera.set_local_pose(local_pose)
        await self.wait_n_updates(1)

        self.assertTranformsAlmostEqual(parent_pose * local_pose, zivid_camera.get_world_pose(), places=6)
        self.assertTranformsAlmostEqual(local_pose, zivid_camera.get_local_pose(), places=6)
