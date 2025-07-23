import asyncio
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.resolution import SamplingMode, get_camera_resolution
from isaacsim.zivid.camera.spawn import spawn_zivid_caseing
from isaacsim.zivid.camera.zivid_camera import ZividCamera
from isaacsim.zivid.utilities.transforms import Rotation, Transform
from pxr import UsdLux


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the
# root of module will make it auto-discoverable by omni.kit.test
class Test(omni.kit.test.AsyncTestCase):
    # Before running each test
    # Before running each test
    async def setUp(self):  # pylint: disable=invalid-name
        await create_new_stage_async()
        self.my_world = World(stage_units_in_meters=1.0)
        self.test_output_dir = Path(__file__).parents[6] / "test_outputs" / "camera" / "test_zivid_camera"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        await self.my_world.initialize_simulation_context_async()
        await update_stage_async()
        self.my_world.scene.add_default_ground_plane()
        self.zivid_prim_path = "/World/ZividCamera"
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_pose = Transform(
            t=np.array([0.0, 0.0, 1.0]),
            rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])),
        )
        self.zivid_prim = spawn_zivid_caseing(
            model_name=model_name,
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
        rgb = zivid_camera.get_data_rgb()
        self.assertIsNotNone(rgb, "RGB data should not be None")
        self.assertEqual(rgb.shape, (resolution.height, resolution.width, 3), "RGB data shape mismatch")
        xyz = zivid_camera.get_data_xyz()
        self.assertIsNotNone(xyz, "XYZ data should not be None")
        self.assertEqual(xyz.shape, (resolution.height, resolution.width, 3), "XYZ data shape mismatch")

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

    async def test_get_resolution(self):
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()
        for sampling_mode in SamplingMode:
            zivid_camera.set_sampling_mode(sampling_mode)
            await self.wait_n_updates(1)
            current_resolution = zivid_camera.get_camera_resolution()
            simulated_resolution = zivid_camera.get_camera_sensor().get_resolution()
            self.assertEqual(
                current_resolution.width,
                simulated_resolution[0],
                "Simulated camera width should match Zivid camera width",
            )
            self.assertEqual(
                current_resolution.height,
                simulated_resolution[1],
                "Simulated camera height should match Zivid camera height",
            )

    async def test_consistent_camera_matrix(self):
        model_name = ZividCameraModelName.ZIVID_2_L100
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()
        for model_name in ZividCameraModelName:
            zivid_camera.set_camera_model(model_name)
            for sampling_mode in SamplingMode:
                zivid_camera.set_sampling_mode(sampling_mode)
                calib = zivid_camera.get_calibration()

                self.assertTrue(
                    np.allclose(
                        calib.camera_model.camera_matrix.as_array(),
                        zivid_camera.get_camera_sensor().get_intrinsics_matrix(),
                    ),
                    "Simulated camera matrix should match Zivid camera matrix",
                )
                self.assertTrue(
                    np.allclose(
                        calib.projector_model.camera_matrix.as_array(),
                        zivid_camera.get_projector().get_intrinsics_matrix(),
                    ),
                    "Simulated projector matrix should match Zivid projector matrix",
                )

    async def test_consistent_extrinsics(self):
        for model_name in ZividCameraModelName:
            zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
            zivid_camera.initialize()
            calib = zivid_camera.get_calibration()
            sensor_pose = zivid_camera.get_sensor_world_pose()
            projector_pose = zivid_camera.get_projector_world_pose()
            simulated_extrinsics = sensor_pose.inverse() * projector_pose

            self.assertTrue(
                np.allclose(simulated_extrinsics.t, calib.extriniscs.t),
                "Simulated extrinsics translation should match Zivid extrinsics translation",
            )
            self.assertTrue(
                np.allclose(simulated_extrinsics.rot.as_quat(), calib.extriniscs.rot.as_quat(), atol=1e-7),
                "Simulated extrinsics rotation should match Zivid extrinsics rotation",
            )

    async def test_consistent_optical_frame(self):
        for model_name in ZividCameraModelName:
            zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
            zivid_camera.initialize()
            sensor_pose = zivid_camera.get_sensor_world_pose()
            case_pose = zivid_camera.get_case_world_pose()
            simualted_optical_frame = case_pose.inverse() * sensor_pose
            desired_optical_frame = zivid_camera.get_camera_model().optical_frame
            self.assertTrue(
                np.allclose(simualted_optical_frame.t, desired_optical_frame.t),
                "Simulated optical frame translation should match Zivid optical frame translation",
            )
            self.assertTrue(
                np.allclose(simualted_optical_frame.rot.as_quat(), desired_optical_frame.rot.as_quat()),
                "Simulated optical frame rotation should match Zivid optical frame rotation",
            )

    async def test_set_aperture(self):

        self.zivid_prim.set_world_pose(position=[0.0, 0.0, 0.30])
        cracker_box_usd_path = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"

        add_reference_to_stage(
            usd_path=cracker_box_usd_path,
            prim_path="/World/CrackerBox",
        )

        await update_stage_async()

        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()
        await self.wait_n_updates(8)

        focuses = []
        for apertuere in range(-4, 5):
            zivid_camera.set_lens_aperture(apertuere)
            await self.wait_n_updates(1)
            rgb = zivid_camera.get_data_rgb()
            save_path = self.test_output_dir / f"apertue_{apertuere}.png"
            plt.imsave(save_path, rgb)
            focuses.append(_estimate_focus(rgb))

        for i in range(len(focuses) - 1):
            self.assertGreater(focuses[i], focuses[i + 1] - 2e-1, "Focus should decrease with increasing aperture")

    async def test_focus_distance(self):

        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_camera = ZividCamera(model_name=model_name, prim_path=self.zivid_prim_path)
        zivid_camera.initialize()

        await self.wait_n_updates(1)

        for model_names in ZividCameraModelName:
            zivid_camera.set_camera_model(model_names)
            focus_distance = zivid_camera.get_focus_distance()
            self.assertTrue(
                abs(focus_distance - zivid_camera.get_camera_sensor().get_focus_distance()) < 1e-6,
                "Focus distance written to the simulator does not match expected value",
            )


def _estimate_focus(image: np.ndarray) -> float:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance
