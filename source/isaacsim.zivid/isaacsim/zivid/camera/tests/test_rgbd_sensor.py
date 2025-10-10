import asyncio
from dataclasses import replace
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from isaacsim.zivid.camera._impl.lensmodel import CameraMatrix, DistortionCoefficients, LensModel
from isaacsim.zivid.camera._impl.projection import get_undistorted_normalized_pixels
from isaacsim.zivid.camera._impl.rgbd_sensor import RGBDSensor
from isaacsim.zivid.resolution import Resolution
from isaacsim.zivid.transforms import Rotation, Transform
from isaacsim.zivid.utilities.test_utils import BaseTest
from pxr import UsdLux


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the
# root of module will make it auto-discoverable by omni.kit.test
class Test(BaseTest):
    # Before running each test
    # Before running each test
    async def setUp(self):  # pylint: disable=invalid-name
        await create_new_stage_async()
        self.my_world: World = World(stage_units_in_meters=1.0)
        self.test_output_dir = Path(__file__).parents[6] / "test_outputs" / "camera" / "test_rgbd_sensor"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        await self.my_world.initialize_simulation_context_async()
        await update_stage_async()
        self.my_world.scene.add_default_ground_plane()
        self.rgbd_prim_path = "/World/RGBDSensor"

        stage = omni.usd.get_context().get_stage()

        # Define the DistantLight
        light_path = "/World/Light"
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.GetIntensityAttr().Set(100.0)
        set_camera_view(
            eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
        )  # set camera view

        self.init_pose = Transform(
            t=np.array([0.0, 0.0, 1.0]), rot=Rotation.from_axis_angle(np.array([0.0, np.pi, 0.0]))
        )

        self.rgbd_sensor = RGBDSensor(
            prim_path=self.rgbd_prim_path, lens_model=_lens_model_1(), pose=self.init_pose, is_local_pose=False
        )

        self.rgbd_sensor.initialize()

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

    async def test_capture(self):

        self.rgbd_sensor.initialize()
        await self.wait_n_updates(10)  # Wait rendering

        self._validate_capture(self.rgbd_sensor.get_resolution())

    async def test_change_lens_model(self):
        self.rgbd_sensor.initialize()
        new_lens_model = _lens_model_2()
        self.rgbd_sensor.set_lens_model(new_lens_model)
        self.assertEqual(self.rgbd_sensor.get_resolution(), new_lens_model.resolution)
        self.assertEqual(self.rgbd_sensor.get_lens_model(), new_lens_model)
        await self.wait_n_updates(3)
        self._validate_capture(new_lens_model.resolution)
        new_lens_model = _lens_model_1()
        self.rgbd_sensor.set_lens_model(new_lens_model)
        self.assertEqual(self.rgbd_sensor.get_resolution(), new_lens_model.resolution)
        self.assertEqual(self.rgbd_sensor.get_lens_model(), new_lens_model)
        await self.wait_n_updates(1)
        self._validate_capture(new_lens_model.resolution)

    async def test_change_pose(self):
        delta_pose = Transform(t=np.array([0.1, 0.1, 0.1]), rot=Rotation.from_axis_angle(np.ones(3) * np.pi / 6))
        new_pose = self.init_pose * delta_pose
        self.rgbd_sensor.set_world_pose(new_pose)
        await self.wait_n_updates(1)
        self.assertTranformsAlmostEqual(new_pose, self.rgbd_sensor.get_world_pose(), places=6)

        self.rgbd_sensor.set_local_pose(new_pose)  # parent is world so this should be equivalent
        await self.wait_n_updates(1)
        self.assertTranformsAlmostEqual(new_pose, self.rgbd_sensor.get_world_pose(), places=6)

    async def test_consistent_camera_matrix(self):

        for applied_lens_model in [_lens_model_1(), _lens_model_2()]:
            self.rgbd_sensor.set_lens_model(applied_lens_model)
            await self.wait_n_updates(1)
            sim_lens_model = _read_lensmodel_from_simulation(self.rgbd_sensor.get_camera_sensor())
            self.assertDataclassAlmostEqual(applied_lens_model, sim_lens_model, places=7)

    async def test_set_aperture(self):
        for fnumber in [1.4, 2.8, 5.6, 11.0]:
            self.rgbd_sensor.set_aperture(fnumber)
            await self.wait_n_updates(1)
            rgb_camera: Camera = self.rgbd_sensor.get_camera_sensor()
            self.assertAlmostEqual(rgb_camera.get_lens_aperture(), fnumber, places=5)

    async def test_fov(self):
        self.rgbd_sensor.set_lens_model(_lens_model_1())
        self.rgbd_sensor.initialize()
        await self._validate_correct_fov(places=1, quatile_thresholds={99: 5e-1, 90: 1e-2, 75: 5e-2, 50: 5e-3})

    async def test_fov_offcentered_sensor(self):
        centered_lens_model = _lens_model_1()

        scales = [0.8, 1.2]

        for s in scales:

            lens_model = replace(
                centered_lens_model,
                camera_matrix=replace(
                    centered_lens_model.camera_matrix,
                    fx=centered_lens_model.camera_matrix.fx * s,
                    fy=centered_lens_model.camera_matrix.fy * s,
                ),
            )
            self.rgbd_sensor.set_lens_model(lens_model)
            await self._validate_correct_fov(places=None, quatile_thresholds={99: 5e-1, 90: 5e-1, 75: 5e-2, 50: 5e-3})

    async def test_fov_distorted_sensor(self):
        distorted_lens_model = _lens_model_3()
        self.rgbd_sensor.set_lens_model(distorted_lens_model)
        await self._validate_correct_fov(places=None, quatile_thresholds={99: 5e-1, 90: 5e-1, 75: 5e-2, 50: 5e-3})

    def _validate_capture(self, expected_resolution: Resolution):
        rgb = self.rgbd_sensor.get_data_rgb()
        self.assertIsNotNone(rgb, "RGB data should not be None")
        self.assertEqual(
            rgb.shape, (expected_resolution.height, expected_resolution.width, 3), "RGB data shape mismatch"
        )
        xyz = self.rgbd_sensor.get_data_xyz()
        self.assertIsNotNone(xyz, "XYZ data should not be None")
        self.assertEqual(
            xyz.shape, (expected_resolution.height, expected_resolution.width, 3), "XYZ data shape mismatch"
        )

        depth = self.rgbd_sensor.get_data_depth()
        self.assertIsNotNone(depth, "Depth data should not be None")
        self.assertEqual(
            depth.shape, (expected_resolution.height, expected_resolution.width), "Depth data shape mismatch"
        )

    async def _validate_correct_fov(self, places: int | None = 1, quatile_thresholds: dict[float, float] | None = None):
        self.rgbd_sensor.initialize()

        init_pose = Transform(
            t=np.array([0.0, 0.0, 1.0]), rot=Rotation.from_euler(np.array([np.pi / 4, np.pi, 0.0]), "zyx")
        )

        zs = [1.0]
        rolls = np.linspace(-30, 30, 3)
        pitchs = np.linspace(-30, 30, 3)

        for z in zs:
            for roll in rolls:
                for pitch in pitchs:
                    pose = Transform(
                        t=np.array([0.0, 0.0, z]),
                        rot=init_pose.rot * Rotation.from_euler(np.array([roll, pitch, 0.0]), "xyz", degrees=True),
                    )

                    assert pose.rot.to_matrix()[2, 2] < 0, "Camera looking away from ground plane"
                    self.rgbd_sensor.set_world_pose(pose)
                    await self.wait_n_updates(10)  # Wait rendering

                    depth = self.rgbd_sensor.get_data_depth()
                    depth_ref = _render_ground_plane_depth_image(pose, self.rgbd_sensor.get_lens_model())

                    depth_ref = np.where(
                        depth_ref < self.rgbd_sensor.get_camera_sensor().get_clipping_range()[1], depth_ref, np.inf
                    )

                    self.assertArraysAlmostEqual(
                        depth,
                        depth_ref,
                        places=places,
                        msg="Depth data should be close to rendered reference."
                        f"z={z}, roll={roll * 180:.2f}, pitch={pitch:.2f}",
                        quantile_thresholds=quatile_thresholds,
                    )


def _lens_model_1() -> LensModel:
    resolution = Resolution(width=640, height=480)
    camera_matrix = CameraMatrix(fx=600, fy=600, cx=320, cy=240)
    distortion = DistortionCoefficients(k1=0.044, k2=-0.005, k3=-0.276)
    pixel_size = 1.6e-6 * 4
    focus_distance = 0.6
    return LensModel(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        pixel_size=pixel_size,
        focus_distance=focus_distance,
        resolution=resolution,
    )


def _lens_model_2() -> LensModel:
    resolution = Resolution(width=640, height=480)
    camera_matrix = CameraMatrix(fx=1100, fy=1100, cx=320, cy=240)
    distortion = DistortionCoefficients(k1=-0.09, k2=0.098, k3=-0.128)
    pixel_size = 1.6e-6 * 4
    focus_distance = 1.0
    return LensModel(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        pixel_size=pixel_size,
        focus_distance=focus_distance,
        resolution=resolution,
    )


def _lens_model_3() -> LensModel:
    lens_model = _lens_model_1()
    return replace(
        lens_model,
        distortion_coefficients=DistortionCoefficients.from_array(lens_model.distortion_coefficients.to_array() * 15),
    )


def _read_lensmodel_from_simulation(camera: Camera) -> LensModel:
    if camera.get_lens_distortion_model() == "opencvPinhole":
        cx, cy, fx, fy, kc = camera.get_opencv_pinhole_properties()
        camera_matrix = CameraMatrix(cx=cx, cy=cy, fx=fx, fy=fy)
        distortion = DistortionCoefficients.from_array(np.array(kc))

    else:
        camera_matrix = CameraMatrix.from_array(camera.get_intrinsics_matrix())
        distortion = DistortionCoefficients()

    resolution = Resolution(width=camera.get_resolution()[0], height=camera.get_resolution()[1])
    pixel_size = camera.get_vertical_aperture() / 1 / resolution.height
    focus_distance = camera.get_focus_distance()
    return LensModel(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        pixel_size=pixel_size,
        focus_distance=focus_distance,
        resolution=resolution,
    )


def _render_ground_plane_depth_image(optical_frame: Transform, lens_model: LensModel) -> np.ndarray:
    a, b, c, d = _ground_plane_in_optical_frame(optical_frame)
    n = np.array([a, b, c]).reshape((1, 1, 3))
    rays = _calculate_camera_rays(lens_model)
    r_dot_n = np.sum(rays * n, axis=-1)
    depth = np.where(r_dot_n < -1e-6, -d / r_dot_n, np.inf)
    return depth


def _calculate_camera_rays(lens_model: LensModel) -> np.ndarray:
    uv_undistorted = get_undistorted_normalized_pixels(lens_model)
    x = uv_undistorted[..., 0]
    y = uv_undistorted[..., 1]
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    return rays


def _ground_plane_in_optical_frame(optical_frame: Transform) -> np.ndarray:
    "return plane coefficients [a, b, c, d] for plane ax + by + cz + d = 0"
    point_on_plane_w = np.array([0.0, 0.0, 0.0])  # origin in world frame
    point_on_plane_o = optical_frame.inverse() * point_on_plane_w
    plane_normal_in_world = np.array([0.0, 0.0, 1.0])  # z axis in world frame

    plane_normal_in_optical = optical_frame.rot.inverse() * plane_normal_in_world
    return _plane_coefficients_form_point_normal(point_on_plane_o, plane_normal_in_optical)


def _plane_coefficients_form_point_normal(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    "return plane coefficients [a, b, c, d] for plane ax + by + cz + d = 0"
    d = -point.dot(normal)
    return np.append(normal, d)
