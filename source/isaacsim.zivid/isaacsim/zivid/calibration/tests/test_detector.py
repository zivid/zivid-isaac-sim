import asyncio
from pathlib import Path

import cv2
import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.calibration.detector import detect_calibration_board
from isaacsim.zivid.utilities.transforms import Transform, rot_diff
from pxr import UsdLux


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the
# root of module will make it auto-discoverable by omni.kit.test
class Test(omni.kit.test.AsyncTestCase):
    # Before running each test
    # Before running each test
    async def setUp(self):  # pylint: disable=invalid-name

        await create_new_stage_async()
        self.my_world = World(stage_units_in_meters=1.0)
        await self.my_world.initialize_simulation_context_async()
        await update_stage_async()
        self.test_data_dir = Path(__file__).parents[6] / "test_data/calibration"
        self.my_world.scene.add_default_ground_plane()

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

    def assertTranformsAlmostEqual(
        self, a: Transform, b: Transform, places: int | None = None
    ):  # pylint: disable=invalid-name
        diff = np.linalg.norm(a.t - b.t)
        self.assertAlmostEqual(diff, 0, msg="Translation part of transform does not match", places=places)
        diff = rot_diff(a.rot, b.rot).mag()
        self.assertAlmostEqual(diff, 0, msg="Rotation part of transform does not match", places=places)

    async def test_detector(self):
        pngs = [self.test_data_dir / "cb01.png", self.test_data_dir / "cb02.png"]
        for png in pngs:
            rgb = cv2.imread(str(png))

            board = detect_calibration_board(rgb)
            self.assertIsNotNone(board, f"Failed to detect calibration board in image {png}")
