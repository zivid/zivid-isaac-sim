import asyncio
from pathlib import Path

import cv2
import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.calibration.calibration_board import BoardID
from isaacsim.zivid.calibration.detection_result import DetectionStatus
from isaacsim.zivid.calibration.detector import (
    detect_calibration_board,
    draw_corners,
    get_ground_truth_feature_points_3d,
)
from isaacsim.zivid.frame import Frame
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

    async def test_detector(self):
        pngs = [self.test_data_dir / "cb01.png", self.test_data_dir / "cb02.png"]
        for png in pngs:
            rgb = cv2.imread(str(png))
            xyz = np.ones_like(rgb, dtype=np.float32)
            frame = Frame(rgb=rgb, xyz=xyz)
            detection_result = detect_calibration_board(frame)
            self.assertTrue(detection_result.valid(), f"Detection failed for {png}")

            rgb_corners = draw_corners(frame, detection_result)
            self.assertArraysAlmostEqual(rgb_corners, rgb, quantile_thresholds={99: 1e-7})
            self.assertFalse(np.allclose(rgb_corners, rgb))

            xyz[:] = np.nan
            frame = Frame(rgb=rgb, xyz=xyz)
            detection_result = detect_calibration_board(frame)
            self.assertFalse(detection_result.valid(), f"Detection should fail for {png} with no depth")
            self.assertEqual(
                detection_result.status,
                DetectionStatus.INSUFFICIENT_3D_QUALITY,
                "Detection status should be INSUFFICIENT_3D_QUALITY",
            )
            self.assertRaises(ValueError, draw_corners, frame, detection_result)

    def test_get_ground_truth_feature_points_3d(self):

        for transform in [
            Transform.identity(),
            Transform(t=np.array([1.0, 2.0, 3.0]), rot=Rotation.from_euler(np.array([0.1, 0.2, 0.3]))),
        ]:
            for board_id in BoardID:
                feature_points_3d = get_ground_truth_feature_points_3d(board_id, transform)

                rows, cols = {BoardID.ZVDCB01: (6, 7), BoardID.ZVDCB02: (4, 5)}[board_id]
                checker_size = {BoardID.ZVDCB01: 30 * 1e-3, BoardID.ZVDCB02: 20 * 1e-3}[board_id]
                expected_feature_points_3d = np.zeros((rows, cols, 3), dtype=np.float32)
                x = 0.0
                y = 0.0
                for c in range(cols):
                    for r in range(rows):
                        expected_feature_points_3d[r, c] = transform * np.array([x, y, 0.0])
                        y += checker_size
                    y = 0.0
                    x += checker_size

                self.assertArraysAlmostEqual(
                    feature_points_3d, expected_feature_points_3d, places=7, quantile_thresholds={99: 5e-6}
                )
