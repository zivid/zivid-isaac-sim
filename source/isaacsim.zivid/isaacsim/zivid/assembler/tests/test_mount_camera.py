import asyncio

import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.assembler import RobotAssembly, assemble_zivid_casing_on_robot
from isaacsim.zivid.camera import ZividCamera, ZividCameraModelName
from isaacsim.zivid.mounts import Mount
from isaacsim.zivid.mounts._internal import MountData
from isaacsim.zivid.transforms import Rotation, Transform
from isaacsim.zivid.utilities.assets import AssetPaths
from isaacsim.zivid.utilities.test_utils import BaseTest
from pxr import UsdLux


class Test(BaseTest):
    async def setUp(self):  # pylint: disable=invalid-name

        await create_new_stage_async()
        self.my_world: World = World(stage_units_in_meters=1.0)
        await self.my_world.initialize_simulation_context_async()
        assert isinstance(self.my_world, World)
        await update_stage_async()
        self.my_world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=str(AssetPaths.franka), prim_path="/World/Franka")
        self.franka = SingleArticulation(prim_path="/World/Franka", name="franka")

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
        # Wait for the stage to finish loading)
        self.franka.initialize()

        self.franka.set_joint_positions(self.franka.get_joints_default_state().positions)

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
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        return

    async def test_assemble_on_robot(self):
        mount = Mount.OA121
        mount_data = MountData.from_mount(mount)
        attach_prim_path = "/World/Franka/panda_hand"
        attach_prim = SingleRigidPrim(attach_prim_path, name="attach_prim")

        robot_prim_path = "/World/Franka"
        mount_local_pose = Transform(
            rot=Rotation.from_matrix(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])), t=np.array([0.0, 0.0, 0.0])
        )

        assembly = assemble_zivid_casing_on_robot(
            attach_prim_path, robot_prim_path, mount, mount_local_pose=mount_local_pose
        )
        await self.my_world.reset_async()
        self.franka = SingleArticulation(prim_path="/World/Franka", name="franka")

        self.franka.initialize()

        await self.wait_n_updates(3)

        self.assertIsInstance(assembly, RobotAssembly)
        self.assertIsInstance(assembly.zivid_prim, SingleRigidPrim)
        self.assertIsInstance(assembly.mount_prim, SingleRigidPrim)
        self.assertEqual(assembly.zivid_prim.prim_path, "/World/Franka/Zivid")
        self.assertEqual(assembly.mount_prim.prim_path, f"/World/Franka/{mount.name}")
        self.assertIn("Zivid", self.franka._prim_view.body_names)  # pylint: disable=protected-access
        self.assertIn(mount.name, self.franka._prim_view.body_names)  # pylint: disable=protected-access

        def check_consistent_poses(places):
            mount_world_pose = Transform.from_isaac_sim(assembly.mount_prim.get_world_pose())
            zivid_world_pose = Transform.from_isaac_sim(assembly.zivid_prim.get_world_pose())
            attach_pose = Transform.from_isaac_sim(attach_prim.get_world_pose())
            self.assertTranformsAlmostEqual(mount_world_pose, attach_pose * mount_local_pose, places=places)
            self.assertTranformsAlmostEqual(
                zivid_world_pose, mount_world_pose * mount_data.zivid_mount_frame, places=places
            )

        check_consistent_poses(places=4)
        await self.wait_n_updates(3)
        self.franka.initialize()

        check_consistent_poses(places=4)
        current_joint_positions = self.franka.get_joint_positions()
        new_joint_positions = current_joint_positions.copy()
        new_joint_positions[0] += np.pi
        self.franka.set_joint_positions(new_joint_positions)
        await self.wait_n_updates(1)
        check_consistent_poses(places=4)

    async def test_assemble_on_robot_hand_eye(self):
        mount = Mount.OA121
        mount_data = MountData.from_mount(mount)
        attach_prim_path = "/World/Franka/panda_hand"
        attach_prim = SingleRigidPrim(attach_prim_path, name="attach_prim")

        robot_prim_path = "/World/Franka"
        hand_eye = Transform(
            rot=Rotation.from_matrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])), t=np.array([0.05, -0.05, 0.02])
        )
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        assembly = assemble_zivid_casing_on_robot(
            attach_prim_path, robot_prim_path, mount, handeye_calibration=hand_eye, model_name=model_name
        )
        zivid_camera = ZividCamera(model_name=model_name, prim_path=assembly.zivid_prim.prim_path)

        await self.my_world.reset_async()
        self.franka = SingleArticulation(prim_path="/World/Franka", name="franka")
        self.franka.initialize()
        zivid_camera.initialize()
        await self.wait_n_updates(3)

        self.assertIsInstance(assembly, RobotAssembly)
        self.assertIsInstance(assembly.zivid_prim, SingleRigidPrim)
        self.assertIsInstance(assembly.mount_prim, SingleRigidPrim)
        self.assertEqual(assembly.zivid_prim.prim_path, "/World/Franka/Zivid")
        self.assertEqual(assembly.mount_prim.prim_path, f"/World/Franka/{mount.name}")
        self.assertIn("Zivid", self.franka._prim_view.body_names)  # pylint: disable=protected-access
        self.assertIn(mount.name, self.franka._prim_view.body_names)  # pylint: disable=protected-access

        def check_consistent_poses(places):
            mount_world_pose = Transform.from_isaac_sim(assembly.mount_prim.get_world_pose())
            zivid_world_pose = Transform.from_isaac_sim(assembly.zivid_prim.get_world_pose())

            attach_pose = Transform.from_isaac_sim(attach_prim.get_world_pose())
            self.assertTranformsAlmostEqual(
                attach_pose * hand_eye, zivid_camera.get_camera_sensor_world_pose(), places=places
            )
            self.assertTranformsAlmostEqual(
                zivid_world_pose, mount_world_pose * mount_data.zivid_mount_frame, places=places
            )

        check_consistent_poses(places=4)
        await self.wait_n_updates(3)
        self.franka.initialize()

        check_consistent_poses(places=4)
        current_joint_positions = self.franka.get_joint_positions()
        new_joint_positions = current_joint_positions.copy()
        new_joint_positions[0] += np.pi
        self.franka.set_joint_positions(new_joint_positions)
        await self.wait_n_updates(1)
        check_consistent_poses(places=4)
