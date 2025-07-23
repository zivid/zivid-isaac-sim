import asyncio

import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path_async
from isaacsim.zivid.assembler import RobotAssembly, assemble_zivid_caseing_on_robot
from isaacsim.zivid.camera import ZividCameraModelName
from isaacsim.zivid.mounts import Mount
from isaacsim.zivid.mounts.mount_models import MountData
from isaacsim.zivid.utilities.transforms import Rotation, Transform, rot_diff
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
        self.my_world.scene.add_default_ground_plane()

        assets_root_path = await get_assets_root_path_async()
        franka_usd_path = f"{assets_root_path}/Isaac/Robots/Franka/franka.usd"

        add_reference_to_stage(usd_path=franka_usd_path, prim_path="/World/Franka")
        self.franka = SingleArticulation(prim_path="/World/Franka", name="franka")
        self.my_world.scene.add(self.franka)

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

    async def test_assemble_on_robot(self):
        mount = Mount.OA121
        mount_data = MountData.from_mount(mount)
        attach_prim_path = "/World/Franka/panda_hand"
        attach_prim = SingleRigidPrim(attach_prim_path, name="attach_prim")

        robot_prim_path = "/World/Franka"
        mount_local_pose = Transform(
            rot=Rotation.from_matrix(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])), t=np.array([0.0, 0.0, 0.0])
        )

        model_name = ZividCameraModelName.ZIVID_2_PLUS_MR130
        assembly = assemble_zivid_caseing_on_robot(
            model_name, attach_prim_path, robot_prim_path, mount, mount_local_pose=mount_local_pose
        )
        await self.my_world.reset_async()
        self.my_world.scene.remove_object("franka", True)
        self.franka = SingleArticulation(prim_path="/World/Franka", name="franka")
        self.my_world.scene.add(self.franka)
        await self.my_world.reset_async()

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
