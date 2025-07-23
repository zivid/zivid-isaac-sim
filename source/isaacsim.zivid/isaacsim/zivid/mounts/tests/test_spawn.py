import asyncio

import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.mounts.mount_models import Mount
from isaacsim.zivid.mounts.spawn import spawn_mount
from isaacsim.zivid.utilities.transforms import Rotation, Transform, rot_diff
from pxr import PhysxSchema, UsdLux, UsdPhysics


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

    async def test_spawn_mounts_no_rigid_body(self):
        init_pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.identity())
        diff = Transform(t=np.array([0.0, 0.5, 0]), rot=Rotation.identity())

        poses: list[Transform] = []
        for mount in Mount:
            if len(poses) == 0:
                poses.append(init_pose)
            else:
                poses.append(poses[-1] * diff)

        prims: list[SingleXFormPrim] = []

        for mount, mount_pose in zip(Mount, poses, strict=False):
            prim_path = f"/World/{mount.name}"
            prim = spawn_mount(mount, prim_path, mount_pose, make_rigid_body=False)
            prims.append(prim)
            self.assertIsInstance(prim, SingleXFormPrim)
            prim_pose = Transform.from_isaac_sim(prim.get_world_pose())
            self.assertTranformsAlmostEqual(mount_pose, prim_pose)
            self.assertFalse(prim.prim.HasAPI(UsdPhysics.RigidBodyAPI))
        await self.wait_n_updates(3)

        for prim, pose in zip(prims, poses, strict=False):
            self.assertTranformsAlmostEqual(Transform.from_isaac_sim(prim.get_world_pose()), pose)

    async def test_spawn_mounts_rigid_body(self):
        init_pose = Transform(t=np.array([0.0, 0.0, 1.0]), rot=Rotation.identity())
        diff = Transform(t=np.array([0.0, 0.5, 0]), rot=Rotation.identity())

        poses: list[Transform] = []
        for mount in Mount:
            if len(poses) == 0:
                poses.append(init_pose)
            else:
                poses.append(poses[-1] * diff)

        prims: list[SingleRigidPrim] = []

        for mount, mount_pose in zip(Mount, poses, strict=False):
            prim_path = f"/World/{mount.name}"
            prim = spawn_mount(mount, prim_path, mount_pose, make_rigid_body=True)
            prims.append(prim)
            self.assertIsInstance(prim, SingleRigidPrim)
            prim_pose = Transform.from_isaac_sim(prim.get_world_pose())
            self.assertTranformsAlmostEqual(mount_pose, prim_pose, places=6)
            self.assertTrue(prim.prim.HasAPI(UsdPhysics.RigidBodyAPI))
            geom_prim = prim.prim.GetPrimAtPath("geometry")
            self.assertTrue(geom_prim.IsValid(), "Zivid camera mesh prim should be valid")

            for mesh_prim in geom_prim.GetAllChildren():
                for api in [
                    UsdPhysics.MeshCollisionAPI,
                    UsdPhysics.CollisionAPI,
                    PhysxSchema.PhysxCollisionAPI,
                ]:
                    self.assertTrue(mesh_prim.HasAPI(api), f"Mount mesh prim should have API: {api}")

        await self.wait_n_updates(30)

        for prim, pose in zip(prims, poses, strict=False):
            h = prim.get_world_pose()[0][2]
            self.assertLess(h, pose.t[2], "mount should be falling due to gravity")
            self.assertGreaterEqual(h, 0, "mount should not fall below ground plane")
