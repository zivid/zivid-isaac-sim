import asyncio

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.spawn import spawn_zivid_caseing
from isaacsim.zivid.utilities.transforms import Rotation, Transform
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
        UsdLux.DistantLight.Define(stage, light_path)

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

    async def test_spawn_zivid_caseing_no_rigid_body(self):
        zivid_prim_path = "/World/ZividCamera"
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_pose = Transform(
            t=np.array([0.0, 0.0, 1.0]),
            rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])),
        )
        zivid_prim = spawn_zivid_caseing(
            model_name=model_name,
            prim_path=zivid_prim_path,
            world_pose=zivid_pose,
            make_rigid_body=False,
        )

        self.assertTrue(isinstance(zivid_prim, SingleXFormPrim), "Zivid camera prim should be a SingleXFormPrim")
        prim_pose = Transform.from_isaac_sim(zivid_prim.get_world_pose())
        self.assertTrue(
            np.allclose(prim_pose.t, zivid_pose.t), "Zivid camera translation should match the specified world pose"
        )
        self.assertTrue(
            np.allclose(prim_pose.rot.as_quat(), zivid_pose.rot.as_quat()),
            "Zivid camera rotation should match the specified world pose",
        )
        self.assertTrue(
            zivid_prim.prim_path == zivid_prim_path, "Zivid camera prim path should match the specified prim path"
        )

        prim = prim_utils.get_prim_at_path(zivid_prim_path)
        self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI), "Zivid camera prim should not have RigidBodyAPI")

    async def test_spawn_zivid_caseing_rigid_body(self):
        zivid_prim_path = "/World/ZividCamera"
        model_name = ZividCameraModelName.ZIVID_2_PLUS_M130
        zivid_pose = Transform(
            t=np.array([0.0, 0.0, 1.0]),
            rot=Rotation.from_axis_angle(np.array([0.0, np.pi / 2, 0.0])),
        )
        zivid_prim = spawn_zivid_caseing(
            model_name=model_name,
            prim_path=zivid_prim_path,
            world_pose=zivid_pose,
            make_rigid_body=True,
        )

        self.assertTrue(isinstance(zivid_prim, SingleRigidPrim), "Zivid camera prim should be a SingleRigidPrim")
        prim_pose = Transform.from_isaac_sim(zivid_prim.get_world_pose())
        self.assertTrue(
            np.allclose(prim_pose.t, zivid_pose.t), "Zivid camera translation should match the specified world pose"
        )
        self.assertTrue(
            np.allclose(prim_pose.rot.as_quat(), zivid_pose.rot.as_quat()),
            "Zivid camera rotation should match the specified world pose",
        )
        self.assertTrue(
            zivid_prim.prim_path == zivid_prim_path, "Zivid camera prim path should match the specified prim path"
        )

        prim = prim_utils.get_prim_at_path(zivid_prim_path)

        self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI), "Zivid camera prim should have RigidBodyAPI")

        mesh_prim = prim.GetPrimAtPath("geometry/mesh")
        self.assertTrue(mesh_prim.IsValid(), "Zivid camera mesh prim should be valid")

        for api in [
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.CollisionAPI,
            PhysxSchema.PhysxCollisionAPI,
        ]:
            self.assertTrue(mesh_prim.HasAPI(api), f"Zivid camera prim should have API: {api}")

        await self.wait_n_updates(100)

        prim_pose = Transform.from_isaac_sim(zivid_prim.get_world_pose())
        print(prim_pose.t)
        self.assertTrue(zivid_pose.t[2] > prim_pose.t[2], "Zivid camera should be falling due to gravity")
        self.assertTrue(prim_pose.t[2] > 0.0, "Zivid camera should not fall below the ground plane")

    # async def test_window_button(self):
    #    # Find a label in our window
    #    label = ui_test.find("{{ extension_display_name }}//Frame/**/Label[*]")


#
#    # Find buttons in our window
#    add_button = ui_test.find(
#        "{{ extension_display_name }}//Frame/**/Button[*].text=='Add'"
#    )
#    reset_button = ui_test.find(
#        "{{ extension_display_name }}//Frame/**/Button[*].text=='Reset'"
#    )
#
#    # Click reset button
#    await reset_button.click()
#    self.assertEqual(label.widget.text, "empty")
#
#    await add_button.click()
#    self.assertEqual(label.widget.text, "count: 1")
#
#    await add_button.click()
#    self.assertEqual(label.widget.text, "count: 2")
