from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.zivid.mounts.mount_models import Mount, MountData
from isaacsim.zivid.utilities.transforms import Transform
from pxr import PhysxSchema, UsdPhysics


def spawn_mount(
    mount: Mount, prim_path: str, world_pose: Transform, make_rigid_body=True
) -> SingleXFormPrim | SingleRigidPrim:
    data = MountData.from_mount(mount)
    add_reference_to_stage(usd_path=str(data.usd_file), prim_path=prim_path)
    mount_prim = SingleXFormPrim(prim_path=prim_path)
    mount_prim.set_world_pose(*world_pose.to_isaac_sim())

    if make_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(mount_prim.prim)
        mass_api = UsdPhysics.MassAPI.Apply(mount_prim.prim)
        mass_api.CreateMassAttr(data.mass)
        geom_prim = mount_prim.prim.GetChild("geometry")

        for mesh_prim in geom_prim.GetAllChildren():
            usd_collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
            usd_collision_api.CreateCollisionEnabledAttr(True)
            usd_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            usd_collision_api.CreateApproximationAttr("convexDecomposition")
            PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
        mount_prim = SingleRigidPrim(prim_path=prim_path)

    return mount_prim
