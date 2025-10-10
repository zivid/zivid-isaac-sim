from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.zivid.transforms import Transform
from isaacsim.zivid.utilities.assets import get_assets_path
from pxr import PhysxSchema, UsdPhysics


def spawn_zivid_casing(
    prim_path: str,
    world_pose: Transform,
    make_rigid_body: bool = True,
) -> SingleXFormPrim | SingleRigidPrim:
    """
    Adds the physical Zivid camera to the stage. Optionally it makes the prim a rigid body.

    Args:
        prim_path: The prim path of the Zivid prim.
        world_pose: The world pose of the Zivid prim.
        make_rigid_body: If true, the prim will be a rigid body with colliders. Default is True.

    Returns:
        SingleXFormPrim | SingleRigidPrim: The spawned Zivid prim.
    """

    zivid_prim_path = prim_path
    zivid_prim = add_reference_to_stage(
        usd_path=str(get_assets_path() / "zivid2.usd"),
        prim_path=zivid_prim_path,
    )
    prim = SingleXFormPrim(prim_path=zivid_prim_path)
    prim.set_world_pose(*world_pose.to_isaac_sim())

    if make_rigid_body:
        mesh_prim = zivid_prim.GetPrimAtPath("geometry/mesh")
        UsdPhysics.RigidBodyAPI.Apply(zivid_prim)
        mass_api = UsdPhysics.MassAPI.Apply(zivid_prim)
        mass_api.CreateMassAttr(0.950)
        usd_collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        usd_collision_api.CreateCollisionEnabledAttr(True)
        usd_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        usd_collision_api.CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
        prim = SingleRigidPrim(prim_path=zivid_prim_path)

    return prim
