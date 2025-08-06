import pathlib

from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.utilities.assets import get_assets_path
from isaacsim.zivid.utilities.transforms import Transform
from pxr import PhysxSchema, UsdPhysics

# from isaacsim.zivid.mounts.mounts import (
#    Mount,
#    attach_mount_to_robot,
#    get_zivid_mount_pose,
#    get_mount_pose_from_zivid_pose,
#    create_mount_zivid_fixed_joint,
# )

ASSETS_PATH: pathlib.Path = get_assets_path()


def spawn_zivid_casing(
    model_name: ZividCameraModelName,  # pylint: disable=unused-argument #TODO: remove model_name argument from api
    prim_path: str,
    world_pose: Transform,
    make_rigid_body: bool = True,
) -> SingleXFormPrim | SingleRigidPrim:
    zivid_prim_path = prim_path
    zivid_prim = add_reference_to_stage(
        usd_path=str(get_assets_path() / "zivid2.usd"),
        prim_path=zivid_prim_path,
    )
    prim = SingleXFormPrim(prim_path=zivid_prim_path)
    prim.set_world_pose(*world_pose.to_isaac_sim())
    prim.set_world_pose(world_pose.t.tolist(), world_pose.rot.as_quat().tolist())

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
