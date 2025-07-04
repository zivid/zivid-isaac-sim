import numpy as np
import pathlib

from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils import prims as prim_utils
from pxr import UsdPhysics, PhysxSchema, Sdf
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.core.prims import SingleXFormPrim
from isaacsim.zivid.cameras.models import ZividCameraModelName
from isaacsim.zivid.cameras.zivid_camera import ZividCameraModel
from isaacsim.zivid.utilities.transforms import Transform
from isaacsim.zivid.utilities.assets import get_assets_path
from isaacsim.zivid.assembler.mounts import (
    Mount,
    attach_mount_to_robot,
    get_zivid_mount_pose,
    get_mount_pose_from_zivid_pose,
    create_mount_zivid_fixed_joint,
)
from ..utilities.assets import get_assets_path

ASSETS_PATH: pathlib.Path = get_assets_path()


def assemble_zivid(
    model_name: ZividCameraModelName, prim_path: str, world_pose: Transform, make_rigid_body: bool = True
) -> str:
    zivid_prim_path = prim_path
    zivid_prim = add_reference_to_stage(
        usd_path=str(get_assets_path() / "zivid2.usd"),
        prim_path=zivid_prim_path,
    )
    SingleXFormPrim(prim_path=zivid_prim_path).set_world_pose(*world_pose.to_isaac_sim())

    SingleXFormPrim(prim_path=zivid_prim_path).set_world_pose(world_pose.t.tolist(), world_pose.rot.to_quat().tolist())

    zivid_prim.CreateAttribute("is_zivid_camera", Sdf.ValueTypeNames.Bool).Set(True)
    zivid_prim.CreateAttribute("zivid_camera_model", Sdf.ValueTypeNames.Int).Set(int(model_name.value))

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

    return zivid_prim_path


def assemble_zivid_on_robot(
    model_name: ZividCameraModelName,
    attach_prim_path: str,
    robot_prim_path: str,
    mount: Mount,
    *,
    mount_local_pose: Transform | None = None,
    handeye_calibration: Transform | None = None,
) -> str:

    if mount_local_pose is None and handeye_calibration is None:
        raise ValueError("Either mount_local_pose or handeye_calibration must be provided.")
    if mount_local_pose is not None and handeye_calibration is not None:
        raise ValueError("Only one of mount_local_pose or handeye_calibration should be provided.")

    if mount_local_pose is None:
        optical_frame = ZividCameraModel.optical_frame_from_model_name(model_name)
        mount_local_pose = get_mount_pose_from_zivid_pose(mount, handeye_calibration * optical_frame.inverse())

    return _assemble_zivid_on_robot(
        model_name=model_name,
        attach_prim_path=attach_prim_path,
        robot_prim_path=robot_prim_path,
        mount=mount,
        mount_local_pose=mount_local_pose,
    )


def _assemble_zivid_on_robot(
    model_name: ZividCameraModelName,
    attach_prim_path: str,
    robot_prim_path: str,
    mount: Mount,
    *,
    mount_local_pose: Transform,
) -> str:

    mount_prim_path = attach_mount_to_robot(
        mount=mount,
        manipulator_prim_path=robot_prim_path,
        attach_prim_path=attach_prim_path,
        local_pose=mount_local_pose,
    )

    zivid_prim_path = robot_prim_path + "/Zivid"

    mount_prim = SingleRigidPrim(mount_prim_path, name="attach")
    mount_world_pose = Transform.from_isaac_sim(mount_prim.get_world_pose())
    zivid_world_pose = get_zivid_mount_pose(mount, mount_world_pose)

    assemble_zivid(model_name, zivid_prim_path, zivid_world_pose, make_rigid_body=True)
    create_mount_zivid_fixed_joint(
        mount=mount,
        mount_prim_path=mount_prim_path,
        zivid_prim_path=zivid_prim_path,
    )

    return zivid_prim_path
