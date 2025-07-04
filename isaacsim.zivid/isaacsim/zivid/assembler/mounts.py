from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import cadquery
import pathlib

from enum import IntEnum, auto
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils import prims as prim_utils
from pxr import UsdPhysics, PhysxSchema, UsdGeom, Gf
from isaacsim.core.prims import SingleXFormPrim

from isaacsim.zivid.utilities.transforms import Transform, Rotation
from isaacsim.zivid.utilities.usd import create_fixed_joint
from isaacsim.core.utils.stage import add_reference_to_stage

from ..utilities.assets import get_assets_path

ASSETS_PATH: pathlib.Path = get_assets_path()


class Mount(IntEnum):
    """
    Enum for Zivid mounts.
    """

    OA111 = 0
    OA121 = auto()
    OA211 = auto()
    OA221 = auto()


@dataclass(frozen=True)
class _MountData:
    usd_file: pathlib.Path
    mass: float
    name: str
    zivid_mount_frame: Transform

    @classmethod
    def from_mount(cls, mount: Mount) -> _MountData:
        match mount:
            case Mount.OA111:
                return cls(
                    ASSETS_PATH / "oa111.usd",
                    0.170,
                    "On-Arm OA 111",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]), rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5]))
                    ),
                )
            case Mount.OA121:
                return cls(
                    ASSETS_PATH / "oa121.usd",
                    0.170,
                    "On-Arm OA 121",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]),
                        rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5]))
                        * Rotation.from_axis_angle(np.array([0, -15 * np.pi / 180, 0])),
                    ),
                )
            case Mount.OA211:
                return cls(
                    ASSETS_PATH / "oa211.usd",
                    0.207,
                    "On-Arm OA 211",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]),
                        rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5])),
                    ),
                )
            case Mount.OA221:
                return cls(
                    ASSETS_PATH / "oa221.usd",
                    0.210,
                    "On-Arm OA 221",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]),
                        rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5]))
                        * Rotation.from_axis_angle(np.array([0, -15 * np.pi / 180, 0])),
                    ),
                )


def attach_mount_to_robot(
    mount: Mount, manipulator_prim_path: str, attach_prim_path: str, local_pose: Transform
) -> str:
    data = _MountData.from_mount(mount)

    mount_prim_path = manipulator_prim_path + "/" + mount.name
    geom_prim_path = mount_prim_path + "/geometry"

    mount_prim = add_reference_to_stage(usd_path=str(data.usd_file), prim_path=mount_prim_path)

    attach_prim = SingleRigidPrim(attach_prim_path, name=data.name)
    attach_world_pose = Transform.from_isaac_sim(attach_prim.get_world_pose())
    mount_world_pose = attach_world_pose * local_pose
    SingleXFormPrim(prim_path=mount_prim_path).set_world_pose(
        mount_world_pose.t.tolist(), mount_world_pose.rot.to_quat().tolist()
    )

    UsdPhysics.RigidBodyAPI.Apply(mount_prim)
    mass_api = UsdPhysics.MassAPI.Apply(mount_prim)
    mass_api.CreateMassAttr(data.mass)

    geom_prim = mount_prim.GetChild("geometry")

    for mesh_prim in geom_prim.GetAllChildren():
        usd_collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        usd_collision_api.CreateCollisionEnabledAttr(True)
        usd_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        usd_collision_api.CreateApproximationAttr("convexDecomposition")
        PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)

    fj = create_fixed_joint(
        attach_prim_path,
        attach_prim_path,
        mount_prim_path,
        local_pose,
        mount.name + "FixedJoint",
    )
    return mount_prim_path


def get_zivid_mount_pose(mount: Mount, mount_pose: Transform) -> Transform:
    return mount_pose * (_MountData.from_mount(mount).zivid_mount_frame)


def get_mount_pose_from_zivid_pose(mount: Mount, zivid_pose: Transform) -> Transform:
    return zivid_pose * (_MountData.from_mount(mount).zivid_mount_frame.inverse())


def create_mount_zivid_fixed_joint(mount: Mount, mount_prim_path: str, zivid_prim_path: str) -> None:
    data = _MountData.from_mount(mount)
    create_fixed_joint(
        mount_prim_path,
        mount_prim_path,
        zivid_prim_path,
        data.zivid_mount_frame,
        "ZividFixedJoint",
    )


def move_mount(
    mount: Mount,
    attach_prim_path: str,
    mount_prim_path: str,
    local_pose: Transform,
) -> None:
    """
    Move the mount to a new position and orientation.

    Args:
        mount_prim_path (str): Prim path of the mount.
        manipulator_prim_path (str): Prim path of the manipulator.
        local_pose (Transform): New local pose for the mount.
    """

    mount_prim = SingleRigidPrim(mount_prim_path, name="mount")
    mount_prim.set_local_pose(translation=local_pose.t, orientation=local_pose.rot.to_quat())
    create_fixed_joint(
        attach_prim_path,
        attach_prim_path,
        mount_prim_path,
        local_pose,
        mount.name + "FixedJoint",
    )
