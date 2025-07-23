from dataclasses import dataclass

from isaacsim.core.prims import SingleRigidPrim
from isaacsim.zivid.camera import ZividCameraModelName, spawn_zivid_caseing
from isaacsim.zivid.camera.zivid_camera_model import ZividCameraModel
from isaacsim.zivid.mounts import Mount, spawn_mount
from isaacsim.zivid.mounts.mount_models import MountData
from isaacsim.zivid.utilities.transforms import Transform
from isaacsim.zivid.utilities.usd import create_fixed_joint


@dataclass
class RobotAssembly:
    zivid_prim: SingleRigidPrim
    mount_prim: SingleRigidPrim


def assemble_zivid_caseing_on_robot(  # pylint: disable=too-many-arguments
    model_name: ZividCameraModelName,
    attach_prim_path: str,
    robot_prim_path: str,
    mount: Mount,
    *,
    mount_local_pose: Transform | None = None,
    handeye_calibration: Transform | None = None,
) -> RobotAssembly:

    if mount_local_pose is None:
        if handeye_calibration is None:
            raise ValueError("handeye_calibration must be provided if mount_local_pose is not given.")
        optical_frame = ZividCameraModel.optical_frame_from_model_name(model_name)
        mount_local_pose = _get_mount_pose_from_zivid_pose(mount, handeye_calibration * optical_frame.inverse())

    return _assemble_zivid_on_robot(
        model_name=model_name,
        attach_prim_path=attach_prim_path,
        robot_prim_path=robot_prim_path,
        mount=mount,
        mount_local_pose=mount_local_pose,
    )


def move_mount(
    mount: Mount,
    attach_prim_path: str,
    mount_prim_path: str,
    local_pose: Transform,
) -> None:

    mount_prim = SingleRigidPrim(mount_prim_path, name="mount")
    mount_prim.set_local_pose(translation=local_pose.t, orientation=local_pose.rot.as_quat())
    create_fixed_joint(
        attach_prim_path,
        attach_prim_path,
        mount_prim_path,
        local_pose,
        mount.name + "FixedJoint",
    )


def _assemble_zivid_on_robot(
    model_name: ZividCameraModelName,
    attach_prim_path: str,
    robot_prim_path: str,
    mount: Mount,
    *,
    mount_local_pose: Transform,
) -> RobotAssembly:

    mount_prim = _attach_mount_to_robot(
        mount=mount,
        manipulator_prim_path=robot_prim_path,
        attach_prim_path=attach_prim_path,
        local_pose=mount_local_pose,
    )

    zivid_prim_path = robot_prim_path + "/Zivid"

    mount_world_pose = Transform.from_isaac_sim(mount_prim.get_world_pose())
    zivid_world_pose = _get_zivid_mount_pose(mount, mount_world_pose)

    zivid_prim = spawn_zivid_caseing(model_name, zivid_prim_path, zivid_world_pose, make_rigid_body=True)
    _create_mount_zivid_fixed_joint(
        mount=mount,
        mount_prim_path=mount_prim.prim_path,
        zivid_prim_path=zivid_prim_path,
    )

    return RobotAssembly(zivid_prim=zivid_prim, mount_prim=mount_prim)


def _attach_mount_to_robot(
    mount: Mount, manipulator_prim_path: str, attach_prim_path: str, local_pose: Transform
) -> SingleRigidPrim:
    data = MountData.from_mount(mount)

    mount_prim_path = manipulator_prim_path + "/" + mount.name

    attach_prim = SingleRigidPrim(attach_prim_path, name=data.name)
    attach_world_pose = Transform.from_isaac_sim(attach_prim.get_world_pose())
    mount_world_pose = attach_world_pose * local_pose
    prim = spawn_mount(mount, mount_prim_path, mount_world_pose, make_rigid_body=True)

    create_fixed_joint(
        attach_prim_path,
        attach_prim_path,
        mount_prim_path,
        local_pose,
        mount.name + "FixedJoint",
    )
    return prim


def _get_zivid_mount_pose(mount: Mount, mount_pose: Transform) -> Transform:
    return mount_pose * (MountData.from_mount(mount).zivid_mount_frame)


def _get_mount_pose_from_zivid_pose(mount: Mount, zivid_pose: Transform) -> Transform:
    return zivid_pose * (MountData.from_mount(mount).zivid_mount_frame.inverse())


def _create_mount_zivid_fixed_joint(mount: Mount, mount_prim_path: str, zivid_prim_path: str) -> None:
    data = MountData.from_mount(mount)
    create_fixed_joint(
        mount_prim_path,
        mount_prim_path,
        zivid_prim_path,
        data.zivid_mount_frame,
        "ZividFixedJoint",
    )
