from isaacsim.zivid.assembler._impl import mount_camera_impl as impl
from isaacsim.zivid.assembler.robot_assembly import RobotAssembly
from isaacsim.zivid.camera import ZividCameraModelName
from isaacsim.zivid.mounts import Mount
from isaacsim.zivid.transforms import Transform


def assemble_zivid_casing_on_robot(  # pylint: disable=too-many-arguments
    attach_prim_path: str,
    robot_prim_path: str,
    mount: Mount,
    *,
    mount_local_pose: Transform | None = None,
    handeye_calibration: Transform | None = None,
    model_name: ZividCameraModelName | None = None,
) -> RobotAssembly:
    """
    Assemble a Zivid casing on a robot using a specified mount.

    The mount is either attached at the given local pose if provided.
    Optionally, the handeye calibration and camera model can be provided.
    The mount pose will be computed based on these. Note: either the local pose,
    or both the handeye calibration and camera model, must be provided.

    Args:
        attach_prim_path (str): The prim path where the mount will be attached.
        robot_prim_path (str): The prim path of the robot.
        mount (Mount): The mount to use for attaching the Zivid casing.
        mount_local_pose (Transform | None, optional): The local pose of the mount relative to the attach prim.
            Defaults to None.
        handeye_calibration (Transform | None, optional): The handeye calibration of the camera. Required if
            mount_local_pose is None. Defaults to None.
        model_name (ZividCameraModelName | None, optional): The model name of the Zivid camera. Required if
            mount_local_pose is None. Defaults to None.

    Returns:
        RobotAssembly: The assembled robot with the Zivid prim and mount prim.
    """

    return impl.assemble_zivid_casing_on_robot_impl(
        attach_prim_path=attach_prim_path,
        robot_prim_path=robot_prim_path,
        mount=mount,
        mount_local_pose=mount_local_pose,
        handeye_calibration=handeye_calibration,
        model_name=model_name,
    )


def move_mount(
    mount: Mount,
    attach_prim_path: str,
    mount_prim_path: str,
    local_pose: Transform,
) -> None:
    """
    Move an existing mount to a new local pose relative to the attach prim.

    Args:
        mount (Mount): The mount to move.
        attach_prim_path (str): The prim path where the mount is attached.
        mount_prim_path (str): The prim path of the mount.
        local_pose (Transform): The new local pose of the mount relative to the attach prim.
    """

    impl.move_mount_impl(
        mount=mount,
        attach_prim_path=attach_prim_path,
        mount_prim_path=mount_prim_path,
        local_pose=local_pose,
    )
