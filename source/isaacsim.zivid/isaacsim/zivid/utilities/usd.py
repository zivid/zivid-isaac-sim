import numpy as np
from isaacsim.core.utils import prims as prim_utils
from isaacsim.zivid.utilities.transforms import Transform
from pxr import Gf, UsdPhysics


def create_fixed_joint(
    prim_path: str,
    target0: str,
    target1: str,
    offset: Transform,
    name: str = "FixedJoint",
) -> UsdPhysics.FixedJoint:

    fixed_joint_path = prim_path + "/" + name
    stage = prim_utils.get_current_stage()
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    if target0 is not None:
        fixed_joint.GetBody0Rel().SetTargets([target0])
    if target1 is not None:
        fixed_joint.GetBody1Rel().SetTargets([target1])

    fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*offset.t.astype(float)))
    fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(*offset.rot.as_quat().astype(float)))
    fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*np.zeros(3).astype(float)))
    fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(*np.array([1, 0, 0, 0]).astype(float)))

    return fixed_joint
