from dataclasses import dataclass

from isaacsim.core.prims import SingleRigidPrim


@dataclass
class RobotAssembly:
    zivid_prim: SingleRigidPrim
    mount_prim: SingleRigidPrim
