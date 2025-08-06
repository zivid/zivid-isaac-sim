from __future__ import annotations

import pathlib
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
from isaacsim.zivid.utilities.assets import get_assets_path
from isaacsim.zivid.utilities.transforms import Rotation, Transform

_ASSETS_PATH = get_assets_path()


class Mount(IntEnum):
    """
    Enum for Zivid mounts.
    """

    OA111 = 0
    OA121 = auto()
    OA211 = auto()
    OA221 = auto()


@dataclass(frozen=True)
class MountData:
    usd_file: pathlib.Path
    mass: float
    name: str
    zivid_mount_frame: Transform

    @classmethod
    def from_mount(cls, mount: Mount) -> MountData:
        match mount:
            case Mount.OA111:
                return cls(
                    _ASSETS_PATH / "oa111.usd",
                    0.170,
                    "On-Arm OA 111",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]), rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5]))
                    ),
                )
            case Mount.OA121:
                return cls(
                    _ASSETS_PATH / "oa121.usd",
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
                    _ASSETS_PATH / "oa211.usd",
                    0.207,
                    "On-Arm OA 211",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]),
                        rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5])),
                    ),
                )
            case Mount.OA221:
                return cls(
                    _ASSETS_PATH / "oa221.usd",
                    0.210,
                    "On-Arm OA 221",
                    Transform(
                        t=np.array([0.025, 0.065, -0.002]),
                        rot=Rotation.from_quat(np.array([-0.5, 0.5, 0.5, 0.5]))
                        * Rotation.from_axis_angle(np.array([0, -15 * np.pi / 180, 0])),
                    ),
                )
