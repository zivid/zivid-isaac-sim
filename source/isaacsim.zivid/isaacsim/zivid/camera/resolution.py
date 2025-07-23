from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from isaacsim.zivid.camera.models import ZividCameraModelName


class SamplingMode(IntEnum):
    FULL = 0
    DOWNSAMPLE2X2 = 1
    DOWNSAMPLE4X4 = 2


@dataclass
class Resolution:
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)


def get_camera_resolution(sampling_mode: SamplingMode, model_name: ZividCameraModelName) -> Resolution:
    full_res = _get_full_res(model_name)
    match sampling_mode:
        case SamplingMode.FULL:
            return full_res
        case SamplingMode.DOWNSAMPLE2X2:
            return Resolution(width=full_res.width // 2, height=full_res.height // 2)
        case SamplingMode.DOWNSAMPLE4X4:
            return Resolution(width=full_res.width // 4, height=full_res.height // 4)
        case _:
            raise ValueError(f"Unknown sampling: {sampling_mode}")


def get_projector_resolution(model_name: ZividCameraModelName) -> Resolution:
    if model_name in [ZividCameraModelName.ZIVID_2_M70, ZividCameraModelName.ZIVID_2_L100]:
        return Resolution(width=1280 // 4, height=720 // 4)
    return Resolution(width=1000 // 4, height=720 // 4)


def _get_full_res(model_name: ZividCameraModelName) -> Resolution:
    if model_name in [ZividCameraModelName.ZIVID_2_M70, ZividCameraModelName.ZIVID_2_L100]:
        return Resolution(width=1944, height=1200)
    return Resolution(width=2448, height=2048)
