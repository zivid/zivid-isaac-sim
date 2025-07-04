from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pathlib

from enum import IntEnum, auto
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils import prims as prim_utils

import cv2
import cv2.aruco as aruco

from isaacsim.core.utils.stage import add_reference_to_stage

from pxr import UsdPhysics, PhysxSchema, UsdGeom, Gf

from isaacsim.zivid.utilities.transforms import Transform, Rotation
from isaacsim.zivid.utilities.usd import create_fixed_joint
from isaacsim.zivid.utilities.assets import get_assets_path


ASSETS_PATH: pathlib.Path = get_assets_path()


class BoardID(IntEnum):
    ZVDCB01 = 0
    ZVDCB02 = auto()


def add_calibration_board_to_stage(id: BoardID, prim_path: str, world_pose: Transform) -> SingleXFormPrim:
    usd_file = _get_usd_file(id)
    add_reference_to_stage(str(usd_file), prim_path)
    board_prim = SingleXFormPrim(prim_path=prim_path, name=id.name)
    board_prim.set_world_pose(*world_pose.to_isaac_sim())
    return board_prim


def detect_calibration_board(rgb: np.ndarray) -> None | np.ndarray:
    id = _classify_board_from_image(rgb)
    if id is None:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checker_shape = _get_board_shape(id)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, checker_shape, cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def _get_usd_file(id: BoardID) -> pathlib.Path:
    """Get the USD file path for the calibration board."""
    match id:
        case BoardID.ZVDCB01:
            return ASSETS_PATH / "CB01.usd"
        case BoardID.ZVDCB02:
            return ASSETS_PATH / "CB02.usd"
        case _:
            raise ValueError(f"Unknown calibration board ID: {id}")


def _get_prim_name(id: BoardID) -> str:
    """Get the name of the calibration board prim."""
    match id:
        case BoardID.ZVDCB01:
            return "CB01"
        case BoardID.ZVDCB02:
            return "CB02"
        case _:
            raise ValueError(f"Unknown calibration board ID: {id}")


def _classify_board_from_image(image: np.ndarray) -> BoardID | None:
    """Classify the board ID from an image using ArUco markers."""
    ids = _detect_aruco_id(image)

    if ids is None or len(ids) != 1:
        return None

    # Assuming only one marker is present in the image
    aruco_id = ids[0][0]  # Extract the first ID from the array
    return _classify_by_aruco_id(aruco_id)


def _detect_aruco_id(image: np.ndarray) -> int:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect the markers
    _, ids, _ = aruco.ArucoDetector(aruco_dict, parameters).detectMarkers(gray)

    return ids


def _classify_by_aruco_id(id: int) -> BoardID | None:
    """Classify the board ID based on the ArUco ID."""
    if id == 1:
        return BoardID.ZVDCB01
    elif id == 239:
        return BoardID.ZVDCB02
    else:
        return None


def _get_board_shape(id: BoardID) -> tuple[int, int]:
    """Get the shape of the calibration board based on its ID."""
    match id:
        case BoardID.ZVDCB01:
            return (5, 4)
        case BoardID.ZVDCB02:
            return (7, 6)
        case _:
            raise ValueError(f"Unknown calibration board ID: {id}")
