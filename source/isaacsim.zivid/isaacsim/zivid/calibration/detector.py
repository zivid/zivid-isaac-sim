from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np
from isaacsim.zivid.calibration.calibration_board import BoardID


def detect_calibration_board(rgb: np.ndarray) -> None | np.ndarray:
    aruco_id = _classify_board_from_image(rgb)
    if aruco_id is None:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checker_shape = _get_board_shape(aruco_id)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, checker_shape, cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def _classify_board_from_image(image: np.ndarray) -> BoardID | None:
    """Classify the board ID from an image using ArUco markers."""
    ids = _detect_aruco_ids(image)

    if ids is None or len(ids) != 1:
        return None

    # Assuming only one marker is present in the image
    aruco_id = ids[0][0]  # Extract the first ID from the array
    return _classify_by_aruco_id(aruco_id)


def _detect_aruco_ids(image: np.ndarray) -> np.ndarray | None:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect the markers
    _, ids, _ = aruco.ArucoDetector(aruco_dict, parameters).detectMarkers(gray)

    return ids


def _classify_by_aruco_id(aruco_id: int) -> BoardID | None:
    """Classify the board ID based on the ArUco ID."""

    match aruco_id:
        case 1:
            return BoardID.ZVDCB01
        case 239:
            return BoardID.ZVDCB02
        case _:
            return None


def _get_board_shape(board_id: BoardID) -> tuple[int, int]:
    """Get the shape of the calibration board based on its ID."""
    match board_id:
        case BoardID.ZVDCB01:
            return (7, 6)
        case BoardID.ZVDCB02:
            return (5, 4)
        case _:
            raise ValueError(f"Unknown calibration board ID: {board_id}")
