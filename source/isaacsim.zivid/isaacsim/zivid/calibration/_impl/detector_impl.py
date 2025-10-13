from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np
from isaacsim.zivid.calibration.calibration_board import BoardID
from isaacsim.zivid.calibration.detection_result import DetectionResult, DetectionStatus
from isaacsim.zivid.frame import Frame
from isaacsim.zivid.transforms import Transform, transform_points
from numpy.typing import NDArray


def detect_calibration_board_impl(frame: Frame) -> DetectionResult:
    aruco_ids = _detect_and_filter_aruco_ids(frame.get_data_rgb())
    if len(aruco_ids) == 0:
        return DetectionResult(status=DetectionStatus.NO_VALID_FIDUCIAL_MARKER_DETECTED)
    if len(aruco_ids) > 1:
        return DetectionResult(status=DetectionStatus.MULTIPLE_VALID_FIDUCIAL_MARKERS_DETECTED)

    aruco_id = aruco_ids[0]
    board_id = _classify_by_aruco_id(aruco_id)
    assert board_id is not None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checker_shape = _get_board_shape(board_id)
    gray = cv2.cvtColor(frame.get_data_rgb(), cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCornersSB(gray, checker_shape, cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        return DetectionResult(status=DetectionStatus.BOARD_DETECTION_FAILED)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    feature_points_2d = _reshape_cv_corners(corners, checker_shape)
    feature_points_3d = _get_feature_points_3d(feature_points_2d, frame.get_data_xyz())
    if feature_points_3d is None:
        return DetectionResult(status=DetectionStatus.INSUFFICIENT_3D_QUALITY)
    return DetectionResult(
        status=DetectionStatus.OK,
        board_id=board_id,
        feature_points_2d=feature_points_2d,
        feature_points_3d=feature_points_3d,
    )


def draw_corners_impl(frame: Frame, detection_result: DetectionResult) -> NDArray:
    """Draw the detected corners on the image."""
    if not detection_result.valid():
        raise ValueError("Detection result is not valid.")
    shape = _get_board_shape(detection_result.board_id)
    corners = detection_result.feature_points_2d.reshape(-1, 1, 2)
    im = cv2.drawChessboardCorners(cv2.cvtColor(frame.get_data_rgb(), cv2.COLOR_RGB2BGR), shape, corners, True)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def get_ground_truth_feature_points_3d_impl(board_id: BoardID, board_pose: Transform) -> NDArray:
    checker_size = _get_checker_size(board_id)
    board_shape = _get_board_shape(board_id)
    cols, rows = board_shape
    points_3d_board_frame = np.zeros((rows, cols, 3), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    points_3d_board_frame[..., 0] = xx * checker_size
    points_3d_board_frame[..., 1] = yy * checker_size
    return transform_points(points_3d_board_frame, board_pose)


def _detect_and_filter_aruco_ids(image: NDArray) -> list[int]:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect the markers
    _, all_ids, _ = aruco.ArucoDetector(aruco_dict, parameters).detectMarkers(gray)

    if all_ids is None:
        return []

    valid_ids = _get_valid_aruco_ids(max_id=250)
    ids = [idx for idx in all_ids if idx in valid_ids]

    return ids


def _classify_by_aruco_id(aruco_id: int) -> BoardID | None:

    match aruco_id:
        case 1:
            return BoardID.ZVDCB01
        case 239:
            return BoardID.ZVDCB02
        case _:
            return None


def _get_valid_aruco_ids(max_id: int = 250) -> list[int]:
    valid_ids = []
    for idx in range(max_id):
        if _classify_by_aruco_id(idx) is not None:
            valid_ids.append(idx)

    return valid_ids


def _get_board_shape(board_id: BoardID) -> tuple[int, int]:
    match board_id:
        case BoardID.ZVDCB01:
            return (7, 6)
        case BoardID.ZVDCB02:
            return (5, 4)
        case _:
            raise ValueError(f"Unknown calibration board ID: {board_id}")


def _get_checker_size(board_id: BoardID) -> float:
    match board_id:
        case BoardID.ZVDCB01:
            return 30 * 1e-3
        case BoardID.ZVDCB02:
            return 20 * 1e-3
        case _:
            raise ValueError(f"Unknown calibration board ID: {board_id}")


def _reshape_cv_corners(corners: NDArray, board_shape: tuple[int, int]) -> NDArray:
    corners = corners.reshape(-1, 2)
    corners = np.flip(corners, axis=0)  # open cv gives corners in opposite order
    corners = corners.reshape(board_shape[1], board_shape[0], 2)
    return corners


def _get_feature_points_3d(feature_points_2d: NDArray, xyz: NDArray) -> NDArray | None:

    feature_points_2d_upper = np.ceil(feature_points_2d).astype(int)
    feature_points_2d_lower = np.floor(feature_points_2d).astype(int)

    alpha = feature_points_2d - feature_points_2d_lower

    dx = alpha[..., [1]]
    dy = alpha[..., [0]]

    points_3d_00 = xyz[feature_points_2d_lower[..., 1], feature_points_2d_lower[..., 0]]
    points_3d_01 = xyz[feature_points_2d_upper[..., 1], feature_points_2d_lower[..., 0]]
    points_3d_10 = xyz[feature_points_2d_lower[..., 1], feature_points_2d_upper[..., 0]]
    points_3d_11 = xyz[feature_points_2d_upper[..., 1], feature_points_2d_upper[..., 0]]

    points_3d_0 = points_3d_00 * (1 - dy) + points_3d_01 * dy
    points_3d_1 = points_3d_10 * (1 - dy) + points_3d_11 * dy
    feature_points_3d = points_3d_0 * (1 - dx) + points_3d_1 * dx

    if np.any(np.isnan(feature_points_3d)) or np.any(np.isinf(feature_points_3d)):
        return None

    return feature_points_3d
