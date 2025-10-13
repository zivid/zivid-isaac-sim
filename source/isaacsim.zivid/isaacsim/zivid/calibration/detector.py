from __future__ import annotations

from isaacsim.zivid.calibration.calibration_board import BoardID
from isaacsim.zivid.calibration.detection_result import DetectionResult
from isaacsim.zivid.frame import Frame
from isaacsim.zivid.transforms import Transform
from numpy.typing import NDArray

from ._impl import detector_impl as impl


def detect_calibration_board(frame: Frame) -> DetectionResult:
    return impl.detect_calibration_board_impl(frame)


def draw_corners(frame: Frame, detection_result: DetectionResult) -> NDArray:
    return impl.draw_corners_impl(frame, detection_result)


def get_ground_truth_feature_points_3d(board_id: BoardID, board_pose: Transform) -> NDArray:
    return impl.get_ground_truth_feature_points_3d_impl(board_id, board_pose)
