from enum import IntEnum, auto

from isaacsim.zivid.calibration.calibration_board import BoardID
from numpy.typing import NDArray


class DetectionStatus(IntEnum):
    OK = 0
    NO_VALID_FIDUCIAL_MARKER_DETECTED = auto()
    MULTIPLE_VALID_FIDUCIAL_MARKERS_DETECTED = auto()
    BOARD_DETECTION_FAILED = auto()
    INSUFFICIENT_3D_QUALITY = auto()


class DetectionResult:
    def __init__(
        self,
        status: DetectionStatus,
        board_id: BoardID | None = None,
        feature_points_2d: NDArray | None = None,
        feature_points_3d: NDArray | None = None,
    ) -> None:
        self._board_id = board_id
        self._feature_points_2d = feature_points_2d
        self._feature_points_3d = feature_points_3d
        self._status = status

    def _raise_if_invalid(self, msg: str) -> None:
        if not self.valid():
            raise RuntimeError(msg + f" detection result is not valid, status: {self._status.name}")

    @property
    def board_id(self) -> BoardID:
        self._raise_if_invalid("Cannot get board ID")
        assert self._board_id is not None
        return self._board_id

    @property
    def feature_points_2d(self) -> NDArray:
        self._raise_if_invalid("Cannot get 2D feature points")
        assert self._feature_points_2d is not None
        return self._feature_points_2d

    @property
    def feature_points_3d(self) -> NDArray:
        self._raise_if_invalid("Cannot get 3D feature points")
        assert self._feature_points_3d is not None
        return self._feature_points_3d

    @property
    def status(self) -> DetectionStatus:
        return self._status

    def valid(self) -> bool:
        return self._status == DetectionStatus.OK
