from enum import IntEnum, auto


class BoardID(IntEnum):
    """Enum for Zivid calibration boards."""

    ZVDCB01 = 0
    """Zivid standard calibration board 7x8-30 mm"""
    ZVDCB02 = auto()
    """Zivid small calibration board 5x6-20 mm"""
