from __future__ import annotations

from enum import IntEnum


class SamplingMode(IntEnum):
    """
    Enum for Zivid camera sampling modes.
    """

    FULL = 0
    "No down sampling, full resolution."
    DOWNSAMPLE2X2 = 1
    "Down sampling by a factor of 2 in both dimensions."
    DOWNSAMPLE4X4 = 2
    "Down sampling by a factor of 4 in both dimensions."
