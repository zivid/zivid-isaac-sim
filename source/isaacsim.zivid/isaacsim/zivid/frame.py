from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from isaacsim.zivid.resolution import Resolution
from numpy.typing import NDArray


class Frame:
    def __init__(self, rgb: NDArray, xyz: NDArray):
        self._rgb = rgb
        self._xyz = xyz
        if rgb.shape[:2] != xyz.shape[:2]:
            raise ValueError(f"RGB and XYZ images must have the same height and width, got {rgb.shape} and {xyz.shape}")
        if rgb.shape[2] != 3 or xyz.shape[2] != 3:
            raise ValueError(f"RGB and XYZ images must have 3 channels, got {rgb.shape} and {xyz.shape}")

        self._resolution = Resolution(height=rgb.shape[0], width=rgb.shape[1])

    def get_data_rgb(self, clone=False) -> NDArray:
        """
        Get the RGB image. Optionally provides a clone of the data.

        Args:
            clone (bool): If True, returns a copy of the data. Default is False.

        Returns:
            NDArray: The RGB image.
        """
        if clone:
            return self._rgb.copy()
        return self._rgb

    def get_data_xyz(self, clone=False) -> NDArray:
        """
        Get the XYZ image. Optionally provides a clone of the data.

        Args:
            clone (bool): If True, returns a copy of the data. Default is False.

        Returns:
            NDArray: The XYZ image.
        """
        if clone:
            return self._xyz.copy()
        return self._xyz

    def save_ply(self, filename: str | Path) -> None:
        """
        Save the point cloud to a file.

        Args:
            filename (str): The filename to save the point cloud to.
        """
        filename = Path(filename)

        if filename.suffix.lower() != ".ply":
            raise ValueError(f"Filename must have a .ply extension, got {filename.suffix}")

        points = self.get_data_xyz(clone=True).reshape(-1, 3)
        colors = self.get_data_rgb(clone=True).reshape(-1, 3) / 255.0  # Normalize colors to [0, 1]
        mask = np.isfinite(points).all(axis=1)  # Keep only valid points
        points = points[mask]
        colors = colors[mask]
        point_cloud = trimesh.PointCloud(points, colors=colors)
        point_cloud.export(filename)

    @property
    def resolution(self) -> Resolution:
        """The resolution of the frame."""
        return self._resolution
