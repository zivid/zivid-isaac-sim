from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import isaacsim.zivid.utilities.arrays as array_utils
import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from numpy.typing import NDArray


class Rotation:
    """
    A class representing a 3D rotation.
    """

    def __init__(self):
        self._quat_representation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @classmethod
    def from_matrix(cls, matrix: NDArray) -> Rotation:
        """
        Create a Rotation from a rotation matrix.

        Args:
            matrix(NDArray): A 3x3 rotation matrix.

        Returns:
            Rotation: The corresponding rotation.
        """
        if matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        if not np.allclose(matrix.T @ matrix, np.eye(3)):
            raise ValueError("Rotation matrix must be orthonormal")

        rotation = cls()
        rotation._quat_representation = rot_matrix_to_quat(matrix)
        return rotation

    @classmethod
    def from_quat(cls, quat: NDArray) -> Rotation:
        """
        Create a Rotation from a quaternion.

        Args:
            quat(NDArray): A quaternion in the format [w, x, y, z].

        Returns:
            Rotation: The corresponding rotation.
        """
        if quat.shape != (4,):
            raise ValueError("Quaternion must be of shape (4,)")
        if not np.isclose(np.linalg.norm(quat), 1.0):
            raise ValueError("Quaternion must be unit norm")

        rotation = cls()
        rotation._quat_representation = quat
        return rotation

    @classmethod
    def from_axis_angle(cls, aa: NDArray) -> Rotation:
        """
        Create a Rotation from an axis-angle representation.

        Args:
            aa(NDArray): A 3D vector representing the axis of rotation multiplied by the angle in radians.

        Returns:
            Rotation: The corresponding rotation.
        """
        if aa.shape != (3,):
            raise ValueError("Axis-angle must be of shape (3,)")

        angle = np.linalg.norm(aa)
        axis = aa / angle if abs(angle) > 1e-6 else np.zeros(3)
        s = np.sin(angle / 2)
        quat = np.array([np.cos(angle / 2), s * axis[0], s * axis[1], s * axis[2]])
        return cls.from_quat(quat)

    @classmethod
    def from_euler(cls, euler: NDArray, seq: str = "xyz", degrees: bool = False) -> Rotation:
        """
        Create a Rotation from Euler angles.

        Args:
            euler(NDArray): A 3D vector representing the Euler angles.
            seq(str): A string representing the order of rotations. Default is "xyz".
            degrees(bool): Whether the input angles are in degrees. Default is False (radians).

        Returns:
            Rotation: The corresponding rotation.
        """

        rot = cls.identity()
        if degrees:
            euler = euler * np.pi / 180.0

        for axis, a in zip(seq, euler, strict=False):
            aa = np.array([a if ax == axis else 0.0 for ax in "xyz"])
            rot = rot * cls.from_axis_angle(aa)
        return rot

    @classmethod
    def identity(cls) -> Rotation:
        """
        Returns the identity rotation.
        Returns:
            Rotation: The identity rotation.
        """
        return cls.from_matrix(np.eye(3))

    @classmethod
    def y_down_z_forward(cls) -> Rotation:
        """
        Get the rotation matrix for Y down Z forward,
        typically used for camera coordinate systems.
        Returns:
            Rotation: The corresponding rotation.
        """
        return cls.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))

    def to_matrix(self) -> NDArray:
        """
        Converts the rotation to a rotation matrix.
        Returns:
            NDArray: A 3x3 rotation matrix."""
        return quat_to_rot_matrix(self._quat_representation)

    def to_quat(self) -> NDArray:
        """
        Converts the rotation to a quaternion.
        Returns:
            NDArray: A quaternion in the format [w, x, y, z]."""
        return self._quat_representation.copy()

    def as_quat(self) -> NDArray:
        """
        Return a view of the rotation as a quaternion.
        Returns:
            NDArray: A quaternion in the format [w, x, y, z].
        """
        return self._quat_representation

    def to_axis_angle(self, eps=1e-6) -> NDArray:
        """
        Converts the rotation to an axis-angle representation.
        Args:
            eps(float): A small value to avoid division by zero. Default is 1e-6.
        Returns:
            NDArray: A 3D vector representing the axis of rotation multiplied by the angle in radians.
        """
        quat = self._quat_representation * (1.0 - 2.0 * (self._quat_representation[0] < 0.0))
        mag = np.linalg.norm(quat[1:])
        half_angle = np.arctan2(mag, quat[0])
        angle = 2.0 * half_angle
        # check whether to apply Taylor approximation
        sin_half_angles_over_angles = np.sin(half_angle) / angle if abs(angle) > eps else 0.5 - angle * angle / 48
        return quat[1:] / sin_half_angles_over_angles

    def mag(self) -> float:
        """Returns the magnitude of the rotation in radians."""
        return float(np.linalg.norm(self.to_axis_angle()))

    def inverse(self) -> Rotation:
        """Returns the inverse of the rotation."""
        return Rotation.from_quat(self._quat_representation * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32))

    @overload
    def __mul__(self, other: Rotation) -> Rotation:
        pass

    @overload
    def __mul__(self, other: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    def __mul__(self, other):
        if isinstance(other, Rotation):
            return Rotation.from_matrix(self.to_matrix() @ other.to_matrix())
        if isinstance(other, np.ndarray):
            if other.shape != (3,):
                raise ValueError("Vector must be shape (3,)")
            return self.to_matrix() @ other
        raise TypeError(f"Unsupported type {type(other)}")


@dataclass
class Transform:
    """A class representing a 3D transformation (rotation and translation)."""

    t: NDArray
    rot: Rotation

    @classmethod
    def identity(cls) -> Transform:
        """Returns the identity transformation."""
        return cls(t=np.zeros(3), rot=Rotation.identity())

    @classmethod
    def from_isaac_sim(cls, pose: tuple[array_utils.TensorData, array_utils.TensorData]) -> Transform:
        """Creates a Transform from an Isaac Sim pose."""
        translation = array_utils.convert_to_numpy(pose[0])
        orientation = array_utils.convert_to_numpy(pose[1])
        return cls(t=translation, rot=Rotation.from_quat(orientation))

    def to_isaac_sim(self) -> tuple[array_utils.TensorData, array_utils.TensorData]:
        """
        Convert the transformation to Isaac Sim format.
        Returns:
            tuple[array_utils.TensorData, array_utils.TensorData]: A tuple containing the translation and rotation
        """
        return (self.t, self.rot.as_quat())

    def to_matrix(self) -> NDArray:
        """
        Converts the transformation to a 4x4 matrix.
        Returns:
            NDArray: A 4x4 transformation matrix.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rot.to_matrix()
        matrix[:3, 3] = self.t
        return matrix

    def inverse(self) -> Transform:
        """Returns the inverse of the transformation."""
        rot_inv = self.rot.inverse()
        return Transform(
            t=rot_inv * (-self.t),
            rot=rot_inv,
        )

    @overload
    def __mul__(self, other: Transform) -> Transform:
        pass

    @overload
    def __mul__(self, other: NDArray) -> NDArray:
        pass

    def __mul__(self, other):
        if isinstance(other, Transform):
            return Transform(
                t=self.t + self.rot.to_matrix() @ other.t,
                rot=self.rot * other.rot,
            )
        if isinstance(other, np.ndarray):
            if other.shape != (3,):
                raise ValueError("Can only multiply Transform with 3D vector")
            return self.t + self.rot * other

        raise NotImplementedError(f"Multiplication with {type(other)} not implemented")


def transform_points(xyz: NDArray, transform: Transform) -> NDArray:
    """
    Transform a set of 3D points using the given transformation.

    Args:
        xyz (NDArray): A numpy array of shape (..., 3) representing the 3D points.
        transform: The transformation to apply.

    Returns:
        NDArray: The transformed 3D points.
    """
    t_shape = tuple(1 for _ in xyz.shape[:-1]) + tuple([3])
    return xyz @ transform.rot.to_matrix().T + transform.t.reshape(t_shape)


def rot_diff(a: Rotation, b: Rotation) -> Rotation:
    """
    Compute the difference between two rotations.

    Defined as: a * diff = b  =>  diff = a^-1 * b.

    Args:
        a (Rotation): The first rotation.
        b (Rotation): The second rotation.

    Returns:
        Rotation: The rotation that, when applied to `a`, gives `b`.
    """
    return a.inverse() * b
