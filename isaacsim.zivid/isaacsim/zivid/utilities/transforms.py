from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

import numpy as np

from isaacsim.core.utils.rotations import rot_matrix_to_quat, quat_to_rot_matrix

import isaacsim.zivid.utilities.arrays as array_utils


class Rotation:
    def __init__(self):
        self._matrix_representation = np.eye(3)
        self._quat_representation = rot_matrix_to_quat(self._matrix_representation)

    @classmethod
    def from_matrix(cls, matrix: NDArray) -> Rotation:
        rotation = cls()
        rotation._matrix_representation = matrix
        rotation._quat_representation = rot_matrix_to_quat(matrix)
        return rotation

    @classmethod
    def from_quat(cls, quat: NDArray) -> Rotation:
        rotation = cls()
        rotation._matrix_representation = quat_to_rot_matrix(quat)
        rotation._quat_representation = quat
        return rotation

    @classmethod
    def from_axis_angle(cls, aa: NDArray) -> Rotation:
        angle = np.linalg.norm(aa)
        axis = aa / angle if angle != 0 else np.zeros(3)
        s = np.sin(angle / 2)
        quat = np.array([np.cos(angle / 2), s * axis[0], s * axis[1], s * axis[2]])
        return cls.from_quat(quat)

    @classmethod
    def identity(cls) -> Rotation:
        """Returns the identity rotation."""
        return cls.from_matrix(np.eye(3))

    @classmethod
    def y_down_z_forward(cls) -> Rotation:
        """Get the rotation matrix for Y down Z forward."""
        return cls.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))

    def to_matrix(self) -> NDArray:
        return self._matrix_representation

    def to_quat(self) -> NDArray:
        return self._quat_representation

    def inverse(self) -> Rotation:
        """Returns the inverse of the rotation."""
        return Rotation.from_matrix(self._matrix_representation.T)

    def __mul__(self, other: Rotation) -> Rotation:
        return Rotation.from_matrix(self.to_matrix() @ other.to_matrix())


@dataclass
class Transform:
    t: NDArray
    rot: Rotation

    def to_isaac_sim(self) -> tuple[array_utils.TensorData, array_utils.TensorData]:
        """Returns the transformation in Isaac Sim format."""
        return (self.t, self.rot.to_quat())

    def as_matrix(self) -> NDArray:
        """Returns the transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rot.to_matrix()
        matrix[:3, 3] = self.t
        return matrix

    def __mul__(self, other: Transform) -> Transform:
        """Returns the composition of two transformations."""
        new_transform = Transform(
            t=self.t + self.rot.to_matrix() @ other.t,
            rot=self.rot * other.rot,
        )
        return new_transform

    def inverse(self) -> Transform:
        """Returns the inverse of the transformation."""
        return Transform(
            t=self.rot.inverse().to_matrix() @ (-self.t),
            rot=self.rot.inverse(),
        )

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


def transform_points(xyz: NDArray, transfrom: Transform) -> NDArray:
    t_shape = tuple([1 for _ in xyz.shape[:-1]]) + tuple([3])
    return xyz @ transfrom.rot.to_matrix().T + transfrom.t.reshape(t_shape)
