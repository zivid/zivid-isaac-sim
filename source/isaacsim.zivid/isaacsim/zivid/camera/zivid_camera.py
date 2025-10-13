from __future__ import annotations

from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.sampling import SamplingMode
from isaacsim.zivid.frame import Frame
from isaacsim.zivid.resolution import Resolution
from isaacsim.zivid.transforms import Transform
from numpy.typing import NDArray

from ._impl.zivid_camera_impl import ZividCameraImpl


class ZividCamera:
    """
    Class representing a Zivid camera in Isaac Sim.

    This class provides the main interface for a simulated Zivid camera in Isaac Sim.
    """

    def __init__(
        self, model_name: ZividCameraModelName, prim_path: str, sampling_mode: SamplingMode = SamplingMode.FULL
    ):
        self.__impl = ZividCameraImpl(model_name, prim_path, sampling_mode)

    def __del__(self):
        """Clean up resources by calling destroy."""
        self.destroy()

    def destroy(self) -> None:
        """Destroy the zivid camera by detaching all related annotators and render products."""
        self.__impl.destroy()

    def initialize(self, physics_sim_view=None) -> None:
        """
        To be called before using this class after a reset of the world.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): Current physics simulation view.
                Defaults to None.
        """
        self.__impl.initialize(physics_sim_view)

    def set_world_pose(self, pose: Transform) -> None:
        """
        Set the world pose of the Zivid camera.

        Args:
            pose (Transform): The desired world pose of the Zivid camera.
        """
        self.__impl.set_world_pose(pose)

    def set_local_pose(self, pose: Transform) -> None:
        """
        Set the local pose of the Zivid camera, relative to the prims parent.

        Args:
            pose (Transform): The desired local pose of the Zivid camera.
        """
        self.__impl.set_local_pose(pose)

    def set_camera_model(self, model_name: ZividCameraModelName) -> None:
        """
        Set the camera model of the Zivid camera and configure the sensors accordingly.

        Args:
            model_name (ZividCameraModelName): The camera model to set.
        """
        self.__impl.set_camera_model(model_name)

    def set_sampling_mode(self, sampling_mode: SamplingMode) -> None:
        """Set the sampling mode of the Zivid camera and configure the sensors accordingly.

        Args:
            sampling_mode (SamplingMode): The sampling mode to set.
        """
        self.__impl.set_sampling_mode(sampling_mode)

    def set_lens_aperture(self, fnumber: float) -> None:
        """
        Set the lens aperture of the camera. The provided f-number will be rounded to the closest valid aperture. See:
        https://support.zivid.com/en/latest/reference-articles/settings/2d-settings/2d-acquisition-settings/
        aperture.html for details on valid apertures.

        Args:
            fnumber (float): The desired f-number to set.
        """
        self.__impl.set_lens_aperture(fnumber)

    def get_camera_sensor_prim_path(self) -> str:
        """
        Get the prim path of the Camera object simulating the camera sensor.

        Returns:
            str: The prim path of the camera sensor.
        """
        return self.__impl.get_camera_sensor_prim_path()

    def get_projector_prim_path(self) -> str:
        """
        Get the prim path of the Camera object simulating the projector.

        Returns:
            str: The prim path of the projector.
        """

        return self.__impl.get_projector_prim_path()

    def get_frame(self) -> Frame:
        """
        Get the current frame from the camera, containing both RGB and XYZ data.

        Returns:
            Frame: The current frame from the camera.
        """
        return self.__impl.get_frame()

    def get_camera_resolution(self) -> Resolution:
        """
        Get the resolution of the camera based on the current model and sampling mode.

        Returns:
            Resolution: The resolution of the camera.
        """
        return self.__impl.get_camera_resolution()

    def get_world_pose(self) -> Transform:
        """
        Get the transform from world to of the cameras reference frame.

        Returns:
            Transform: The world pose of the Zivid Camera.
        """
        return self.__impl.get_world_pose()

    def get_local_pose(self) -> Transform:
        """
        Get the transform from the parent prim to the Zivid cameras reference frame.

        Returns:
            Transform: The local pose of the Zivid camera.
        """
        return self.__impl.get_local_pose()

    def get_camera_sensor_world_pose(self) -> Transform:
        """
        Get the transform from world to the camera sensor frame.

        Returns:
            Transform: The world pose of the camera sensor.
        """
        return self.__impl.get_camera_sensor_world_pose()

    def get_projector_world_pose(self) -> Transform:
        """Get the transform from world to the projector frame.

        Returns:
            Transform: The world pose of the projector.
        """
        return self.__impl.get_projector_world_pose()

    def get_focus_distance(self) -> float:
        """
        Get the focus distance of the camera in meters.

        Returns:
            float: The focus distance of the camera in meters.
        """
        return self.__impl.get_focus_distance()

    def project_points(self, points: NDArray) -> NDArray:
        """
        Project 3D points in the camera frame to 2D pixel coordinates.

        Args:
            points (NDArray): A HxWx3 or Nx3 array of 3D points in the camera frame.

        Returns:
            NDArray: A HxWx2 or Nx2 array of 2D pixel coordinates.
        """
        return self.__impl.project_points(points)

    @property
    def prim_path(self) -> str:
        """
        Get the prim path of the Zivid camera.

        Returns:
            str: The prim path of the Zivid camera.
        """
        return self.__impl.prim_path

    @property
    def model_name(self) -> ZividCameraModelName:
        """
        Get the model name of the Zivid camera.
        Returns:
            ZividCameraModelName: The model name of the Zivid camera.
        """
        return self.__impl.model_name
