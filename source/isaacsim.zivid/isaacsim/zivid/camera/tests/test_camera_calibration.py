from isaacsim.zivid.camera._impl.camera_calibration import CameraCalibration
from isaacsim.zivid.camera._impl.resolutions import get_camera_resolution
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.sampling import SamplingMode
from isaacsim.zivid.utilities.test_utils import BaseTest


class Test(BaseTest):
    async def setUp(self):  # pylint: disable=invalid-name
        return

    # After running each test
    async def tearDown(self):  # pylint: disable=invalid-name
        return

    async def test_all_models(self):
        for model_name in ZividCameraModelName:
            for sampling_mode in SamplingMode:
                calib = CameraCalibration.from_model_name(model_name, sampling_mode)
                for lens_model in [calib.camera_model, calib.projector_model]:
                    self.assertEqual(lens_model.camera_matrix.to_array().shape, (3, 3))
                    self.assertEqual(lens_model.distortion_coefficients.to_array().shape, (12,))
                self.assertEqual(
                    calib.camera_model.resolution,
                    get_camera_resolution(sampling_mode, model_name),
                    "camera resolution mismatch",
                )

    async def test_throws_if_invalid_model_name(self):
        with self.assertRaises(ValueError):
            CameraCalibration.from_model_name("InvalidModelName", SamplingMode.FULL)  # type: ignore[reportArgumentType]

    async def test_throws_if_invalid_sampling_mode(self):
        with self.assertRaises(ValueError):
            CameraCalibration.from_model_name(
                ZividCameraModelName.ZIVID_2_PLUS_M130, "InvalidSamplingMode"  # type: ignore[reportArgumentType]
            )
