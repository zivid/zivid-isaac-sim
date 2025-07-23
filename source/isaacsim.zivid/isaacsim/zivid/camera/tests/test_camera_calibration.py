import omni.kit.test
import omni.usd
from isaacsim.zivid.camera.camera_calibration import CameraCalibration
from isaacsim.zivid.camera.models import ZividCameraModelName
from isaacsim.zivid.camera.resolution import SamplingMode


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the
# root of module will make it auto-discoverable by omni.kit.test
class Test(omni.kit.test.AsyncTestCase):
    # Before running each test
    # Before running each test
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
                    self.assertTrue(lens_model.camera_matrix.as_array().shape == (3, 3))

    async def test_throws_if_invalid_model_name(self):
        with self.assertRaises(ValueError):
            CameraCalibration.from_model_name("InvalidModelName", SamplingMode.FULL)  # type: ignore[reportArgumentType]

    async def test_throws_if_invalid_sampling_mode(self):
        with self.assertRaises(ValueError):
            CameraCalibration.from_model_name(
                ZividCameraModelName.ZIVID_2_PLUS_M130, "InvalidSamplingMode"  # type: ignore[reportArgumentType]
            )
