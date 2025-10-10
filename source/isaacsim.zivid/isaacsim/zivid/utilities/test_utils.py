from dataclasses import asdict
from pathlib import Path

import numpy as np
import omni.kit.test
from isaacsim.zivid.transforms import Transform, rot_diff


class BaseTest(omni.kit.test.AsyncTestCase):

    def assertDataclassAlmostEqual(self, obj1, obj2, places=7):  # pylint: disable=invalid-name
        self.assertEqual(type(obj1), type(obj2), "Objects are of different types")
        dict1 = asdict(obj1)
        dict2 = asdict(obj2)
        self.assertDictAlmostEqual(dict1, dict2, places=places)

    def assertDictAlmostEqual(self, d1, d2, places=7, msg=None):  # pylint: disable=invalid-name
        self.assertEqual(d1.keys(), d2.keys(), msg or "Dictionaries have different keys")
        for key in d1.keys():
            val1 = d1[key]
            val2 = d2[key]
            if isinstance(val1, dict):
                self.assertIsInstance(val2, dict, msg or f"Value for key '{key}' is not a dict in both dictionaries")
                self.assertDictAlmostEqual(val1, val2, places=places, msg=msg)
            else:
                self.assertAlmostEqual(val1, val2, places, msg or f"Value for key '{key}' differs")

    def assertTranformsAlmostEqual(
        self, a: Transform, b: Transform, places: int | None = None
    ):  # pylint: disable=invalid-name
        diff = np.linalg.norm(a.t - b.t)
        self.assertAlmostEqual(diff, 0, msg="Translation part of transform does not match", places=places)
        diff = rot_diff(a.rot, b.rot).mag()
        self.assertAlmostEqual(diff, 0, msg="Rotation part of transform does not match", places=places)

    def assertArraysAlmostEqual(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        places: int | None = None,
        msg: str = "Arrays are not almost equal",
        quantile_thresholds: dict[float, float] | None = None,
    ):  # pylint: disable=invalid-name,too-many-arguments
        """
        Assert arrays are almost equal, with extra quantile-based thresholds.

        Args:
            a, b: Arrays to compare.
            places: Decimal places for absolute tolerance (10^-places).
            msg: Base error message.
            quantile_thresholds: Dict mapping quantile -> max allowed difference.
                                 Example: {50: 1e-8, 100: 1e-6}
        """
        if places is not None:
            atol = 10 ** (-places)

            # First, normal elementwise closeness check

            np.testing.assert_allclose(a, b, atol=atol, err_msg=msg)

        if quantile_thresholds:
            abs_diff = np.abs(np.ravel(a - b))
            for q, threshold in quantile_thresholds.items():
                qval = np.percentile(abs_diff, q)
                assert qval < threshold, f"{msg}: quantile {q}% = {qval:.3e} exceeds threshold {threshold:.3e}"


def get_test_data_dir() -> Path:
    """Get the path to the test data directory."""
    return Path(__file__).parents[5] / "test_data"
