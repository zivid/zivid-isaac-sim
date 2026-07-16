from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ModelName(str, Enum):
    MR60 = "zivid2PlusMR60"
    MR130 = "zivid2PlusMR130"


@dataclass(frozen=True)
class NoiseProfile:
    distances_mm: NDArray[np.float64]
    lux_values: NDArray[np.float64]
    point_precision_std_mm: NDArray[np.float64]


def _parse_profile_table(table_text: str) -> NoiseProfile:
    rows = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("Profile table must contain a header and at least one data row")

    header_tokens = [token.strip() for token in rows[0].split("\t")]
    if len(header_tokens) < 2 or header_tokens[0] != "d":
        raise ValueError("Header must start with 'd' followed by lux columns")

    lux_values = np.array(
        [float(token.split("=", maxsplit=1)[1].replace(",", ".")) for token in header_tokens[1:]],
        dtype=np.float64,
    )

    distances: list[float] = []
    precision_rows: list[list[float]] = []

    for row in rows[1:]:
        tokens = [token.strip() for token in row.split("\t")]
        if len(tokens) != len(header_tokens):
            raise ValueError(f"Row has {len(tokens)} columns, expected {len(header_tokens)}: {row}")
        distances.append(float(tokens[0].replace(",", ".")))
        precision_rows.append([float(token.replace(",", ".")) for token in tokens[1:]])

    return NoiseProfile(
        distances_mm=np.array(distances, dtype=np.float64),
        lux_values=lux_values,
        point_precision_std_mm=np.array(precision_rows, dtype=np.float64),
    )


_POINT_PRECISION_STD_MM_TABLE_TEXT_MR60 = """d\tlux=0\tlux=250\tlux=500\tlux=750
300\t0,04\t0,045\t0,05\t0,055
400\t0,043\t0,051\t0,058\t0,065
500\t0,053\t0,064\t0,075\t0,089
600\t0,063\t0,083\t0,101\t0,119
700\t0,079\t0,109\t0,138\t0,163
800\t0,099\t0,143\t0,183\t0,216
900\t0,117\t0,187\t0,241\t0,283
1000\t0,142\t0,239\t0,304\t0,385
1100\t0,168\t0,298\t0,4\t0,518
1200\t0,2\t0,366\t0,526\t0,681
"""

_POINT_PRECISION_STD_MM_TABLE_TEXT_MR130 = """d\tlux=0\tlux=250\tlux=500\tlux=750
800\t0,118024192\t0,136033878\t0,155811399\t0,172843467
900\t0,125706846\t0,149426085\t0,1745575\t0,197027911
1000\t0,139059493\t0,173137894\t0,205954173\t0,238657117
1100\t0,157254711\t0,203980847\t0,247356901\t0,29261446
1200\t0,179804595\t0,242946625\t0,300537974\t0,362565458
1300\t0,206617408\t0,291775939\t0,368721718\t0,453683725
1400\t0,242528781\t0,350197792\t0,45093604\t0,561945993
1500\t0,277872419\t0,419129232\t0,549002713\t0,692282712
1600\t0,322058302\t0,502176183\t0,662174416\t0,855239868
1700\t0,365763956\t0,596987903\t0,801643425\t1,042871463
1800\t0,407662243\t0,702323252\t0,962266797\t1,266219079
1900\t0,470933938\t0,82963559\t1,144167531\t1,524643254
2000\t0,526259041\t0,970818871\t1,354550612\t1,812409163
2100\t0,622351474\t1,127821279\t1,593930435\t2,078602266
"""

_POINT_PRECISION_STD_MM_MR60 = _parse_profile_table(_POINT_PRECISION_STD_MM_TABLE_TEXT_MR60)
_POINT_PRECISION_STD_MM_MR130 = _parse_profile_table(_POINT_PRECISION_STD_MM_TABLE_TEXT_MR130)

_POINT_PRECISION_STD_MM_LOOKUP_TABLE = {
    ModelName.MR60: _POINT_PRECISION_STD_MM_MR60,
    ModelName.MR130: _POINT_PRECISION_STD_MM_MR130,
}


def get_supported_lux_range(model_name: ModelName) -> tuple[float, float]:
    if model_name not in _POINT_PRECISION_STD_MM_LOOKUP_TABLE:
        raise ValueError(f"Unsupported camera model: {model_name}")
    profile = _POINT_PRECISION_STD_MM_LOOKUP_TABLE[model_name]
    return float(profile.lux_values[0]), float(profile.lux_values[-1])


def get_point_precision_profile(model_name: ModelName, lux: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(model_name, ModelName):
        raise TypeError(f"Expected ModelName, got {type(model_name).__name__}")

    if model_name not in _POINT_PRECISION_STD_MM_LOOKUP_TABLE:
        raise ValueError(f"Unsupported camera model: {model_name}")
    profile = _POINT_PRECISION_STD_MM_LOOKUP_TABLE[model_name]
    if lux < profile.lux_values[0] or lux > profile.lux_values[-1]:
        raise ValueError(
            f"Lux value {lux} is outside supported range {profile.lux_values[0]}..{profile.lux_values[-1]}"
        )

    point_precision_std_mm = np.array(
        [
            np.interp(lux, profile.lux_values, point_precision_at_distance)
            for point_precision_at_distance in profile.point_precision_std_mm
        ]
    )
    return profile.distances_mm.copy(), point_precision_std_mm


def apply_point_precision_noise(xyz: np.ndarray, model_name: ModelName, lux: float) -> np.ndarray:
    lookup_distances_mm, lookup_sigma_mm = get_point_precision_profile(
        model_name,
        lux,
    )
    noisy_xyz = xyz.copy()
    z_mm = noisy_xyz[:, :, 2] * 1000.0
    valid_mask = np.isfinite(z_mm) & (z_mm != 0.0)
    noise_sigmas = np.interp(z_mm[valid_mask], lookup_distances_mm, lookup_sigma_mm)
    noise = np.random.normal(loc=0.0, scale=noise_sigmas)

    scale = 1 + noise / z_mm[valid_mask]
    noisy_xyz[valid_mask] *= scale[:, None]
    return noisy_xyz
