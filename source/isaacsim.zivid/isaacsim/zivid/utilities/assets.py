import pathlib
from dataclasses import dataclass


def get_assets_path() -> pathlib.Path:
    p = pathlib.Path(__file__)
    return p.parent.parent.parent.parent / "assets"


@dataclass(frozen=True)
class AssetPaths:
    franka: pathlib.Path = get_assets_path() / "robots" / "third_party" / "franka.usd"
