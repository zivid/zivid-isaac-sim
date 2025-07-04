import pathlib

def get_assets_path() -> pathlib.Path:
    p = pathlib.Path(__file__)
    return p.parent.parent.parent.parent / "assets"
