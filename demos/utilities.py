from pathlib import Path

import omni.kit.app
from isaacsim.core.utils.extensions import enable_extension
from omni.ext import ExtensionPathType


def enable_zivid_extension():
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.add_path(
        (Path(__file__).parent.parent / "source" / "isaacsim.zivid").as_posix(), ExtensionPathType.DIRECT_PATH
    )
    if not enable_extension("isaacsim.zivid"):
        raise RuntimeError("Failed to enable isaacsim.zivid extension. Please check the extension path.")
