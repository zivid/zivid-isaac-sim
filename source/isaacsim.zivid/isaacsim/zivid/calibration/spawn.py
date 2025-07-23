from pathlib import Path

from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.zivid.calibration.calibration_board import BoardID
from isaacsim.zivid.utilities.assets import get_assets_path
from isaacsim.zivid.utilities.transforms import Transform


def spawn_calibration_board(board_id: BoardID, prim_path: str, world_pose: Transform) -> SingleXFormPrim:
    usd_file = _get_usd_file(board_id)
    add_reference_to_stage(str(usd_file), prim_path)
    board_prim = SingleXFormPrim(prim_path=prim_path, name=board_id.name)
    board_prim.set_world_pose(*world_pose.to_isaac_sim())
    return board_prim


def _get_usd_file(board_id: BoardID) -> Path:
    assets_path = get_assets_path()
    match board_id:
        case BoardID.ZVDCB01:
            return assets_path / "cb01.usd"
        case BoardID.ZVDCB02:
            return assets_path / "cb02.usd"
        case _:
            raise ValueError(f"Unknown calibration board board_id: {board_id}")
