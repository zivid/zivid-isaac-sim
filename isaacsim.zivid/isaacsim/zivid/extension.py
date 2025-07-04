# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import omni
import omni.ext
import omni.kit.app
import omni.kit.widget
import omni.kit.menu.utils as menu_utils
import omni.usd
import pxr.Usd

from typing import Optional, Tuple
from .gui import CameraWidget, NewCameraWindow

import numpy as np

from .utilities.transforms import Transform, Rotation
from .assembler import assemble, mounts, calibration_board
from .cameras.zivid_camera import ZividCamera
from .cameras.models import ZividCameraModelName
from isaacsim.core.prims import SingleRigidPrim


def get_applicable_parent(rigid: pxr.Usd.Prim) -> Optional[pxr.Usd.Prim]:
    parent = rigid.GetParent()
    while parent.IsValid():
        if parent.HasAttribute("physxArticulation:articulationEnabled"):
            return parent
        parent = parent.GetParent()

    return None


class ZividExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self._ext_id = None
        self._usd_context = omni.usd.get_context()
        self._create_menu_extension = None
        self._action_handles = {}
        self._context_menu = None
        self._cameras = []
        self._camera_menu_items = []
        self._location = None
        self._icon_path = None
        self._new_camera_window = None
        self._camera_widget = None

    def on_startup(self, ext_id):
        self._ext_id = ext_id

        self._location = omni.kit.app.get_app_interface().get_extension_manager().get_extension_path(ext_id)
        self._icon_path = self._location + "/assets/ZividStudio.png"

        self._new_camera_window = NewCameraWindow(self._create_camera_prim, get_applicable_parent)
        self._camera_widget = CameraWidget()

        self._action_handles["open_new_camera_window"] = omni.kit.actions.core.get_action_registry().register_action(
            self._ext_id,
            "open_new_camera_window",
            self._open_new_camera_window,
            display_name="Add Zivid Camera",
            description="Add Zivid Camera to scene",
        )

        self._action_handles["open_camera_widget"] = omni.kit.actions.core.get_action_registry().register_action(
            self._ext_id,
            "open_camera_widget",
            self._camera_widget.open,
            display_name="Interact with camera",
            description="Interact with camera",
        )

        self._action_handles["add_calibration_board"] = omni.kit.actions.core.get_action_registry().register_action(
            self._ext_id,
            "add_calibration_board",
            self._add_calibration_board_to_stage,
            display_name="Add Zivid Calibration Board to stage",
            description="Add Zivid Calibration Board to stage",
        )

        self._menu_list = [
            menu_utils.MenuItemDescription(
                name="Zivid Camera",
                appear_after="Camera",
                onclick_action=(self._ext_id, "open_new_camera_window"),
                glyph=self._location + "/assets/ZividStudio.png",
            ),
            menu_utils.MenuItemDescription(
                name="Zivid Calibration board",
                appear_after="Zivid Camera",
                sub_menu=[
                    menu_utils.MenuItemDescription(
                        name="CB01",
                        onclick_action=(self._ext_id, "add_calibration_board", calibration_board.BoardID.ZVDCB01),
                        glyph=self._location + "/assets/ZividStudio.png",
                    ),
                    menu_utils.MenuItemDescription(
                        name="CB02",
                        onclick_action=(self._ext_id, "add_calibration_board", calibration_board.BoardID.ZVDCB02),
                        glyph=self._location + "/assets/ZividStudio.png",
                    ),
                ],
                glyph=self._location + "/assets/ZividStudio.png",
            ),
        ]

        omni.kit.menu.utils.add_menu_items(self._menu_list, "Create")

        self._cameras = []

        self._context_menu = omni.kit.widget.context_menu.add_menu(
            {
                "glyph": self._icon_path,
                "name": "Zivid Camera",
                "onclick_fn": self._open_new_camera_window_from_context,
                "enabled_fn": self._has_applicable_selection,
            },
            "CREATE",
        )

        events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.OPENED):
            for prim in self._usd_context.get_stage().Traverse():
                if prim.GetAttribute("is_zivid_camera"):
                    model = ZividCameraModelName(prim.GetAttribute("zivid_camera_model").Get())
                    self._create_camera(str(prim.GetPath()), model)

    def _create_camera_prim(self, mount_prim, manipulator_prim, camera, mount=mounts.Mount.OA111):
        mount_local_pose = Transform(
            rot=Rotation.from_matrix(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])), t=np.array([0.0, 0.0, 0.0])
        )

        if not manipulator_prim:
            # No mount was selected. Attach camera directly onto robot or default prim

            if not mount_prim:
                mount_prim = self._usd_context.get_stage().GetDefaultPrim().GetPath()
                world_pose = Transform.identity()
            else:
                world_pose = Transform.from_isaac_sim(SingleRigidPrim(mount_prim).get_world_pose())

            zivid_prim_path = assemble.assemble_zivid(
                model_name=camera,
                prim_path=str(mount_prim) + "/Zivid",
                world_pose=world_pose,
                make_rigid_body=True,
            )
        else:
            zivid_prim_path = assemble.assemble_zivid_on_robot(
                model_name=camera,
                attach_prim_path=mount_prim,
                robot_prim_path=manipulator_prim,
                mount=mount,
                mount_local_pose=mount_local_pose,
            )

        self._create_camera(zivid_prim_path, camera)

    def _add_calibration_board_to_stage(self, board_id):
        p = self._usd_context.get_stage().GetDefaultPrim().GetPath()

        # Rotate to make the board flat on the ground by default
        board_pose = Transform(t=np.array([0.0, 0.0, 0.1]), rot=Rotation.from_axis_angle(np.array([np.pi / 2, 0, 0.0])))

        calibration_board.add_calibration_board_to_stage(
            id=board_id,
            prim_path=f"{p}/{board_id.name}",
            world_pose=board_pose,
        )

    def _create_camera(self, prim_path: str, model_name: ZividCameraModelName):
        c = ZividCamera(model_name=model_name, prim_path=prim_path)
        c.initialize()
        self._cameras.append(c)
        self._refresh_menu(c)

    def _has_applicable_selection(self, _context):
        return self._get_applicable_prims() is not None

    def _get_applicable_prims(self) -> Optional[Tuple[str, str]]:
        selection = self._usd_context.get_selection().get_selected_prim_paths()
        if len(selection) != 1:
            return None

        prim: pxr.Usd.Prim = self._usd_context.get_stage().GetPrimAtPath(selection[0])

        if not prim.HasAttribute("physics:rigidBodyEnabled"):
            return None

        parent_path = get_applicable_parent(prim)

        if parent_path:
            return (prim.GetPath(), parent_path)

        return None

    def _refresh_menu(self, new_camera: ZividCamera):
        if len(self._camera_menu_items):
            menu_utils.remove_menu_items(self._camera_menu_items, "Zivid Cameras")

        self._camera_menu_items.append(
            menu_utils.MenuItemDescription(
                name=f"Camera {new_camera.prim_path}", onclick_action=(self._ext_id, "open_camera_widget", new_camera)
            )
        )
        menu_utils.add_menu_items(self._camera_menu_items, "Zivid Cameras")

    def _open_new_camera_window(self):
        self._new_camera_window.open(free_standing=True)

    def _open_new_camera_window_from_context(self, _context):
        self._new_camera_window.open(free_standing=False)

    def on_shutdown(self):
        if self._menu_list is not None:
            menu_utils.remove_menu_items(self._menu_list, "Create")
            self._menu_list = None

        for action in self._action_handles.values():
            a = omni.kit.actions.core.get_action_registry()
            a.deregister_action(action)
            self._action_handle = None
        self._action_handles.clear()

        if self._context_menu is not None:
            self._context_menu.release()
            self._context_menu = None

        if self._usd_context is not None:
            self._usd_context = None

        if self._camera_menu_items is not None:
            menu_utils.remove_menu_items(self._camera_menu_items, "Zivid Cameras")
            self._camera_menu_items = None

        if self._new_camera_window is not None:
            self._new_camera_window = None

        if self._camera_widget is not None:
            self._camera_widget = None
