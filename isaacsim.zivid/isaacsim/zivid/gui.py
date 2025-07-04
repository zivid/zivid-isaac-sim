import omni.ui
import asyncio

from typing import Optional, Tuple, Callable
from isaacsim.gui.components.element_wrappers import (
    Button,
    CheckBox,
    DropDown,
    IntField,
    StringField,
    ScrollingWindow,
    CollapsableFrame,
)
from omni.kit.window.popup_dialog import MessageDialog
from isaacsim.gui.components.ui_utils import get_style
from .cameras.models import ZividCameraModelName
from .cameras.zivid_camera import ZividCamera
from .cameras.resolution import SamplingMode
from .assembler.mounts import Mount
from .utilities.fov import draw_fov, clear_fov
import pxr


def _make_model_dropdown(fn=None) -> DropDown:
    d = DropDown(
        "Camera model",
        tooltip=" Select a camera model",
        populate_fn=lambda: list(map(lambda c: str(c.name), list(ZividCameraModelName))),
        on_selection_fn=fn,
    )

    d.repopulate()
    return d


def _make_mount_dropdown(extra: list[str] = []) -> DropDown:
    d = DropDown(
        "Mount",
        tooltip=" Select a mount arm",
        populate_fn=lambda: list(map(lambda a: str(a.name), list(Mount))) + extra,
    )
    d.repopulate()
    return d


def _show_error(msg: str):
    m = MessageDialog(title="Error", warning_message=msg, disable_cancel_button=True, ok_handler=lambda d: d.hide())
    m.show()


class NewCameraWindow:
    def __init__(self, add_fn, find_parent_fn):
        self._add_fn = add_fn
        self._find_parent_fn = find_parent_fn
        self._usd_context = omni.usd.get_context()
        self._free_standing = False

        self._no_mount = "No mount"

        self._new_camera_window = omni.ui.Window(
            title="Add camera to scene",
            width=475,
            height=150,
            visible=False,
            dockPreference=omni.ui.DockPreference.DISABLED,
        )

        with self._new_camera_window.frame:
            with omni.ui.VStack():

                self._model_dropdown = _make_model_dropdown()
                self._mount_dropdown = _make_mount_dropdown([self._no_mount])

                self._mount_point = StringField(
                    "Mount point USD path",
                    tooltip="Type a string or use the file picker to set a value",
                )

                # def set_visibility(c: str):
                #     self._mount_point.visible = c == self._no_mount

                # self._mount_dropdown.set_on_selection_fn(set_visibility)

                with omni.ui.HStack():
                    self._ad_button = omni.ui.Button("Add", clicked_fn=self._add_clicked)
                    self._cancel_button = omni.ui.Button("Cancel", clicked_fn=self.close)

    def _add_clicked(self):
        mount_type = self._mount_dropdown.get_selection()
        camera = self._model_dropdown.get_selection()
        mount_path = self._mount_point.get_value()

        if not mount_path and not self._free_standing:
            _show_error("Mount point cannot be empty")
            return

        camera = [c for c in list(ZividCameraModelName) if str(c.name) == camera][0]
        mount_prim: pxr.Usd.Prim = self._usd_context.get_stage().GetPrimAtPath(mount_path)

        if self._free_standing:
            if mount_prim is None:
                self._add_fn(None, None, camera, None)
            else:
                self._add_fn(str(mount_prim.GetPath()), None, camera, None)
        else:
            if mount_type != self._no_mount:
                mount_type = [a for a in list(Mount) if str(a.name) == mount_type][0]

                if mount_prim:
                    parent = self._find_parent_fn(mount_prim)
                    if parent:
                        self._add_fn(str(mount_prim.GetPath()), str(parent.GetPath()), camera, mount_type)
            else:
                if mount_prim:
                    self._add_fn(str(mount_prim.GetPath()), None, camera, None)

        self.close()

    def set_free_standing(self, b: bool):
        self._free_standing = b

        if b:
            self._mount_point.set_value("")
            self._mount_dropdown.set_selection(self._no_mount)
            self._mount_point.visible = False
        else:
            selected_prim = self._usd_context.get_selection().get_selected_prim_paths()
            if len(selected_prim) == 1:
                self._mount_point.set_value(selected_prim[0])
            else:
                self._mount_point.set_value("")
            self._mount_point.visible = True

    def close(self):
        self._new_camera_window.visible = False

    def open(self, free_standing: bool):
        self.set_free_standing(free_standing)
        self._new_camera_window.visible = True


class CameraWidget:
    def __init__(self):
        self._camera = None
        self._usd_context = omni.usd.get_context()

        self._window = ScrollingWindow(
            title="Zivid Camera",
            width=600,
            height=500,
            visible=False,
            dockPreference=omni.ui.DockPreference.LEFT,
        )

        self._image_provider = omni.ui.ByteImageProvider()

        with self._window.frame:
            self._build_ui()

    def _capture_button_clicked(self):
        data = self._camera.get_data_rgb()
        if data.size != 0:
            self._image_provider.set_data_array(data, data.shape)
            self._image_widget.prepare_draw(data.shape[1], data.shape[0])
            self._width_widget.set_value(data.shape[1])
            self._height_widget.set_value(data.shape[0])

    def _update_sampling_mode(self, mode_str: str):
        if self._camera is not None:
            sampling_mode = [s for s in list(SamplingMode) if str(s.name) == mode_str][0]
            self._camera.set_sampling_mode(sampling_mode)

    def _update_camera_model(self, model_str: str):
        if self._camera is not None:
            model = [s for s in list(ZividCameraModelName) if str(s.name) == model_str][0]
            self._camera.set_camera_model(model)

    def _build_ui(self):
        with omni.ui.VStack(spacing=5, height=0):
            self._build_mount_frame()
            self._build_camera_frame()
            self._build_capture_frame()
            self._build_image_frame()

    def _build_mount_frame(self):
        model_frame = CollapsableFrame("Model", collapsed=False)
        with model_frame:
            with omni.ui.VStack(style=get_style(), spacing=5, height=0):
                self._model_dropdown = _make_model_dropdown(self._update_camera_model)

    def _build_camera_frame(self):
        settings_frame = CollapsableFrame("Settings", collapsed=False)
        with settings_frame:
            self._sampling_mode = DropDown(
                "Sampling mode",
                tooltip="The 3D sampling mode of the camera",
                populate_fn=lambda: list(map(lambda s: str(s.name), list(SamplingMode))),
                keep_old_selections=True,
                on_selection_fn=self._update_sampling_mode,
            )
            self._sampling_mode.repopulate()

    def _build_capture_frame(self):
        buttons_frame = CollapsableFrame("Capture", collapsed=False)
        with buttons_frame:
            with omni.ui.VStack(style=get_style(), spacing=5, height=0):
                self._capture_button = Button(
                    "Capture", "Capture", "Simulate a capture with the camera", on_click_fn=self._capture_button_clicked
                )
                self._visualize_fov = CheckBox(
                    "Visualize Field of View",
                    default_value=False,
                    tooltip="Click this checkbox to visualize the field of view",
                    on_click_fn=self._on_visualize_fov_clicked,
                )

    def _build_image_frame(self):
        image_frame = CollapsableFrame("Depth", collapsed=False)
        with image_frame:
            with omni.ui.VStack(style=get_style(), spacing=5, height=0):
                self._image_widget = omni.ui.ImageWithProvider(self._image_provider, width=400, height=300)
                self._width_widget = width = IntField(
                    "Width",
                    default_value=0,
                    tooltip="The width of the current capture in pixels",
                    lower_limit=0,
                    upper_limit=10000,
                )
                self._width_widget.enabled = False
                self._height_widget = IntField(
                    "Width",
                    default_value=0,
                    tooltip="The height of the current capture in pixels",
                    lower_limit=0,
                    upper_limit=10000,
                )
                self._height_widget.enabled = False

    def _on_visualize_fov_clicked(self, on):
        if on:
            draw_fov(self._camera)
        else:
            clear_fov()

    def _refresh_ui(self, camera: ZividCamera):
        self._camera = camera
        self._model_dropdown.set_selection(camera._model_name.name)
        self._window.title = f"Camera {self._camera.prim_path}"
        self._model_dropdown

    def open(self, camera: ZividCamera):
        self._refresh_ui(camera)
        self._window.visible = True

        async def dock_window():
            await omni.kit.app.get_app().next_update_async()

            def dock(space, name, location, pos=0.5):
                window = omni.ui.Workspace.get_window(name)
                if window and space:
                    window.dock_in(space, location, pos)
                return window

            tgt = omni.ui.Workspace.get_window("Viewport")
            dock(tgt, self._window.title, omni.ui.DockPosition.LEFT, 0.33)
            await omni.kit.app.get_app().next_update_async()

        self._task = asyncio.ensure_future(dock_window())
