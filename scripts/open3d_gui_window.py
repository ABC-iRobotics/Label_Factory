"""Open3D SceneWidget window with a persistent bottom keyboard-help bar."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


class HelpSceneWindow:
    def __init__(self, title: str, help_text: str, width: int, height: int,
                 background=(0.025, 0.025, 0.035, 1.0)) -> None:
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(title, width, height)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background(np.asarray(background, dtype=np.float32))
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.help = gui.Label(help_text)
        self.help.text_color = gui.Color(0.92, 0.92, 0.92)
        self.help.background_color = gui.Color(0.08, 0.08, 0.10)
        self.window.add_child(self.widget)
        self.window.add_child(self.help)
        self._key_callbacks: dict[int, Callable[[], None]] = {}
        self.window.set_on_layout(self._layout)
        self.widget.set_on_key(self._on_key)

    def _layout(self, _context) -> None:
        rect = self.window.content_rect
        bar_height = int(max(46, self.window.theme.font_size * 2.6))
        self.widget.frame = gui.Rect(rect.x, rect.y, rect.width,
                                     max(1, rect.height - bar_height))
        self.help.frame = gui.Rect(rect.x + 12, rect.get_bottom() - bar_height,
                                   max(1, rect.width - 24), bar_height)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.Type.DOWN and int(event.key) in self._key_callbacks:
            self._key_callbacks[int(event.key)]()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def register_key(self, key: int, callback: Callable[[], None]) -> None:
        # Legacy Visualizer callbacks use GLFW-style uppercase codes, whereas
        # SceneWidget KeyName values use lowercase ASCII and Escape=27. Crucially,
        # SceneWidget uses 256 for Left Shift, so never retain legacy Escape=256.
        if key == 256:
            self._key_callbacks[27] = callback
            return
        self._key_callbacks[key] = callback
        if ord("A") <= key <= ord("Z"):
            self._key_callbacks[key + 32] = callback

    @staticmethod
    def point_material(point_size: float = 2.0):
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = point_size
        return material

    @staticmethod
    def line_material(line_width: float = 1.5):
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = line_width
        return material

    @staticmethod
    def mesh_material():
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        return material

    def add(self, name: str, geometry, material) -> None:
        if self.widget.scene.has_geometry(name):
            self.widget.scene.remove_geometry(name)
        self.widget.scene.add_geometry(name, geometry, material)

    def remove(self, name: str) -> None:
        if self.widget.scene.has_geometry(name):
            self.widget.scene.remove_geometry(name)

    def fit_view(self, bounds: o3d.geometry.AxisAlignedBoundingBox) -> None:
        self.widget.setup_camera(60.0, bounds, bounds.get_center())

    def run_one_tick(self) -> bool:
        return bool(self.app.run_one_tick())

    def run(self) -> None:
        self.app.run()

    def save_image(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        def write(image) -> None:
            o3d.io.write_image(str(output), image)

        self.widget.scene.scene.render_to_image(write)

    def close(self) -> None:
        self.window.close()
