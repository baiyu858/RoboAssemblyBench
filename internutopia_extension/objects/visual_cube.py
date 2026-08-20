import numpy as np

from internutopia.core.object import BaseObject
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.objects import VisualCubeCfg


@BaseObject.register('VisualCube')
class VisualCube(BaseObject):
    def __init__(self, config: VisualCubeCfg, scene: IScene):
        super().__init__(config, scene)
        self._config = config

    def set_up_to_scene(self, scene: IScene):
        try:
            from isaacsim.core.api.objects import VisualCuboid
        except ImportError:
            from omni.isaac.core.objects.cuboid import VisualCuboid

        visual_cuboid = VisualCuboid(
            prim_path=self._config.prim_path,
            name=self._config.name,
            position=np.array(self._config.position),
            scale=np.array(self._config.scale),
            color=np.array(self._config.color),
        )
        scene.add(visual_cuboid)
        if self._config.texture_path:
            from internutopia_extension.objects.preview_surface import (
                bind_preview_surface_texture,
            )

            bind_preview_surface_texture(
                self._config.prim_path,
                texture_path=self._config.texture_path,
                texture_scale=tuple(self._config.texture_scale or (1.0, 1.0)),
                texture_rotation_degrees=float(self._config.texture_rotation_degrees or 0.0),
                surface_scale=tuple(float(value) for value in self._config.scale),
                surface_position=tuple(float(value) for value in self._config.position),
            )
