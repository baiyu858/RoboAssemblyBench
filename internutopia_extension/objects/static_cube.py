import numpy as np

from internutopia.core.object import BaseObject
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.objects import StaticCubeCfg


@BaseObject.register('StaticCube')
class StaticCube(BaseObject):
    def __init__(self, config: StaticCubeCfg, scene: IScene):
        super().__init__(config, scene)
        self._config = config

    def set_up_to_scene(self, scene: IScene):
        try:
            from isaacsim.core.api.objects import FixedCuboid
        except ImportError:
            try:
                from omni.isaac.core.objects import FixedCuboid
            except ImportError:
                from omni.isaac.core.objects.cuboid import FixedCuboid

        static_cube = FixedCuboid(
            prim_path=self._config.prim_path,
            name=self._config.name,
            position=np.array(self._config.position),
            orientation=np.array(self._config.orientation),
            scale=np.array(self._config.scale),
            color=np.array(self._config.color),
        )
        if (
            self._config.static_friction is not None
            or self._config.dynamic_friction is not None
            or self._config.restitution is not None
        ):
            try:
                from isaacsim.core.api.materials import PhysicsMaterial

                material_name = self._config.name.replace('/', '_')
                physics_material = PhysicsMaterial(
                    prim_path=f'/World/Physics_Materials/{material_name}_physics_material',
                    name=f'{material_name}_physics_material',
                    static_friction=self._config.static_friction,
                    dynamic_friction=self._config.dynamic_friction,
                    restitution=self._config.restitution,
                )
                static_cube.apply_physics_material(physics_material)
            except Exception:
                pass
        scene.add(static_cube)
        if self._config.texture_path:
            from internutopia_extension.objects.preview_surface import bind_preview_surface_texture

            bind_preview_surface_texture(
                self._config.prim_path,
                texture_path=self._config.texture_path,
                texture_scale=tuple(self._config.texture_scale or (1.0, 1.0)),
                texture_rotation_degrees=float(self._config.texture_rotation_degrees or 0.0),
                surface_scale=tuple(float(value) for value in self._config.scale),
                surface_position=tuple(float(value) for value in self._config.position),
            )
