import numpy as np

from internutopia.core.object import BaseObject
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.objects import DynamicCubeCfg


@BaseObject.register('DynamicCube')
class DynamicCube(BaseObject):
    def __init__(self, config: DynamicCubeCfg, scene: IScene):
        super().__init__(config, scene)
        self._config = config

    def set_up_to_scene(self, scene: IScene):
        from omni.isaac.core.objects import DynamicCuboid

        dynamic_cube = DynamicCuboid(
            prim_path=self._config.prim_path,
            name=self._config.name,
            position=np.array(self._config.position),
            orientation=np.array(self._config.orientation),
            scale=np.array(self._config.scale),
            color=np.array(self._config.color),
            mass=self._config.mass,
            density=self._config.density,
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
                dynamic_cube.apply_physics_material(physics_material)
            except Exception:
                # The object remains usable with Isaac's default material if material
                # helpers are unavailable in a particular extension startup order.
                pass
        scene.add(dynamic_cube)
