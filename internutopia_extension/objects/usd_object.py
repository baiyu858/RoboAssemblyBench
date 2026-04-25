import os
from typing import Optional, Sequence

import numpy as np

from internutopia.core.object import BaseObject
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.objects import UsdObjCfg


@BaseObject.register('UsdObject')
class UsdObject(BaseObject):
    def __init__(self, config: UsdObjCfg, scene: IScene):
        super().__init__(config, scene)
        self._config = config

    def set_up_to_scene(self, scene: IScene):
        from omni.isaac.core.prims import GeometryPrim, RigidPrim
        from omni.isaac.core.utils.prims import is_prim_path_valid
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.physx.scripts import utils

        class RigidObject(RigidPrim):
            def __init__(
                self,
                prim_path: str,
                usd_path: str,
                name: str = 'custom_obj',
                position: Optional[np.ndarray] = None,
                translation: Optional[np.ndarray] = None,
                orientation: Optional[np.ndarray] = None,
                scale: Optional[np.ndarray] = None,
                visible: Optional[bool] = None,
                mass: Optional[float] = None,
                density: Optional[float] = None,
                linear_velocity: Optional[Sequence[float]] = None,
                angular_velocity: Optional[Sequence[float]] = None,
                collider: Optional[bool] = True,
                static_friction: Optional[float] = None,
                dynamic_friction: Optional[float] = None,
                restitution: Optional[float] = None,
            ) -> None:
                if not is_prim_path_valid(prim_path):
                    if mass is None:
                        mass = 1
                prim = add_reference_to_stage(os.path.abspath(usd_path), prim_path)
                if collider:
                    utils.setCollider(prim, approximationShape=None)
                RigidPrim.__init__(
                    self,
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    translation=translation,
                    orientation=orientation,
                    scale=scale,
                    visible=visible,
                    mass=mass,
                    density=density,
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                )
                self._apply_optional_physics_material(
                    name=name,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=restitution,
                )

            def _apply_optional_physics_material(
                self,
                *,
                name: str,
                static_friction: Optional[float],
                dynamic_friction: Optional[float],
                restitution: Optional[float],
            ) -> None:
                if static_friction is None and dynamic_friction is None and restitution is None:
                    return
                try:
                    from isaacsim.core.api.materials import PhysicsMaterial

                    material_name = name.replace('/', '_')
                    physics_material = PhysicsMaterial(
                        prim_path=f'/World/Physics_Materials/{material_name}_physics_material',
                        name=f'{material_name}_physics_material',
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=restitution,
                    )
                    self.apply_physics_material(physics_material)
                except Exception:
                    return

        class GeometryObject(GeometryPrim):
            def __init__(
                self,
                prim_path: str,
                usd_path: str,
                name: str = 'visual_cube',
                position: Optional[Sequence[float]] = None,
                translation: Optional[Sequence[float]] = None,
                orientation: Optional[Sequence[float]] = None,
                scale: Optional[Sequence[float]] = None,
                visible: Optional[bool] = None,
                color: Optional[np.ndarray] = None,
                size: Optional[float] = None,
                collider: Optional[bool] = False,
                static_friction: Optional[float] = None,
                dynamic_friction: Optional[float] = None,
                restitution: Optional[float] = None,
            ) -> None:
                prim = add_reference_to_stage(os.path.abspath(usd_path), prim_path)
                if collider:
                    utils.setCollider(prim, approximationShape=None)
                self.size = size
                self.color = color
                GeometryPrim.__init__(
                    self,
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    translation=translation,
                    orientation=orientation,
                    scale=scale,
                    visible=visible,
                    collision=collider,
                )
                if collider and (
                    static_friction is not None
                    or dynamic_friction is not None
                    or restitution is not None
                ):
                    try:
                        from isaacsim.core.api.materials import PhysicsMaterial

                        material_name = name.replace('/', '_')
                        physics_material = PhysicsMaterial(
                            prim_path=f'/World/Physics_Materials/{material_name}_physics_material',
                            name=f'{material_name}_physics_material',
                            static_friction=static_friction,
                            dynamic_friction=dynamic_friction,
                            restitution=restitution,
                        )
                        self.apply_physics_material(physics_material)
                    except Exception:
                        pass

        if self._config.rigid_body:
            scene.add(
                RigidObject(
                    usd_path=self._config.usd_path,
                    prim_path=self._config.prim_path,
                    name=self._config.name,
                    position=self._config.position,
                    orientation=self._config.orientation,
                    scale=self._config.scale,
                    collider=self._config.collider,
                    static_friction=self._config.static_friction,
                    dynamic_friction=self._config.dynamic_friction,
                    restitution=self._config.restitution,
                )
            )
        else:
            scene.add(
                GeometryObject(
                    usd_path=self._config.usd_path,
                    prim_path=self._config.prim_path,
                    name=self._config.name,
                    position=self._config.position,
                    orientation=self._config.orientation,
                    scale=self._config.scale,
                    collider=self._config.collider,
                    static_friction=self._config.static_friction,
                    dynamic_friction=self._config.dynamic_friction,
                    restitution=self._config.restitution,
                )
            )
