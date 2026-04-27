import os
from typing import Optional, Sequence
from urllib.parse import urlparse

import numpy as np

from internutopia.core.object import BaseObject
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.objects import UsdObjCfg


@BaseObject.register('UsdObject')
class UsdObject(BaseObject):
    def __init__(self, config: UsdObjCfg, scene: IScene):
        super().__init__(config, scene)
        self._config = config

    @staticmethod
    def _resolve_usd_path(usd_path: str) -> str:
        path = os.path.expanduser(str(usd_path))
        parsed = urlparse(path)
        if parsed.scheme in {'http', 'https', 'omniverse'}:
            return path
        if path.startswith('${ISAAC_ASSETS_ROOT}') or path.startswith('/Isaac/'):
            try:
                from isaacsim.storage.native import get_assets_root_path

                assets_root = get_assets_root_path()
            except Exception:
                assets_root = None
            if assets_root:
                suffix = path.removeprefix('${ISAAC_ASSETS_ROOT}')
                if path.startswith('/Isaac/'):
                    suffix = path
                return assets_root.rstrip('/') + '/' + suffix.lstrip('/')
            raise FileNotFoundError('Cannot resolve Isaac Sim assets root for object USD path: ' + path)
        return os.path.abspath(path)

    def set_up_to_scene(self, scene: IScene):
        from omni.isaac.core.prims import RigidPrim
        from omni.isaac.core.prims.xform_prim import XFormPrim
        from omni.isaac.core.utils.prims import is_prim_path_valid
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.physx.scripts import utils
        from pxr import UsdPhysics

        def set_nested_collision_enabled(prim, enabled: bool) -> None:
            if prim is None or not prim.IsValid():
                return
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.GetCollisionEnabledAttr().Set(enabled)
            for child in prim.GetChildren():
                set_nested_collision_enabled(child, enabled)

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
                prim = add_reference_to_stage(UsdObject._resolve_usd_path(usd_path), prim_path)
                if collider:
                    utils.setCollider(prim, approximationShape=None)
                else:
                    set_nested_collision_enabled(prim, False)
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

        class GeometryObject(XFormPrim):
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
                prim = add_reference_to_stage(UsdObject._resolve_usd_path(usd_path), prim_path)
                if collider:
                    utils.setCollider(prim, approximationShape=None)
                else:
                    set_nested_collision_enabled(prim, False)
                self.size = size
                self.color = color
                # Complex referenced USD assets should stay as XForms when they
                # are visual-only. Wrapping them as GeometryPrim can make root
                # collision/visibility handling unstable for factory props.
                XFormPrim.__init__(
                    self,
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    translation=translation,
                    orientation=orientation,
                    scale=scale,
                    visible=visible,
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
                        if hasattr(self, 'apply_physics_material'):
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
