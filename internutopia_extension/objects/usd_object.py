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
        def require_local_file(resolved_path: str) -> str:
            resolved_scheme = urlparse(resolved_path).scheme
            if resolved_scheme in {'http', 'https', 'omniverse'}:
                return resolved_path
            if not os.path.isfile(resolved_path):
                raise FileNotFoundError(
                    f'Object USD asset does not exist: {resolved_path} (configured as {usd_path})'
                )
            return resolved_path

        path = os.path.expanduser(str(usd_path))
        parsed = urlparse(path)
        if parsed.scheme in {'http', 'https', 'omniverse'}:
            return path
        if path.startswith('${ISAAC_ASSETS_ROOT}') or path.startswith('/Isaac/'):
            assets_root_override = os.environ.get('ISAAC_ASSETS_ROOT', '').strip()
            if assets_root_override:
                suffix = path.removeprefix('${ISAAC_ASSETS_ROOT}')
                if path.startswith('/Isaac/'):
                    suffix = path
                resolved_path = (
                    os.path.abspath(os.path.expanduser(assets_root_override)).rstrip('/')
                    + '/'
                    + suffix.lstrip('/')
                )
                return require_local_file(resolved_path)
            try:
                from isaacsim.storage.native import get_assets_root_path

                assets_root = get_assets_root_path()
            except Exception:
                assets_root = None
            if assets_root:
                suffix = path.removeprefix('${ISAAC_ASSETS_ROOT}')
                if path.startswith('/Isaac/'):
                    suffix = path
                return require_local_file(assets_root.rstrip('/') + '/' + suffix.lstrip('/'))
            raise FileNotFoundError('Cannot resolve Isaac Sim assets root for object USD path: ' + path)
        return require_local_file(os.path.abspath(path))

    def set_up_to_scene(self, scene: IScene):  # noqa: C901
        try:
            from isaacsim.core.prims import SingleRigidPrim as RigidPrim
            from isaacsim.core.prims import SingleXFormPrim as XFormPrim
            from isaacsim.core.utils.prims import is_prim_path_valid
            from isaacsim.core.utils.stage import add_reference_to_stage
        except ImportError:
            from omni.isaac.core.prims import RigidPrim
            from omni.isaac.core.prims.xform_prim import XFormPrim
            from omni.isaac.core.utils.prims import is_prim_path_valid
            from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.physx.scripts import utils
        from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        force_renderable = bool(getattr(self._config, 'force_renderable', False))

        def prepare_nested_visuals(prim) -> None:
            if prim is None or not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                imageable.CreatePurposeAttr().Set('default')
            if prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayOpacityAttr([1.0])
                gprim.CreateDoubleSidedAttr(True)
                binding = prim.GetRelationship('material:binding')
                if binding and binding.IsValid() and binding.HasAuthoredTargets():
                    UsdShade.MaterialBindingAPI.Apply(prim)
            for child in prim.GetChildren():
                prepare_nested_visuals(child)

        def require_renderable_geometry(prim, *, prim_path: str, usd_path: str) -> None:
            geometry_count = 0

            def count_geometry(current_prim) -> None:
                nonlocal geometry_count
                if current_prim is None or not current_prim.IsValid():
                    return
                if current_prim.IsA(UsdGeom.Gprim):
                    geometry_count += 1
                for child in current_prim.GetChildren():
                    count_geometry(child)

            count_geometry(prim)
            if geometry_count == 0:
                raise RuntimeError(
                    f'Referenced USD contains no renderable geometry at {prim_path}: {usd_path}'
                )

        def set_nested_collision_enabled(prim, enabled: bool) -> None:
            if prim is None or not prim.IsValid():
                return
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.GetCollisionEnabledAttr().Set(enabled)
            for child in prim.GetChildren():
                set_nested_collision_enabled(child, enabled)

        def set_nested_rigid_body_enabled(prim, enabled: bool) -> None:
            if prim is None or not prim.IsValid():
                return
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
                rigid_body_api.GetRigidBodyEnabledAttr().Set(enabled)
            for child in prim.GetChildren():
                set_nested_rigid_body_enabled(child, enabled)

        def set_nested_colliders(prim, *, dynamic_body: bool) -> None:
            if prim is None or not prim.IsValid():
                return
            # Preserve collision approximation authored in referenced assets.
            # Fabrica parts carry PhysX SDF settings; calling setCollider with
            # approximationShape=None replaces those settings with a triangle
            # mesh, which PhysX cannot use on a dynamic rigid body.
            if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
                try:
                    approximation_shape = 'convexHull' if dynamic_body else None
                    utils.setCollider(prim, approximationShape=approximation_shape)
                except Exception:
                    pass
            for child in prim.GetChildren():
                set_nested_colliders(child, dynamic_body=dynamic_body)

        def prepare_dynamic_mesh_colliders(prim) -> None:
            colliders = []

            def _has_sdf_collision(current_prim) -> bool:
                try:
                    from pxr import PhysxSchema

                    return bool(current_prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI))
                except Exception:
                    return False

            def collect(current_prim) -> None:
                if current_prim is None or not current_prim.IsValid():
                    return
                if current_prim.IsA(UsdGeom.Mesh) and current_prim.HasAPI(UsdPhysics.CollisionAPI):
                    mesh_collision_api = (
                        UsdPhysics.MeshCollisionAPI(current_prim)
                        if current_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                        else UsdPhysics.MeshCollisionAPI.Apply(current_prim)
                    )
                    approximation = mesh_collision_api.GetApproximationAttr().Get()
                    approximation_name = '' if approximation is None else str(approximation).strip().lower()
                    if _has_sdf_collision(current_prim):
                        approximation_name = 'sdf'
                    colliders.append((current_prim, mesh_collision_api, approximation_name))
                for child in current_prim.GetChildren():
                    collect(child)

            collect(prim)
            unsupported = {'', 'none', 'meshsimplification', 'trianglemesh'}
            has_supported_collider = any(name not in unsupported for _, _, name in colliders)
            for collider_prim, mesh_collision_api, approximation_name in colliders:
                if approximation_name not in unsupported:
                    continue
                if has_supported_collider:
                    UsdPhysics.CollisionAPI(collider_prim).GetCollisionEnabledAttr().Set(False)
                else:
                    mesh_collision_api.GetApproximationAttr().Set(
                        getattr(UsdPhysics.Tokens, 'convexHull', 'convexHull')
                    )

        def apply_collision_approximation(prim, approximation: str, *, existing_colliders_only=False) -> None:
            if prim is None or not prim.IsValid() or not approximation:
                return
            token = str(approximation).strip()
            if not token:
                return
            token_lower = token.lower()
            # Literal "sdf" (not UsdPhysics.Tokens) — matches grscenes preprocess.
            token_attr = 'sdf' if token_lower in {'sdf', 'sdfmesh'} else getattr(UsdPhysics.Tokens, token, token)
            # Use a resolution distinct from authored sdf512 so PhysX re-cooks.
            sdf_resolution = 500

            def walk(current_prim) -> None:
                if current_prim is None or not current_prim.IsValid():
                    return
                if (current_prim.IsA(UsdGeom.Mesh)
                        and (not existing_colliders_only or (
                            current_prim.HasAPI(UsdPhysics.CollisionAPI)
                            and UsdPhysics.CollisionAPI(current_prim).GetCollisionEnabledAttr().Get() is not False))):
                    try:
                        from pxr import PhysxSchema

                        if not current_prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI.Apply(current_prim)
                        UsdPhysics.CollisionAPI(current_prim).GetCollisionEnabledAttr().Set(True)
                        mesh_collision_api = (
                            UsdPhysics.MeshCollisionAPI(current_prim)
                            if current_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                            else UsdPhysics.MeshCollisionAPI.Apply(current_prim)
                        )
                        approx_attr = mesh_collision_api.GetApproximationAttr()
                        if approx_attr is None or not approx_attr.IsValid():
                            mesh_collision_api.CreateApproximationAttr().Set(token_attr)
                        else:
                            approx_attr.Set(token_attr)
                        if existing_colliders_only and token_lower not in {'sdf', 'sdfmesh'}:
                            # A referenced SDF API can override the mesh token.
                            # Remove it in this scene layer before physics starts;
                            # keep the source asset, vertices and holes unchanged.
                            sdf_type = PhysxSchema.PhysxSDFMeshCollisionAPI
                            if current_prim.HasAPI(sdf_type):
                                current_prim.RemoveAPI(sdf_type)
                                if current_prim.HasAPI(sdf_type):
                                    raise RuntimeError('Could not remove referenced SDF collision API')
                                for attr in current_prim.GetAttributes():
                                    if attr.GetName().startswith('physxSDFMeshCollision:'):
                                        attr.Block()
                        if token_lower in {'sdf', 'sdfmesh'}:
                            for remove_name in (
                                'PhysxTriangleMeshCollisionAPI',
                                'PhysxTriangleMeshSimplificationCollisionAPI',
                                'PhysxConvexHullCollisionAPI',
                                'PhysxConvexDecompositionCollisionAPI',
                                'PhysxSDFMeshCollisionAPI',
                            ):
                                remove_api = getattr(PhysxSchema, remove_name, None)
                                if remove_api is not None and current_prim.HasAPI(remove_api):
                                    try:
                                        current_prim.RemoveAPI(remove_api)
                                    except Exception:
                                        pass
                            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(current_prim)
                            sdf_api.CreateSdfResolutionAttr().Set(int(sdf_resolution))
                    except Exception as exc:
                        if existing_colliders_only:
                            raise RuntimeError(
                                f'Cannot configure static collider {current_prim.GetPath()} as {token}'
                            ) from exc
                for child in current_prim.GetChildren():
                    walk(child)

            walk(prim)

        def bind_physics_material_to_colliders(
            prim,
            *,
            name: str,
            static_friction: Optional[float],
            dynamic_friction: Optional[float],
            restitution: Optional[float],
            friction_combine_mode: Optional[str] = None,
        ) -> None:
            # Referenced Fabrica meshes carry their own direct
            # material:binding:physics (friction 0); a root-level binding loses
            # to it, so the override must land on each collider prim.
            if prim is None or not prim.IsValid():
                return
            if static_friction is None and dynamic_friction is None and restitution is None:
                return
            from isaacsim.core.api.materials import PhysicsMaterial
            from pxr import Usd, UsdShade

            material_name = name.replace('/', '_')
            physics_material = PhysicsMaterial(
                prim_path=f'/World/Physics_Materials/{material_name}_physics_material',
                name=f'{material_name}_physics_material',
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            )
            if friction_combine_mode:
                from pxr import PhysxSchema

                physx_material = PhysxSchema.PhysxMaterialAPI.Apply(physics_material.prim)
                physx_material.CreateFrictionCombineModeAttr().Set(str(friction_combine_mode))
            targets = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)] or [prim]
            for target in targets:
                UsdShade.MaterialBindingAPI.Apply(target).Bind(
                    physics_material.material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose='physics',
                )

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
                auto_collider: Optional[bool] = True,
                collision_approximation: Optional[str] = None,
                static_friction: Optional[float] = None,
                dynamic_friction: Optional[float] = None,
                restitution: Optional[float] = None,
                friction_combine_mode: Optional[str] = None,
                linear_damping: Optional[float] = None,
                angular_damping: Optional[float] = None,
                sleep_threshold: Optional[float] = None,
                stabilization_threshold: Optional[float] = None,
                solver_position_iteration_count: Optional[int] = None,
                solver_velocity_iteration_count: Optional[int] = None,
            ) -> None:
                if not is_prim_path_valid(prim_path):
                    if mass is None:
                        mass = 1
                prim = add_reference_to_stage(UsdObject._resolve_usd_path(usd_path), prim_path)
                if force_renderable:
                    prepare_nested_visuals(prim)
                    require_renderable_geometry(prim, prim_path=prim_path, usd_path=usd_path)
                set_nested_rigid_body_enabled(prim, False)
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                rigid_body_api.GetRigidBodyEnabledAttr().Set(True)
                if collider and auto_collider:
                    set_nested_colliders(prim, dynamic_body=True)
                elif not collider:
                    set_nested_collision_enabled(prim, False)
                if collider:
                    prepare_dynamic_mesh_colliders(prim)
                    if collision_approximation:
                        apply_collision_approximation(prim, collision_approximation)
                RigidPrim.__init__(
                    self,
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    translation=translation,
                    orientation=orientation,
                    scale=scale,
                    visible=True if force_renderable else visible,
                    mass=mass,
                    density=density,
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                )
                physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                optional_physics_values = (
                    ('GetLinearDampingAttr', linear_damping),
                    ('GetAngularDampingAttr', angular_damping),
                    ('GetSleepThresholdAttr', sleep_threshold),
                    ('GetStabilizationThresholdAttr', stabilization_threshold),
                    ('GetSolverPositionIterationCountAttr', solver_position_iteration_count),
                    ('GetSolverVelocityIterationCountAttr', solver_velocity_iteration_count),
                )
                for attribute_getter, value in optional_physics_values:
                    if value is not None:
                        getattr(physx_rigid_body, attribute_getter)().Set(value)
                bind_physics_material_to_colliders(
                    prim,
                    name=name,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=restitution,
                    friction_combine_mode=friction_combine_mode,
                )

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
                auto_collider: Optional[bool] = True,
                collision_approximation: Optional[str] = None,
                static_friction: Optional[float] = None,
                dynamic_friction: Optional[float] = None,
                restitution: Optional[float] = None,
                friction_combine_mode: Optional[str] = None,
                kinematic_anchor: Optional[bool] = None,
            ) -> None:
                prim = add_reference_to_stage(UsdObject._resolve_usd_path(usd_path), prim_path)
                if force_renderable:
                    prepare_nested_visuals(prim)
                    require_renderable_geometry(prim, prim_path=prim_path, usd_path=usd_path)
                set_nested_rigid_body_enabled(prim, False)
                if collider and auto_collider:
                    set_nested_colliders(prim, dynamic_body=False)
                elif not collider:
                    set_nested_collision_enabled(prim, False)
                if collider and collision_approximation:
                    apply_collision_approximation(prim, collision_approximation, existing_colliders_only=True)
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
                    visible=True if force_renderable else visible,
                )
                if collider:
                    bind_physics_material_to_colliders(
                        prim,
                        name=name,
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=restitution,
                        friction_combine_mode=friction_combine_mode,
                    )
                if kinematic_anchor:
                    # The pickup tray is a thin SDF with no rigid body. PhysX
                    # then treats it as a light dynamic shell, and part friction
                    # lifts the whole nest (v81–v82). A kinematic body stays put.
                    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)
                    rigid_body_api.CreateKinematicEnabledAttr().Set(True)
                    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    physx_body.CreateDisableGravityAttr().Set(True)

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
                    auto_collider=self._config.auto_collider,
                    collision_approximation=getattr(self._config, 'collision_approximation', None),
                    mass=self._config.mass,
                    density=self._config.density,
                    static_friction=self._config.static_friction,
                    dynamic_friction=self._config.dynamic_friction,
                    restitution=self._config.restitution,
                    friction_combine_mode=getattr(self._config, 'friction_combine_mode', None),
                    linear_damping=self._config.linear_damping,
                    angular_damping=self._config.angular_damping,
                    sleep_threshold=self._config.sleep_threshold,
                    stabilization_threshold=self._config.stabilization_threshold,
                    solver_position_iteration_count=self._config.solver_position_iteration_count,
                    solver_velocity_iteration_count=self._config.solver_velocity_iteration_count,
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
                    auto_collider=self._config.auto_collider,
                    collision_approximation=getattr(self._config, 'collision_approximation', None),
                    static_friction=self._config.static_friction,
                    dynamic_friction=self._config.dynamic_friction,
                    restitution=self._config.restitution,
                    friction_combine_mode=getattr(self._config, 'friction_combine_mode', None),
                    kinematic_anchor=bool(getattr(self._config, 'kinematic_anchor', False)),
                )
            )
