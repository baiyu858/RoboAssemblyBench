import numpy as np

from internutopia.core.object import BaseObject
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.objects import DynamicCompoundCuboidCfg


@BaseObject.register('DynamicCompoundCuboid')
class DynamicCompoundCuboid(BaseObject):
    def __init__(self, config: DynamicCompoundCuboidCfg, scene: IScene):
        super().__init__(config, scene)
        self._config = config

    def set_up_to_scene(self, scene: IScene):
        import omni.usd
        from omni.isaac.core.prims import GeometryPrim, RigidPrim
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError('USD stage is unavailable while creating DynamicCompoundCuboid.')

        root_xform = UsdGeom.Xform.Define(stage, self._config.prim_path)
        root_prim = root_xform.GetPrim()
        root_xformable = UsdGeom.Xformable(root_prim)
        root_xformable.ClearXformOpOrder()
        root_xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in self._config.position]))
        orientation = tuple(float(value) for value in (self._config.orientation or (1.0, 0.0, 0.0, 0.0)))
        root_xformable.AddOrientOp().Set(
            Gf.Quatf(
                orientation[0],
                Gf.Vec3f(orientation[1], orientation[2], orientation[3]),
            )
        )
        root_scale = tuple(float(value) for value in (self._config.scale or (1.0, 1.0, 1.0)))
        root_xformable.AddScaleOp().Set(Gf.Vec3f(*root_scale))

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
        rigid_body_api.GetRigidBodyEnabledAttr().Set(True)
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        if self._config.mass is not None:
            mass_api.CreateMassAttr().Set(float(self._config.mass))
        if self._config.density is not None:
            mass_api.CreateDensityAttr().Set(float(self._config.density))

        child_paths = []
        default_color = tuple(float(value) for value in (self._config.color or (0.5, 0.5, 0.5)))
        for part in self._config.parts:
            part_path = f'{self._config.prim_path}/{part.name}'
            cube = UsdGeom.Cube.Define(stage, part_path)
            cube.CreateSizeAttr(1.0)
            color = tuple(float(value) for value in (part.color or default_color))
            cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

            part_prim = cube.GetPrim()
            part_xformable = UsdGeom.Xformable(part_prim)
            part_xformable.ClearXformOpOrder()
            part_xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in part.offset]))
            part_xformable.AddScaleOp().Set(Gf.Vec3f(*[float(value) for value in part.scale]))

            collision_api = UsdPhysics.CollisionAPI.Apply(part_prim)
            collision_api.GetCollisionEnabledAttr().Set(True)
            child_paths.append(part_path)

        rigid = RigidPrim(
            prim_path=self._config.prim_path,
            name=self._config.name,
        )
        self._apply_optional_physics_material(rigid=rigid, child_paths=child_paths)
        scene.add(rigid)

    def _apply_optional_physics_material(self, *, rigid, child_paths):
        if (
            self._config.static_friction is None
            and self._config.dynamic_friction is None
            and self._config.restitution is None
        ):
            return

        try:
            from isaacsim.core.api.materials import PhysicsMaterial
        except Exception:
            return

        material_name = self._config.name.replace('/', '_')
        try:
            physics_material = PhysicsMaterial(
                prim_path=f'/World/Physics_Materials/{material_name}_physics_material',
                name=f'{material_name}_physics_material',
                static_friction=self._config.static_friction,
                dynamic_friction=self._config.dynamic_friction,
                restitution=self._config.restitution,
            )
        except Exception:
            return

        try:
            rigid.apply_physics_material(physics_material)
        except Exception:
            pass

        for child_path in child_paths:
            try:
                geometry = GeometryPrim(
                    prim_path=child_path,
                    name=child_path.split('/')[-1],
                    collision=True,
                )
                geometry.apply_physics_material(physics_material)
            except Exception:
                continue
