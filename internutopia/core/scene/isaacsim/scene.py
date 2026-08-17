import os
from typing import List

from internutopia.core.config import TaskCfg
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.scene import validate_scene_file
from internutopia.core.scene.scene import IScene
from internutopia.core.task_config_manager.base import runtime_root_path


class IsaacsimScene(IScene):
    """IsaacSim's implementation on `IScene` class."""

    def __init__(self):
        try:
            from isaacsim.core.api import World
            from isaacsim.core.api.scenes import Scene
        except ImportError:
            from omni.isaac.core import World
            from omni.isaac.core.scenes import Scene

        self._scene: Scene = World.instance().scene

    @staticmethod
    def _set_scene_transform(scene_prim, *, position, orientation, scale) -> None:
        from pxr import Gf, UsdGeom

        xformable = UsdGeom.Xformable(scene_prim)

        def get_or_add_op(name, add_op):
            attribute = scene_prim.GetAttribute(f'xformOp:{name}')
            if attribute.IsValid():
                return UsdGeom.XformOp(attribute)
            return add_op()

        translate_op = get_or_add_op('translate', xformable.AddTranslateOp)
        orient_op = get_or_add_op('orient', xformable.AddOrientOp)
        scale_op = get_or_add_op('scale', xformable.AddScaleOp)

        def vec3(op, values):
            values = [float(value) for value in values]
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                return Gf.Vec3d(*values)
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionHalf:
                return Gf.Vec3h(*values)
            return Gf.Vec3f(*values)

        def quat(op, values):
            real, imaginary = float(values[0]), [float(value) for value in values[1:]]
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                return Gf.Quatd(real, Gf.Vec3d(*imaginary))
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionHalf:
                return Gf.Quath(real, Gf.Vec3h(*imaginary))
            return Gf.Quatf(real, Gf.Vec3f(*imaginary))

        translate_op.Set(vec3(translate_op, position))
        orient_op.Set(quat(orient_op, orientation))
        scale_op.Set(vec3(scale_op, scale))
        xformable.SetXformOpOrder([translate_op, orient_op, scale_op])

    def load(self, task_config: TaskCfg, env_id: int, env_offset: List[float]):
        """See `IScene.load` for documentation."""
        usd_path = self._resolve_scene_asset_path(task_config)
        fallback_path = getattr(task_config, 'scene_asset_fallback_path', None)
        resolved_fallback_path = None
        if fallback_path:
            try:
                resolved_fallback_path = self._resolve_isaac_asset_path(fallback_path)
            except Exception:
                pass
        using_fallback = bool(resolved_fallback_path and usd_path == resolved_fallback_path)
        profile_metadata = getattr(task_config, 'scene_profile_metadata', {}) or {}
        task_config.scene_asset_source = 'fallback' if using_fallback else 'primary'
        task_config.resolved_scene_family = str(
            profile_metadata.get('fallback_scene_family' if using_fallback else 'scene_family', '')
        )
        task_config.scene_asset_path = usd_path
        prim_path_root = f'{runtime_root_path(task_config, env_id).lstrip("/")}/scene'
        source, prim_path = validate_scene_file(usd_path, prim_path_root)

        try:
            from isaacsim.core.utils.prims import create_prim
        except ImportError:
            from omni.isaac.core.utils.prims import create_prim
        position = [env_offset[idx] + i for idx, i in enumerate(task_config.scene_position)]
        scene_prim = create_prim(prim_path, usd_path=source, scale=task_config.scene_scale, translation=position)
        orientation = [float(value) for value in task_config.scene_orientation]
        if orientation != [1.0, 0.0, 0.0, 0.0]:
            self._set_scene_transform(
                scene_prim,
                position=position,
                orientation=orientation,
                scale=task_config.scene_scale,
            )
        self.scene_prim = scene_prim
        self._load_scene_lights(task_config, env_id)

    @staticmethod
    def _light_prim_type(kind: str):
        from pxr import UsdLux

        normalized = kind.lower().replace('-', '_')
        if normalized in {'dome', 'dome_light', 'domelight'}:
            return UsdLux.DomeLight
        if normalized in {'distant', 'distant_light', 'distantlight'}:
            return UsdLux.DistantLight
        if normalized in {'rect', 'rect_light', 'rectlight'}:
            return UsdLux.RectLight
        if normalized in {'sphere', 'sphere_light', 'spherelight'}:
            return UsdLux.SphereLight
        raise ValueError(f'Unsupported scene light kind: {kind!r}')

    @staticmethod
    def _set_light_transform(light_prim, light_spec: dict):
        from pxr import Gf, UsdGeom

        xformable = UsdGeom.Xformable(light_prim)
        xformable.ClearXformOpOrder()
        position = light_spec.get('position')
        if position is not None:
            xformable.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in position)))

        rotation = light_spec.get('rotation_euler', light_spec.get('rotation'))
        if rotation is not None:
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(*(float(value) for value in rotation)))

    def _load_scene_lights(self, task_config: TaskCfg, env_id: int):
        scene_lights = getattr(task_config, 'scene_lights', None) or []
        if not scene_lights:
            return

        from pxr import Gf, Sdf, UsdGeom

        stage = self.scene_prim.GetStage()
        lights_root = f'{runtime_root_path(task_config, env_id)}/lights'
        UsdGeom.Scope.Define(stage, Sdf.Path(lights_root))

        for index, light_spec in enumerate(scene_lights):
            name = light_spec.get('name') or f'scene_light_{index}'
            prim_path = f'{lights_root}/{name}'
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)

            light_cls = self._light_prim_type(light_spec.get('kind', 'dome'))
            light = light_cls.Define(stage, Sdf.Path(prim_path))
            if light_spec.get('intensity') is not None:
                light.CreateIntensityAttr(float(light_spec['intensity']))
            if light_spec.get('color') is not None:
                light.CreateColorAttr(Gf.Vec3f(*(float(value) for value in light_spec['color'])))
            if light_spec.get('exposure') is not None:
                light.CreateExposureAttr(float(light_spec['exposure']))

            if hasattr(light, 'CreateAngleAttr') and light_spec.get('angle') is not None:
                light.CreateAngleAttr(float(light_spec['angle']))
            if hasattr(light, 'CreateRadiusAttr') and light_spec.get('radius') is not None:
                light.CreateRadiusAttr(float(light_spec['radius']))
            if hasattr(light, 'CreateWidthAttr') and light_spec.get('width') is not None:
                light.CreateWidthAttr(float(light_spec['width']))
            if hasattr(light, 'CreateHeightAttr') and light_spec.get('height') is not None:
                light.CreateHeightAttr(float(light_spec['height']))

            self._set_light_transform(light.GetPrim(), light_spec)

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        return path.startswith(('omniverse://', 'http://', 'https://'))

    @classmethod
    def _resolve_isaac_asset_path(cls, path: str) -> str:
        if path.startswith('${ISAAC_ASSETS_ROOT}'):
            suffix = path.removeprefix('${ISAAC_ASSETS_ROOT}')
        elif path.startswith('/Isaac/'):
            suffix = path
        else:
            return path if cls._is_remote_path(path) else os.path.abspath(path)

        assets_root_override = os.environ.get('ISAAC_ASSETS_ROOT', '').strip()
        if assets_root_override:
            assets_root_override = os.path.expanduser(os.path.expandvars(assets_root_override))
            if not cls._is_remote_path(assets_root_override):
                assets_root_override = os.path.abspath(assets_root_override)
            return assets_root_override.rstrip('/') + '/' + suffix.lstrip('/')

        from isaacsim.storage.native import get_assets_root_path

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise FileNotFoundError('Cannot resolve Isaac Sim assets root for scene path: ' + path)
        return assets_root_path.rstrip('/') + '/' + suffix.lstrip('/')

    @classmethod
    def _scene_asset_exists(cls, path: str) -> bool:
        if cls._is_remote_path(path):
            from isaacsim.storage.native import is_file

            return bool(is_file(path))
        return os.path.exists(path)

    @classmethod
    def _resolve_scene_asset_path(cls, task_config: TaskCfg) -> str:
        candidates = [task_config.scene_asset_path]
        fallback_path = getattr(task_config, 'scene_asset_fallback_path', None)
        if fallback_path:
            candidates.append(fallback_path)

        errors: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                resolved_path = cls._resolve_isaac_asset_path(candidate)
                if cls._scene_asset_exists(resolved_path):
                    return resolved_path
                errors.append(f'{candidate} -> {resolved_path} not found')
            except Exception as exc:
                errors.append(f'{candidate}: {exc}')

        raise FileNotFoundError('No loadable scene asset found. Tried: ' + '; '.join(errors))

    def add(self, target: any):
        """See `IScene.add` for documentation."""
        if hasattr(target, 'initialize') and hasattr(target, 'unwrap'):
            # TODO: Implement initialize method on IArticulation._articulation to make
            # 'self._scene._scene_registry.add_articulated_system' -> 'self._scene.add'
            self._scene._scene_registry.add_articulated_system(name=target.name, articulated_system=target)
        elif hasattr(target, 'unwrap'):
            self._scene.add(target.unwrap())
        else:
            # For instance of isaac-sim native classes
            self._scene.add(target)

    def remove(self, target: any, registry_only: bool = False):
        """See `IScene.remove` for documentation."""
        self._scene.remove_object(name=target, registry_only=registry_only)

    def remove_prim_path(self, prim_path: str) -> None:
        """Delete an exact USD subtree instead of inferring it from a registry object."""
        try:
            from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
        except ImportError:
            from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid

        if prim_path and is_prim_path_valid(prim_path):
            delete_prim(prim_path)

    def flush_updates(self) -> None:
        """Let USD, Hydra, and PhysX consume deletions before paths are reused."""
        try:
            from isaacsim.core.utils.stage import update_stage
        except ImportError:
            from omni.isaac.core.utils.stage import update_stage

        update_stage()

    def object_exists(self, target: any) -> bool:
        """See `IScene.object_exists` for documentation."""
        return self._scene.object_exists(target)

    def get(self, target: any) -> IRigidBody:
        """See `IScene.get` for documentation."""
        object = self._scene.get_object(target)
        return IRigidBody.create(prim_path=object.prim_path, name=object.prim_path)

    def unwrap(self):
        """See `IScene.unwrap` for documentation."""
        return self._scene
