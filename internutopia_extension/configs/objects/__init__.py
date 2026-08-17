from typing import List, Optional, Tuple

from pydantic import BaseModel

from internutopia.core.config.object import ObjectCfg


class DynamicCubeCfg(ObjectCfg):
    type: Optional[str] = 'DynamicCube'
    color: Optional[Tuple[float, float, float]] = None
    mass: Optional[float] = None
    density: Optional[float] = None
    collider: Optional[bool] = True
    disable_gravity: Optional[bool] = None
    static_friction: Optional[float] = None
    dynamic_friction: Optional[float] = None
    restitution: Optional[float] = None


class CompoundCuboidPartCfg(BaseModel):
    name: str
    offset: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    color: Optional[Tuple[float, float, float]] = None


class DynamicCompoundCuboidCfg(ObjectCfg):
    type: Optional[str] = 'DynamicCompoundCuboid'
    color: Optional[Tuple[float, float, float]] = None
    parts: List[CompoundCuboidPartCfg]
    mass: Optional[float] = None
    density: Optional[float] = None
    collider: Optional[bool] = True
    static_friction: Optional[float] = None
    dynamic_friction: Optional[float] = None
    restitution: Optional[float] = None


class VisualCubeCfg(ObjectCfg):
    type: Optional[str] = 'VisualCube'
    color: Optional[List[float]] = None
    texture_path: Optional[str] = None
    texture_scale: Optional[Tuple[float, float]] = None
    texture_rotation_degrees: Optional[float] = None


class StaticCubeCfg(ObjectCfg):
    type: Optional[str] = 'StaticCube'
    color: Optional[List[float]] = None
    texture_path: Optional[str] = None
    texture_scale: Optional[Tuple[float, float]] = None
    texture_rotation_degrees: Optional[float] = None
    static_friction: Optional[float] = None
    dynamic_friction: Optional[float] = None
    restitution: Optional[float] = None


class UsdObjCfg(ObjectCfg):
    type: Optional[str] = 'UsdObject'
    usd_path: str
    collider: Optional[bool] = True
    auto_collider: Optional[bool] = True
    rigid_body: Optional[bool] = True
    mass: Optional[float] = None
    density: Optional[float] = None
    static_friction: Optional[float] = None
    dynamic_friction: Optional[float] = None
    restitution: Optional[float] = None
    linear_damping: Optional[float] = None
    angular_damping: Optional[float] = None
    sleep_threshold: Optional[float] = None
    stabilization_threshold: Optional[float] = None
    solver_position_iteration_count: Optional[int] = None
    solver_velocity_iteration_count: Optional[int] = None
