"""Explicit scene-wide contact solver options, applied before physics starts."""

ATTRIBUTES = {
    'enable_stabilization': 'physxScene:enableStabilization',
    'enable_external_forces_every_iteration': 'physxScene:enableExternalForcesEveryIteration',
}


def validate_physics_scene_settings(settings):
    if not isinstance(settings, dict) or set(settings) - set(ATTRIBUTES):
        raise ValueError('Unknown physics_scene settings; only explicit solver flags are supported.')
    if any(type(value) is not bool for value in settings.values()):
        raise ValueError('physics_scene settings must be booleans.')
    return dict(settings)


def shared_physics_scene_settings(task_configs):
    settings = [validate_physics_scene_settings(getattr(cfg, 'physics_scene', {})) for cfg in task_configs]
    if settings and any(value != settings[0] for value in settings[1:]):
        raise ValueError('Tasks sharing one physics scene must request identical physics_scene settings.')
    return settings[0] if settings else {}


def apply_physics_scene_settings(context, stage, settings):
    """Require authored readback; a failed setting must not silently fall back."""
    settings = validate_physics_scene_settings(settings)
    if not settings:
        return {}
    prim = stage.GetPrimAtPath(context.prim_path)
    for name, value in settings.items():
        attr = prim.GetAttribute(ATTRIBUTES[name])
        if not attr.IsValid() or not attr.Set(value) or attr.Get() is not value:
            raise RuntimeError(f'Could not apply physics_scene.{name} before physics initialization.')
    return dict(settings)
