"""Apply phase-specific friction to existing native rigid-body materials.

No USD APIs, poses, velocities or collision filters are changed at runtime.
"""
import math

import numpy as np

KEY = 'object_friction'


def friction_settings(value):
    if not isinstance(value, dict) or set(value) != {'static_friction', 'dynamic_friction'}:
        raise ValueError('Friction requires static_friction and dynamic_friction')
    for name, coefficient in value.items():
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not math.isfinite(coefficient)
            or coefficient < 0
        ):
            raise ValueError(f'Invalid friction coefficient: {name}')
    if value['dynamic_friction'] > value['static_friction']:
        raise ValueError('dynamic_friction must not exceed static_friction')
    return {name: float(coefficient) for name, coefficient in value.items()}


def _settings(phase):
    value = phase.get(KEY, {})
    if not isinstance(value, dict) or any(not isinstance(n, str) or not n for n in value):
        raise ValueError(f'{KEY} must map object names to friction settings')
    return {name: friction_settings(settings) for name, settings in value.items()}


def friction_through(phases, resume_phase):
    """Restore the current recipe's material policy when resuming a snapshot."""
    result = {}
    for phase in phases:
        result.update(_settings(phase))
        if phase['name'] == resume_phase:
            return result
    raise ValueError(f'Cannot restore friction for missing phase {resume_phase}')


def apply_phase_friction(task, phase):
    settings = _settings(phase)
    if not settings:
        return
    for name in settings:
        if (
            name not in task.objects
            or task._is_static_environment_object(name)
            or not getattr(task.objects[name].config, 'rigid_body', False)
        ):
            raise ValueError(f'{KEY} requires a dynamic task object: {name}')
    import omni.physics.tensors as tensors

    simulation = tensors.create_simulation_view('numpy')
    for name, material in settings.items():
        task._resolve_object(name)
        view = simulation.create_rigid_body_view(str(task._object_prims[name].GetPath()))
        if view.count != 1 or view.max_shapes < 1:
            raise RuntimeError(f'Cannot resolve native material for {name}')
        before = view.get_material_properties().copy()
        expected = before.copy()
        expected[:, :, 0] = material['static_friction']
        expected[:, :, 1] = material['dynamic_friction']
        if not np.array_equal(before, expected):
            view.set_material_properties(expected, np.array([0], dtype=np.uint32))
        actual = view.get_material_properties().copy()
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f'Native friction readback mismatch: {name}')
        if not hasattr(task, '_friction_policy_history'):
            task._friction_policy_history = []
        task._friction_policy_history.append(
            dict(
                object=name,
                phase=phase['name'],
                step=int(task.step_counter),
                previous_materials=before.tolist(),
                applied_materials=actual.tolist(),
            )
        )
