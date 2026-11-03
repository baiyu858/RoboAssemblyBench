"""Keep initialization settling separate from active rigid-body manipulation.

Only the numerical stabilization threshold changes. These actions never write
poses, velocities, grasp references, materials or the immutable scene flag.
"""
import math


KEY = 'object_stabilization_thresholds'


def _thresholds(value):
    if not isinstance(value, dict):
        raise ValueError(f'{KEY} must map object names to nonnegative thresholds')
    for name, threshold in value.items():
        if (not isinstance(name, str) or not name or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0):
            raise ValueError(f'Invalid {KEY} entry')
    return {name: float(threshold) for name, threshold in value.items()}


def annotate_active_parts(phases):
    """Disable freeze at each part's first loaded motion, preserving settling."""
    active = set()
    for phase in phases:
        skills = [phase.get('local_skill') or {}, *(phase.get('local_skills') or {}).values()]
        for skill in skills:
            name = skill.get('object') or skill.get('held_object')
            if (name and name not in active and skill.get('requires_held_object')
                    and skill.get('continuous_physics')):
                phase.setdefault(KEY, {})[name] = 0.0
                active.add(name)


def thresholds_through(phases, resume_phase):
    """Restore current recipe policy through a snapshot's entry phase."""
    result = {}
    for phase in phases:
        result.update(_thresholds(phase.get(KEY, {})))
        if phase['name'] == resume_phase:
            return result
    raise ValueError(f'Cannot restore stabilization for missing phase {resume_phase}')


def apply_phase_thresholds(task, phase):
    settings = _thresholds(phase.get(KEY, {}))
    for name in settings:
        if (name not in task.objects or task._is_static_environment_object(name)
                or not getattr(task.objects[name].config, 'rigid_body', False)):
            raise ValueError(f'{KEY} requires a dynamic task object: {name}')
    for name, threshold in settings.items():
        task._resolve_object(name)
        attr = task._object_prims[name].GetAttribute('physxRigidBody:stabilizationThreshold')
        if not attr.IsValid() or attr.Get() is None:
            raise RuntimeError(f'Stabilization was not configured before physics initialization: {name}')
        previous = float(attr.Get())
        if previous != threshold:
            if not attr.Set(threshold) or not math.isclose(float(attr.Get()), threshold, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError(f'Could not apply numerical stabilization threshold: {name}')
            task._stabilization_policy_history.append(dict(
                object=name, previous_threshold=previous, threshold=threshold,
                phase=phase['name'], step=int(task.step_counter)))
        task._object_stabilization_thresholds[name] = threshold
