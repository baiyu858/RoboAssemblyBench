"""Frozen-reference precision checks for bodies sharing one physical gripper."""
import math

import numpy as np


def precision_hold_settings(value):
    if value is None:
        return None
    defaults = dict(
        drift_limit_m=0.00015,
        alignment_limit_m=0.00035,
        preload_drift_m=0.000075,
        minimum_force_N=4.0,
        target_force_N=5.0,
        stable_steps=24,
        recovery_timeout_steps=480,
    )
    if not isinstance(value, dict) or set(value) - set(defaults) - {'reference_points_m', 'enabled'}:
        raise ValueError('Invalid compound precision_hold settings')
    if type(value.get('enabled', True)) is not bool:
        raise ValueError('precision_hold.enabled must be a boolean')
    if not value.get('enabled', True):
        return None
    result = {**defaults, **value}
    points = np.asarray(result.get('reference_points_m'), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError('precision_hold requires finite three-dimensional reference points')
    for key in defaults:
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'precision_hold.{key} must be positive and finite')
    for key in ('stable_steps', 'recovery_timeout_steps'):
        if type(result[key]) is not int:
            raise ValueError(f'precision_hold.{key} must be an integer')
    if (
        result['preload_drift_m'] >= result['drift_limit_m']
        or result['minimum_force_N'] >= result['target_force_N']
        or result['stable_steps'] >= result['recovery_timeout_steps']
    ):
        raise ValueError('Invalid precision preload, force or recovery ordering')
    result['reference_points_m'] = points.tolist()
    return result


def _points(pose, points):
    p, q = (np.asarray(x, dtype=float) for x in pose)
    q = q / np.linalg.norm(q)
    t = 2 * np.cross(q[1:], points)
    return p + points + q[0] * t + np.cross(q[1:], t)


def precision_hold_sample(config, state, *, pose, anchor, forces, contact, step):
    """Compare actual mating points to BOTH CAD alignment and capture history.

    A low force can reclose the jaws; a large relative drift pauses feed.
    Neither a phase change nor ongoing drift recaptures either reference.
    """
    if state.get('step') == step:
        return state['sample']
    if step < state.get('step', -1):
        state.clear()  # A reset changes counters, never the external pose anchor.
    elif step != state.get('step', step - 1) + 1:
        state['stable_steps'] = 0
    points = np.asarray(config['reference_points_m'])
    actual = _points(pose, points)
    original = _points((anchor['position'], anchor['orientation']), points)
    drift = float(np.max(np.linalg.norm(actual - original, axis=1)))
    alignment = float(np.max(np.linalg.norm(actual - points, axis=1)))
    valid = contact and all(np.isfinite(f) for f in forces)
    good = bool(
        valid
        and min(forces) >= config['minimum_force_N']
        and drift <= config['drift_limit_m']
        and alignment <= config['alignment_limit_m']
    )
    state['stable_steps'] = state.get('stable_steps', 0) + 1 if good else 0
    ready = state['stable_steps'] >= config['stable_steps']
    if ready:
        state.pop('unready_since', None)
    else:
        state.setdefault('unready_since', step)
    timed_out = not ready and step - state['unready_since'] >= config['recovery_timeout_steps']
    result = dict(
        ready=ready,
        drift_m=drift,
        alignment_error_m=alignment,
        drift_limit_m=config['drift_limit_m'],
        alignment_limit_m=config['alignment_limit_m'],
        preload_required=bool(valid and min(forces) < config['minimum_force_N']),
        preload_requested=bool(
            valid
            and min(forces) < config['target_force_N']
            and (min(forces) < config['minimum_force_N'] or drift >= config['preload_drift_m'])
        ),
        failure='support_precision_recovery_timeout' if timed_out else None,
    )
    state.update(step=step, sample=result)
    return result
