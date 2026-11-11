"""Choose a shorter carry orientation for explicitly free-spinning round shafts.

Rotate about the CAD shaft centre, not the asset origin. This transforms goal
poses only; pickup poses, grasp transforms and physical bodies are untouched.
"""
from __future__ import annotations

import math

import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import (
    normalize_quat,
    quat_multiply,
    quat_rotate,
)


def equivalent_axial_pose(position, orientation, plan):
    if plan is None:
        return list(position), list(orientation)
    q = normalize_quat(orientation)
    rotated = normalize_quat(quat_multiply(q, plan['twist']))
    point = np.asarray(plan['reference_point'], dtype=float)
    centre = np.asarray(position, dtype=float) + quat_rotate(q, point)
    return (centre - quat_rotate(rotated, point)).tolist(), rotated.tolist()


def _angle(a, b):
    return 2 * math.acos(float(np.clip(abs(np.dot(normalize_quat(a), normalize_quat(b))), 0.0, 1.0)))


def axial_transport_plan(*, task, spec, pickup_orientation, assembly_orientation):
    requested = spec.get('physical_free_axial_transport_parts', [])
    if not requested:
        return {}
    if not spec.get('continuous_physics') or not isinstance(requested, list):
        raise ValueError('physical_free_axial_transport_parts requires a list and continuous_physics')
    parts = {str(p['part_id']): p for p in task['parts']}
    steps = {str(s['move_part']): s for s in task['assembly_steps']}
    result = {}
    for value in requested:
        part_id = str(value)
        if part_id in result or part_id not in steps:
            raise ValueError(f'physical_free_axial_transport_parts: invalid/duplicate move part {part_id}')
        settings = spec.get('precision_insertion', {}).get(part_id, {})
        axial = settings.get('released_axial_rotation') or {}
        axis = np.asarray(axial.get('axis', []), dtype=float)
        point = np.asarray(spec.get('physical_success_reference_points_m', {}).get(part_id, []), dtype=float)
        spin_limit = float(axial.get('twist_tolerance_rad', 0.0))
        if (
            axis.shape != (3,)
            or point.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not np.all(np.isfinite(point))
            or np.linalg.norm(axis) < 1e-12
            or not math.isfinite(spin_limit)
            or spin_limit < math.pi - 1e-10
        ):
            raise ValueError(f'part {part_id}: free axial transport requires a shaft axis, CAD centre and free spin')
        axis /= np.linalg.norm(axis)
        entry = steps[part_id]['disassembly_path'][-1]['orientation']
        source = normalize_quat(quat_multiply(pickup_orientation, parts[part_id]['pickup_orientation']))
        target = normalize_quat(quat_multiply(assembly_orientation, entry))
        inverse = np.asarray(target).copy()
        inverse[1:] *= -1
        relative = quat_multiply(inverse, source)
        twist = np.r_[relative[0], axis * np.dot(relative[1:], axis)]
        # At the swing/twist singularity every axial angle costs the same.
        twist = normalize_quat(twist) if np.linalg.norm(twist) > 1e-10 else np.array([1.0, 0.0, 0.0, 0.0])
        goal = normalize_quat(quat_multiply(target, twist))
        result[part_id] = dict(
            axis=axis.tolist(),
            reference_point=point.tolist(),
            twist=twist.tolist(),
            original_carry_angle_rad=_angle(source, target),
            selected_carry_angle_rad=_angle(source, goal),
        )
    return result
