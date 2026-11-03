"""Live multi-bore alignment and a rigid-grasp command envelope, in metres.

Only poses are read. The predicted shaft uses a rigid transform of the measured
part, never a state write or a replacement of the frozen grasp reference.
"""
import math

import numpy as np

from .control import basis_for_axis, rotation_quaternion, rotation_vector, vector
from .geometry import compose_pose, quat_conjugate, quat_multiply, quat_rotate, relative_pose


def alignment_settings(value):
    if value is None:
        return None
    allowed = {'shaft_origin_m', 'shaft_axis', 'sections', 'radial_clearance_m',
               'margin_m', 'linear_speed', 'angular_speed'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('axis_alignment has unknown settings or is not a mapping')
    result = dict(radial_clearance_m=.00025, margin_m=.00008, linear_speed=.002, angular_speed=.04)
    result.update(value)
    for key in ('radial_clearance_m', 'margin_m', 'linear_speed', 'angular_speed'):
        x = result[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f'axis_alignment.{key} must be positive and finite')
    if result['margin_m'] >= result['radial_clearance_m']:
        raise ValueError('axis_alignment margin must be below radial clearance')
    result['shaft_origin_m'] = vector(result['shaft_origin_m'], 3).tolist()
    result['shaft_axis'] = basis_for_axis(result['shaft_axis'])[:, 2].tolist()
    sections = result.get('sections')
    if not isinstance(sections, (list, tuple)) or len(sections) < 2:
        raise ValueError('axis_alignment requires at least two bore sections')
    result['sections'] = []
    for section in sections:
        if not isinstance(section, dict) or set(section) != {'object', 'center_m', 'axis'}:
            raise ValueError('axis_alignment sections require object, center_m and axis')
        if not isinstance(section['object'], str) or not section['object']:
            raise ValueError('axis_alignment section object must be named')
        result['sections'].append(dict(object=section['object'], center_m=vector(section['center_m'], 3).tolist(),
                                       axis=basis_for_axis(section['axis'])[:, 2].tolist()))
    return result


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class BoreAlignment:
    def __init__(self, settings, binding):
        self.settings = alignment_settings(settings)
        self.binding = binding
        if any(s['object'] not in binding.objects for s in self.settings['sections']):
            raise ValueError('axis_alignment sections must reference declared insertion objects')

    @property
    def allowance(self):
        return self.settings['radial_clearance_m'] - self.settings['margin_m']

    def measure(self, moving, poses, pivot, basis):
        c = self.settings
        origin = moving[0] + quat_rotate(moving[1], c['shaft_origin_m'])
        axis = quat_rotate(moving[1], c['shaft_axis'])
        errors, jacobians, sections = [], [], []
        for spec in c['sections']:
            p, q = poses[spec['object']]
            center = p + quat_rotate(q, spec['center_m'])
            normal = quat_rotate(q, spec['axis'])
            denominator = float(np.dot(normal, axis))
            if abs(denominator) < .5:
                raise ValueError('shaft axis is not within 60 degrees of the bore')
            hit = origin + axis * (np.dot(normal, center - origin) / denominator)
            plane = basis_for_axis(normal)[:, :2]
            error = plane.T @ (hit - center)
            projection = np.eye(3) - np.outer(axis, normal) / denominator
            jacobian = plane.T @ projection @ np.column_stack((basis[:, :2], -_skew(hit - pivot) @ basis[:, :2]))
            errors.append(error)
            jacobians.append(jacobian)
            sections.append(dict(object=spec['object'], center_local_m=spec['center_m'],
                                 axis_error_m=float(np.linalg.norm(error)),
                                 intersection_world_m=hit, center_world_m=center))
        # Scaling makes millimetres and milliradians comparable for the fit.
        scale = np.diag([1., 1., 20., 20.])
        j = np.vstack(jacobians) @ scale
        if np.linalg.matrix_rank(j) < 4:
            raise ValueError('axis_alignment needs distinct axial section heights')
        correction = scale @ np.linalg.lstsq(j, -np.concatenate(errors), rcond=None)[0]
        world = np.r_[basis[:, :2] @ correction[:2], basis[:, :2] @ correction[2:]]
        return dict(max_error_m=max(s['axis_error_m'] for s in sections),
                    allowance_m=self.allowance, ready=all(s['axis_error_m'] <= self.allowance for s in sections),
                    correction_world=world, sections=sections)

    def bounded_correction(self, sample, dt):
        raw = sample['correction_world']
        # A common scale preserves the fit's translation/rotation ratio.
        factor = min(1., self.settings['linear_speed'] * dt / max(np.linalg.norm(raw[:3]), 1e-15),
                     self.settings['angular_speed'] * dt / max(np.linalg.norm(raw[3:]), 1e-15))
        return raw * factor

    def constrain_target(self, *, target, current, moving, poses, basis, sample, dt, relief_start=None):
        """Limit residual/admittance targets too, not just geometric guidance."""
        local = relative_pose(*current, *moving)

        def predicted(pose):
            return self.measure(compose_pose(*pose, *local), poses, pose[0], basis)['max_error_m']

        if relief_start is not None:
            # Only scale the already validated unloading segment. Do not add
            # alignment movement against the reaction or lose holding bias.
            limit = max(self.allowance, predicted(relief_start))
            angle = rotation_vector(quat_multiply(target[1], quat_conjugate(relief_start[1])))
            for fraction in (1., .5, .25, .125, .0625):
                candidate = (relief_start[0] + fraction * (target[0] - relief_start[0]),
                             quat_multiply(rotation_quaternion(angle * fraction), relief_start[1]))
                if predicted(candidate) <= limit + 1e-10:
                    return candidate, fraction < 1.
            return relief_start, True
        limit = max(self.allowance, sample['max_error_m'])
        if predicted(target) <= limit + 1e-10:
            return target, False
        correction = self.bounded_correction(sample, dt)
        axial = float(np.dot(basis[:, 2], target[0] - current[0]))
        if not sample['ready']:
            axial = min(0., axial)
        safe = (current[0] + correction[:3] + basis[:, 2] * axial,
                quat_multiply(rotation_quaternion(correction[3:]), current[1]))
        if predicted(safe) > limit + 1e-10:
            safe = (current[0].copy(), current[1].copy())
        angle = rotation_vector(quat_multiply(target[1], quat_conjugate(safe[1])))
        for fraction in (1., .5, .25, .125, .0625):
            candidate = (safe[0] + fraction * (target[0] - safe[0]),
                         quat_multiply(rotation_quaternion(angle * fraction), safe[1]))
            if predicted(candidate) <= limit + 1e-10:
                return candidate, True
        return safe, True
