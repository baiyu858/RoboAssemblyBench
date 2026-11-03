"""Simulator-independent task geometry, bounded admittance and success gates.

Quaternions use wxyz, lengths metres, angles radians, forces N, torques N m.
These defaults are experiment settings, not measured manufacturing clearances.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .geometry import (
    normalize_quat, quat_conjugate, quat_multiply, quat_rotate, relative_pose,
)

OBSERVATION_VERSION = 'contact-insertion-55-v1'
OBSERVATION_SIZE = 55


@dataclass(frozen=True)
class InsertionConfig:
    version: int = 3
    physics_hz: int = 240
    decimation: int = 4
    translation_step: float = 0.0002
    rotation_step: float = math.radians(0.1)
    guide_translation_step: float = 0.00008
    guide_rotation_step: float = math.radians(0.02)
    position_lead: float = 0.0005
    rotation_lead: float = math.radians(0.5)
    joint_step: float = 0.008
    joint_tracking_limit: float = 0.10
    force_filter_tau: float = 0.02
    # Diagonal Cartesian admittance M*x_dd + B*x_d + K*x = measured wrench.
    translation_mass: float = 0.5
    translation_damping: float = 80.0
    translation_stiffness: float = 800.0
    rotation_mass: float = 0.02
    rotation_damping: float = 1.5
    rotation_stiffness: float = 10.0
    compliance_position_limit: float = 0.002
    compliance_rotation_limit: float = math.radians(2.0)
    compliance_linear_speed: float = 0.012
    compliance_angular_speed: float = 0.10
    insertion_force_soft: float = 8.0
    insertion_force_hard: float = 30.0
    insertion_torque_hard: float = 2.0
    base_position_limit: float = 0.003
    base_rotation_limit: float = math.radians(5.0)
    grasp_position_limit: float = 0.003
    grasp_rotation_limit: float = math.radians(5.0)
    seated_axial_tolerance: float = 0.001
    seated_lateral_tolerance: float = 0.0005
    seated_rotation_tolerance: float = math.radians(1.0)
    seated_stable_steps: int = 96  # physics steps = 0.4 simulated seconds
    horizon: int = 1200  # 20 simulated seconds at 60 Hz
    reset_warmup_steps: int = 120

    def __post_init__(self):
        if self.version != 3:
            raise ValueError('This controller requires version 3')
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('physics_hz', 'decimation', 'horizon', 'reset_warmup_steps', 'seated_stable_steps'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.insertion_force_soft >= self.insertion_force_hard:
            raise ValueError('Insertion soft force limit must be below the hard limit')


def virtual_reference_limits(config):
    """Allow the virtual spring to deflect; bound the final arm target separately.

    Clamping the nominal reference to 0.5 mm before adding a 2 mm compliance
    offset would prevent a steady insertion load from being overcome at all.
    The published nominal+offset target still uses the original 0.5 mm limit.
    """
    return (config.position_lead + config.compliance_position_limit,
            config.rotation_lead + config.compliance_rotation_limit)


def geometric_guide(measurement, config, *, alignment_deadband=False):
    """Align before feeding; do not normalize away a small lateral correction.

    This is nominal guidance. The residual can still back off or rotate;
    force/slip arbitration remains a separate safety gate.
    """
    error = vector(measurement['error'], 3)
    angle = vector(measurement['rotation_error'], 3)
    angular_gate = config.seated_rotation_tolerance / (4. if alignment_deadband else 2.)
    translation = np.r_[clip_norm(error[:2], config.guide_translation_step), 0.]
    rotation = clip_norm(angle, config.guide_rotation_step)
    if alignment_deadband:
        if np.linalg.norm(error[:2]) <= config.seated_lateral_tolerance / 2.:
            translation[:2] = 0.
        if np.linalg.norm(angle) <= angular_gate:
            rotation[:] = 0.
    if (np.linalg.norm(error[:2]) <= config.seated_lateral_tolerance
            and np.linalg.norm(angle) <= angular_gate):
        translation[2] = np.clip(error[2], -config.guide_translation_step, config.guide_translation_step)
    return np.r_[clip_norm(translation, config.guide_translation_step),
                 rotation]


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'Expected {size} finite values')
    return result


def rotation_vector(quaternion):
    q = normalize_quat(vector(quaternion, 4))
    q = q if q[0] >= 0 else -q
    sine = np.linalg.norm(q[1:])
    return 2.0 * q[1:] if sine < 1e-12 else q[1:] * (2.0 * math.atan2(sine, q[0]) / sine)


def rotation_quaternion(value):
    value = vector(value, 3)
    angle = np.linalg.norm(value)
    return normalize_quat(np.r_[1.0, value / 2]) if angle < 1e-12 else np.r_[
        math.cos(angle / 2), value * math.sin(angle / 2) / angle]


def clip_norm(value, maximum):
    value = np.asarray(value, dtype=float)
    return value * min(1., maximum / max(float(np.linalg.norm(value)), 1e-15))


def basis_for_axis(axis):
    """Columns are lateral, lateral, insertion-axis unit vectors."""
    z = vector(axis, 3)
    norm = np.linalg.norm(z)
    if norm < 1e-12:
        raise ValueError('Insertion axis cannot be zero')
    z = z / norm
    helper = np.eye(3)[np.argmin(abs(z))]
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


class InsertionGeometry:
    def __init__(self, *, base_pose, entry_pose, goal_pose, insertion_axis=None, position_reference_point=None):
        self.base_anchor = tuple(np.array(x, dtype=float, copy=True) for x in base_pose)
        self.goal_local = relative_pose(*base_pose, *goal_pose)
        entry_local = relative_pose(*base_pose, *entry_pose)
        self.position_reference_point = (None if position_reference_point is None
                                         else vector(position_reference_point, 3).copy())
        goal_point = self._position(*self.goal_local)
        entry_point = self._position(*entry_local)
        self.basis = basis_for_axis(goal_point - entry_point if insertion_axis is None else insertion_axis)
        self.path_length = float(np.linalg.norm(goal_point - entry_point))
        if self.path_length < 1e-9:
            raise ValueError('Insertion entry and goal must define a nonzero local travel')

    def _position(self, position, orientation):
        if self.position_reference_point is None:
            return np.asarray(position)
        return np.asarray(position) + quat_rotate(orientation, self.position_reference_point)

    def world_basis(self, base_pose):
        return np.column_stack([quat_rotate(base_pose[1], self.basis[:, i]) for i in range(3)])

    def measure(self, part_pose, base_pose):
        position, orientation = relative_pose(*base_pose, *part_pose)
        error = self.basis.T @ (self._position(*self.goal_local) - self._position(position, orientation))
        angle = self.basis.T @ rotation_vector(quat_multiply(self.goal_local[1], quat_conjugate(orientation)))
        base_delta = np.asarray(base_pose[0]) - self.base_anchor[0]
        base_angle = rotation_vector(quat_multiply(base_pose[1], quat_conjugate(self.base_anchor[1])))
        world_basis = self.world_basis(base_pose)
        return {
            'error': error, 'rotation_error': angle,
            'axial_remaining': float(error[2]),
            'lateral_error': float(np.linalg.norm(error[:2])),
            'angle_error': float(np.linalg.norm(angle)),
            'base_position_drift': float(np.linalg.norm(base_delta)),
            'base_rotation_drift': float(np.linalg.norm(base_angle)),
            'base_delta': world_basis.T @ base_delta,
            'base_angle': world_basis.T @ base_angle,
            'progress': float(1. - error[2] / self.path_length),
        }


def pair_wrench(observation, origin):
    """Normal + friction wrench ON the probed part, about the supplied world point.

    The filtered matrix is the normal resultant. Friction comes from the
    separate friction anchors, never from the gripper's clamp-force readings.
    Refuse missing/truncated data instead of inventing a zero-force sample.
    """
    separated = bool(observation.get('valid') and observation.get('details_valid')
                     and not observation.get('contacts')
                     and np.linalg.norm(vector(observation.get('force_world', [np.nan] * 3), 3)) < 1e-12)
    if not (observation.get('valid') and observation.get('details_valid')
            and not observation.get('details_at_capacity')
            and (observation.get('friction_valid') or separated)):
        raise ValueError('Complete part/base normal and friction contact data are required')
    origin = vector(origin, 3)
    force = vector(observation['force_world'], 3).copy()
    torque = np.zeros(3)
    for contact in observation.get('contacts', []):
        f = vector(contact['normal_world'], 3) * float(contact['normal_force'])
        torque += np.cross(vector(contact['point_world'], 3) - origin, f)
    for contact in observation.get('friction_anchors', []):
        f = vector(contact['force_world'], 3)
        force += f
        torque += np.cross(vector(contact['point_world'], 3) - origin, f)
    return np.r_[force, torque]


def validate_torque_soft_limit(value, config):
    """Optional post-policy guard; it does not change the trained action scale."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0 < value < config.insertion_torque_hard):
        raise ValueError('torque_soft_limit must be positive and below insertion_torque_hard')
    return None if value is None else float(value)


def effective_torque_soft_limit(requested, trained, config):
    """Deployment may tighten, but cannot remove a recorded training guard."""
    values = [validate_torque_soft_limit(x, config) for x in (requested, trained)]
    return min((x for x in values if x is not None), default=None)


def validate_torque_hard_limit(value, config, soft_limit=None):
    """Explicit execution stop override; trained control/taper parameters stay fixed."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value <= 0):
        raise ValueError('torque_hard_limit must be positive and finite')
    effective = config.insertion_torque_hard if value is None else float(value)
    if soft_limit is not None and effective <= soft_limit:
        raise ValueError('torque_hard_limit must exceed torque_soft_limit')
    return effective


def arbitrate_delta(delta, controls, wrench_task, config, *, torque_soft_limit=None, safety_wrench=None):
    """Allow bounded axial backoff during mild force limiting; never force feed.

    Hard force, missing contact, slip and probe failures remain terminal. Under
    a soft recovery, lateral/rotational exploration also stops until ready.
    """
    delta = vector(delta, 6).copy()
    wrench_task = vector(wrench_task, 6)
    if not controls or any(c.get('failure') for c in controls):
        return np.zeros(6), 'grasp_failure'
    if any(not c.get('motion_ready') for c in controls):
        safe_backoff = all(c.get('motion_ready') or (
            c.get('mode') in ('force_limited', 'force_relief')
            and c.get('weak_force', 0.) >= c.get('low_force', 6.)
            and c.get('strong_force', math.inf) < c.get('unload_force', 18.)
            and len(c.get('forces', [])) == 2
            and all(math.isfinite(f) and f >= c.get('low_force', 6.) for f in c['forces'])
            and not c.get('slipping') and c.get('probe_valid')) for c in controls)
        result = np.zeros(6)
        if safe_backoff:
            result[2] = min(0., delta[2])
            if any(c.get('mode') == 'force_relief' and not c.get('motion_ready') for c in controls):
                return result, 'compliant_relief'
        return result, 'retreat_only' if safe_backoff else 'grasp_recovery_hold'
    soft = validate_torque_soft_limit(torque_soft_limit, config)
    if soft is not None:
        # Use the same physical wrist origin as the raw hard guard, not the
        # virtual pin tip used by admittance. A long pin has a different arm.
        torque = float(np.linalg.norm(vector(safety_wrench, 6)[3:]))
        if torque >= soft:
            scale = np.clip((config.insertion_torque_hard - torque)
                            / (config.insertion_torque_hard - soft), 0., 1.)
            delta[2] = min(0., delta[2])
            delta[:2] *= scale
            delta[3:] *= scale
            return delta, 'insertion_torque_limit'
    if np.linalg.norm(wrench_task[:3]) > config.insertion_force_soft:
        delta[2] = min(0., delta[2])
        return delta, 'insertion_force_limit'
    return delta * min(c.get('motion_scale', 1.) for c in controls), 'policy'


def bounded_relief_step(candidate, wrench_task, config):
    """Restrict an admittance correction to unloading work and one physics step.

    The candidate is relative to the previous published target, preserving its
    holding bias. Policy lateral/rotational exploration is already suppressed.
    No forward feed; tangential/rotational components may only follow reaction.
    """
    result = vector(candidate, 6).copy()
    wrench = vector(wrench_task, 6)
    result[2] = min(0., result[2])
    for selection in (slice(0, 2), slice(3, 6)):
        reaction = wrench[selection]
        norm_sq = float(np.dot(reaction, reaction))
        if norm_sq < 1e-12:
            result[selection] = 0.
        else:
            loading = min(0., float(np.dot(result[selection], reaction)))
            result[selection] -= loading / norm_sq * reaction
    result[:3] = clip_norm(result[:3], config.compliance_linear_speed / config.physics_hz)
    result[3:] = clip_norm(result[3:], config.compliance_angular_speed / config.physics_hz)
    return result


class BoundedAdmittance:
    def __init__(self, config):
        self.config = config
        self.offset = np.zeros(6)
        self.velocity = np.zeros(6)
        self.filtered_wrench = np.zeros(6)

    def update(self, wrench, dt):
        c = self.config
        wrench = vector(wrench, 6)
        self.filtered_wrench += -math.expm1(-dt / c.force_filter_tau) * (wrench - self.filtered_wrench)
        mass = np.r_[np.full(3, c.translation_mass), np.full(3, c.rotation_mass)]
        damping = np.r_[np.full(3, c.translation_damping), np.full(3, c.rotation_damping)]
        stiffness = np.r_[np.full(3, c.translation_stiffness), np.full(3, c.rotation_stiffness)]
        # Backward Euler avoids an explicit-step damping instability at 240 Hz.
        self.velocity = (mass * self.velocity + dt * (self.filtered_wrench - stiffness * self.offset)) / (
            mass + dt * damping + dt * dt * stiffness)
        self.velocity[:3] = clip_norm(self.velocity[:3], c.compliance_linear_speed)
        self.velocity[3:] = clip_norm(self.velocity[3:], c.compliance_angular_speed)
        candidate = self.offset + dt * self.velocity
        for selection, limit in ((slice(0, 3), c.compliance_position_limit),
                                 (slice(3, 6), c.compliance_rotation_limit)):
            bounded = clip_norm(candidate[selection], limit)
            if np.linalg.norm(candidate[selection] - bounded) > 1e-15:
                # Saturated compliance must not accumulate hidden velocity.
                self.velocity[selection] = 0.
            self.offset[selection] = bounded
        return self.offset.copy()


def safety_failure(measurement, grasp_samples, wrench, config, *, torque_hard_limit=None):
    if not np.all(np.isfinite(vector(wrench, 6))):
        return 'insertion_wrench_invalid'
    if np.linalg.norm(wrench[:3]) > config.insertion_force_hard:
        return 'insertion_force_hard_limit'
    if np.linalg.norm(wrench[3:]) > validate_torque_hard_limit(torque_hard_limit, config):
        return 'insertion_torque_hard_limit'
    if measurement['base_position_drift'] > config.base_position_limit:
        return 'base_position_drift'
    if measurement['base_rotation_drift'] > config.base_rotation_limit:
        return 'base_rotation_drift'
    if measurement.get('support_position_drift', 0.) > config.base_position_limit:
        return 'support_position_drift'
    if measurement.get('support_rotation_drift', 0.) > config.base_rotation_limit:
        return 'support_rotation_drift'
    for grasp in grasp_samples:
        if not grasp.get('valid') and not grasp.get('recovering_contact'):
            return 'physical_grasp_invalid'
        if grasp['position_slip'] > config.grasp_position_limit:
            return 'grasp_position_slip'
        if grasp['orientation_slip'] > config.grasp_rotation_limit:
            return 'grasp_rotation_slip'
    return None


def seated(measurement, config):
    return bool(abs(measurement['axial_remaining']) <= config.seated_axial_tolerance
                and measurement['lateral_error'] <= config.seated_lateral_tolerance
                and measurement['angle_error'] <= config.seated_rotation_tolerance
                and measurement['base_position_drift'] <= config.base_position_limit
                and measurement['base_rotation_drift'] <= config.base_rotation_limit)


def seated_load_transfer_ready(controls, grasps):
    """Permit release of a seated part, never renewed arm motion on a weak grip.

    All support arms keep their full motion-readiness gate. The inserting
    hand may be force limited only while its real bilateral grasp remains
    valid, below the unloading limit and without slip/probe/control faults.
    Pose stability and all hard guards are checked separately by the caller.
    """
    if not controls or len(controls) != len(grasps) or not all(g.get('valid') for g in grasps):
        return False
    if any(c.get('failure') for c in controls):
        return False
    if not all(c.get('motion_ready') for c in controls[1:]):
        return False
    moving = controls[0]
    if moving.get('motion_ready'):
        return True
    forces = moving.get('forces', [])
    return bool(not moving.get('failure') and moving.get('probe_valid') and moving.get('force_within_limit')
                and not moving.get('slipping') and moving.get('mode') in ('preload', 'force_limited', 'force_relief', 'hold')
                and len(forces) == 2 and all(math.isfinite(f) and f > 0 for f in forces)
                and max(forces) < moving.get('unload_force', 18.))


def insertion_reward(before, after, wrench, action, previous_action, failure, local_seated, config=None):
    """Dense local progress cannot be earned by translating the supported base."""
    c = config or InsertionConfig()
    progress = float(np.clip(after['progress'] - before['progress'], -.05, .05))
    result = (30. * progress - .01 * after['lateral_error'] / (2 * c.seated_lateral_tolerance)
              - .005 * after['angle_error'] / c.seated_rotation_tolerance
              - .02 * (after['base_position_drift'] / c.base_position_limit) ** 2
              - .01 * (after['base_rotation_drift'] / c.base_rotation_limit) ** 2
              - .002 * max(0., float(np.linalg.norm(wrench[:3])) - c.insertion_force_soft / 2.) ** 2
              - .001 * float(np.sum((np.asarray(action) - previous_action) ** 2)) - .001)
    return float(result + (10. if local_seated else 0.) - (10. if failure else 0.))
