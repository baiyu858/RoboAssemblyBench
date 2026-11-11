"""Execution-only unloading of a pin; checkpoint inputs and gains stay intact."""
import numpy as np

from .control import bounded_relief_step, rotation_quaternion, rotation_vector
from .geometry import quat_conjugate, quat_multiply


def relief_enabled(value, soft_limit):
    if type(value) is not bool:
        raise ValueError('torque_relief must be a boolean')
    if value and soft_limit is None:
        raise ValueError('torque_relief requires torque_soft_limit')
    return value


def unloading_grip_ready(controls, grasps):
    """Raw bilateral contact can unload during filtered-force/slip prewarning.

    This never authorizes feed. Hard frozen-reference slip and contact guards
    run before this predicate, on the same physics-step sample.
    """
    if not controls or len(controls) != len(grasps):
        return False
    for ctrl, grasp in zip(controls, grasps):
        forces = np.asarray(ctrl.get('forces', []), dtype=float)
        if (
            not grasp.get('valid')
            or grasp.get('recovering_contact')
            or ctrl.get('failure')
            or not ctrl.get('probe_valid')
            or forces.shape != (2,)
            or not np.all(np.isfinite(forces))
            or np.min(forces) < ctrl.get('low_force', 6.0)
            or ctrl.get('strong_force', np.max(forces)) >= ctrl.get('unload_force', 18.0)
        ):
            return False
    return True


class TorqueRelief:
    def __init__(self, config, soft_limit):
        self.config, self.soft_limit = config, soft_limit
        self.active = False
        self.anchor = None

    def update(self, wrist_torque):
        # Release hysteresis prevents alternating feed/unload at the soft limit.
        threshold = self.soft_limit * (0.75 if self.active else 1.0)
        self.active = wrist_torque >= threshold
        if not self.active:
            self.anchor = None
        return self.active

    def target(self, command, basis, wrench_task):
        c = self.config
        if self.anchor is None:
            self.anchor = tuple(np.asarray(x).copy() for x in command)
        # Follow measured reaction, not the residual's possibly loading action.
        # Axial backoff is always negative in the insertion frame.
        candidate = np.r_[
            np.asarray(wrench_task[:3]) / c.translation_damping / c.physics_hz,
            np.asarray(wrench_task[3:]) / c.rotation_damping / c.physics_hz,
        ]
        candidate[2] = min(candidate[2], -0.5 * c.compliance_linear_speed / c.physics_hz)
        step = bounded_relief_step(candidate, wrench_task, c)

        def pose(fraction):
            return (
                command[0] + basis @ step[:3] * fraction,
                quat_multiply(rotation_quaternion(basis @ step[3:] * fraction), command[1]),
            )

        def within(pose):
            return (
                np.linalg.norm(pose[0] - self.anchor[0]) <= c.compliance_position_limit + 1e-12
                and np.linalg.norm(rotation_vector(quat_multiply(pose[1], quat_conjugate(self.anchor[1]))))
                <= c.compliance_rotation_limit + 1e-12
            )

        # Bound the episode as well as each command. Scaling keeps the reaction
        # direction and prohibits accumulated forward pressure or wandering.
        fraction = 1.0
        if not within(pose(fraction)):
            low, high = 0.0, 1.0
            for _ in range(30):
                middle = (low + high) / 2.0
                if within(pose(middle)):
                    low = middle
                else:
                    high = middle
            fraction = low
        if np.linalg.norm(step * fraction) < 1e-10:
            raise ValueError('insertion_torque_relief_travel_limit')
        return pose(fraction)
