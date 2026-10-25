"""Rate/acceleration-limited SO(3) reference with measured-pose anti-windup.

Quaternions are wxyz; angular velocities are world-frame radians/second.
This governs a reference, not the simulated body's velocity or contact torque.
The caller commits the returned state only after accepting the motion command.
"""
from __future__ import annotations

import math

import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import quat_multiply


def _quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError('Rotation command requires a finite nonzero wxyz quaternion.')
    return q / np.linalg.norm(q)


def rotation_vector(target, reference):
    """Shortest world-frame rotation from reference to target (sign invariant)."""
    inverse = _quaternion(reference).copy()
    inverse[1:] *= -1
    delta = _quaternion(quat_multiply(_quaternion(target), inverse))
    if delta[0] < 0:
        delta = -delta
    sine = float(np.linalg.norm(delta[1:]))
    return (2.0 * delta[1:] if sine < 1e-12 else
            delta[1:] * (2.0 * math.atan2(sine, delta[0]) / sine))


def _advance(orientation, displacement):
    angle = float(np.linalg.norm(displacement))
    increment = (np.r_[1.0, displacement * 0.5] if angle < 1e-12 else
                 np.r_[math.cos(angle / 2), displacement * (math.sin(angle / 2) / angle)])
    return _quaternion(quat_multiply(increment, orientation))


def _limit_norm(vector, maximum):
    magnitude = float(np.linalg.norm(vector))
    return vector * min(1.0, maximum / magnitude) if magnitude > 0 else vector.copy()


def bounded_rotation_command(*, state, measured_orientation, target_orientation,
                             initial_orientation, step, dt, speed, acceleration, lookahead):
    """Propose one interval, braking before both the goal and the tracking edge.

    No catch-up integration after skipped ticks. A missed tick/rejected command
    restarts from the last accepted orientation at rest. Abrupt disturbances that
    make the hard lookahead incompatible with bounded acceleration raise instead
    of silently snapping the reference back or increasing the lookahead.
    """
    if not all(np.isfinite(x) and x > 0 for x in (dt, speed, acceleration, lookahead)):
        raise ValueError('Rotation timestep, speed, acceleration and lookahead must be positive and finite.')
    if lookahead >= math.pi:
        raise ValueError('Rotation lookahead must be less than pi.')
    measured = _quaternion(measured_orientation)
    target = _quaternion(target_orientation)
    previous = _quaternion(initial_orientation if state is None else state['orientation'])
    velocity = np.zeros(3)
    if state is not None and int(state['step']) == int(step) - 1:
        velocity = np.asarray(state['velocity'], dtype=float)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError('Invalid rotation velocity state.')
    if state is not None and int(step) <= int(state['step']):
        raise ValueError('Rotation proposal must use a new simulation step.')
    error = rotation_vector(target, previous)
    angle = float(np.linalg.norm(error))
    lead_vector = rotation_vector(previous, measured)
    lead = float(np.linalg.norm(lead_vector))
    if lead > lookahead + 1e-10:
        raise ValueError('rotation_tracking_boundary: accepted reference already exceeds lookahead')

    # Conservative discrete braking bound: reserve one interval as well as the
    # continuous v^2/(2a) stopping distance. Never accumulate a hidden endpoint
    # beyond the measured-pose tracking envelope.
    room = angle
    if float(np.dot(error, lead_vector)) >= 0:
        room = min(room, max(0.0, lookahead - lead))
    braking_speed = max(0.0, math.sqrt((acceleration * dt) ** 2 + 2 * acceleration * room)
                        - acceleration * dt)
    desired = error * (min(speed, braking_speed, angle / dt) / angle) if angle > 1e-12 else np.zeros(3)
    next_velocity = velocity + _limit_norm(desired - velocity, acceleration * dt)
    next_velocity = _limit_norm(next_velocity, speed)
    orientation = _advance(previous, next_velocity * dt)
    next_lead = float(np.linalg.norm(rotation_vector(orientation, measured)))
    if next_lead > lookahead + 1e-10:
        raise ValueError('rotation_tracking_boundary: bounded braking cannot satisfy lookahead')
    next_state = {'orientation': orientation.tolist(), 'velocity': next_velocity.tolist(), 'step': int(step)}
    detail = {
        'dt': float(dt), 'orientation_lead': next_lead, 'goal_error': angle,
        'angular_speed': float(np.linalg.norm(next_velocity)),
        'angular_acceleration': float(np.linalg.norm(next_velocity - velocity) / dt),
        'velocity': next_velocity.tolist(), 'tracking_room': max(0.0, lookahead - lead),
        'speed_limit': float(speed), 'acceleration_limit': float(acceleration),
        'lookahead_limit': float(lookahead),
        'restarted_after_gap': state is not None and int(state['step']) != int(step) - 1,
    }
    return orientation, next_state, detail
