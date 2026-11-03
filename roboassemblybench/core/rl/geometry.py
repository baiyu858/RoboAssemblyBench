"""Minimal wxyz rigid transforms used by the simulator-independent RL core."""
import numpy as np


def normalize_quat(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    return q / np.linalg.norm(q)


def quat_conjugate(value):
    q = normalize_quat(value)
    return q * np.array([1., -1., -1., -1.])


def quat_multiply(lhs, rhs):
    a, b, c, d = np.asarray(lhs, dtype=float)
    e, f, g, h = np.asarray(rhs, dtype=float)
    return np.array([a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g,
                     a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e])


def quat_rotate(q, vector):
    q = normalize_quat(q)
    return quat_multiply(quat_multiply(q, np.r_[0., np.asarray(vector)]), quat_conjugate(q))[1:]


def compose_pose(position, orientation, local_position, local_orientation):
    return (np.asarray(position) + quat_rotate(orientation, local_position),
            normalize_quat(quat_multiply(orientation, local_orientation)))


def relative_pose(position, orientation, target_position, target_orientation):
    inverse = quat_conjugate(orientation)
    return (quat_rotate(inverse, np.asarray(target_position) - position),
            normalize_quat(quat_multiply(inverse, target_orientation)))
