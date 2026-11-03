from __future__ import annotations

import math
from random import Random
from typing import Iterable, Sequence, Tuple

import numpy as np


def normalize_quat(quat: Sequence[float]) -> np.ndarray:
    quat_array = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat_array)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_array / norm


def quat_conjugate(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quat(quat)
    return np.array([w, -x, -y, -z], dtype=float)


def quat_multiply(lhs: Sequence[float], rhs: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(lhs, dtype=float)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_rotate(quat: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    pure = np.array([0.0, *np.asarray(vector, dtype=float)], dtype=float)
    unit_quat = normalize_quat(quat)
    rotated = quat_multiply(quat_multiply(unit_quat, pure), quat_conjugate(unit_quat))
    return rotated[1:]


def quat_angle_error(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    lhs = normalize_quat(lhs)
    rhs = normalize_quat(rhs)
    relative = quat_multiply(quat_conjugate(lhs), rhs)
    scalar = float(np.clip(abs(normalize_quat(relative)[0]), -1.0, 1.0))
    return float(2.0 * math.acos(scalar))


def axis_rotation_error(current_orientation, target_orientation, local_axis) -> Tuple[float, float]:
    """Return unsigned swing and twist about an object-local axis, in radians.

    Decompose target^-1 * current = swing * twist. This is independent of the
    world frame and quaternion sign; swing measures shaft tilt, not axial spin.
    """
    axis = np.asarray(local_axis, dtype=float)
    if (axis.shape != (3,) or not np.all(np.isfinite(axis))
            or not np.isfinite(np.linalg.norm(axis)) or np.linalg.norm(axis) < 1e-12):
        raise ValueError('Orientation axis must be a finite nonzero local three-vector')
    axis = axis / np.linalg.norm(axis)
    for orientation in (current_orientation, target_orientation):
        q = np.asarray(orientation, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
            raise ValueError('Axis rotation requires two finite nonzero quaternions')
    relative = normalize_quat(quat_multiply(quat_conjugate(target_orientation), normalize_quat(current_orientation)))
    projection = float(np.dot(relative[1:], axis))
    twist_norm = math.hypot(float(relative[0]), projection)
    swing = 2.0 * math.acos(float(np.clip(twist_norm, 0., 1.)))
    # At 180 degrees of swing, twist is indeterminate: reject conservatively.
    twist = math.pi if twist_norm < 1e-12 else 2.0 * math.atan2(abs(projection), abs(float(relative[0])))
    return swing, twist


def pose_error(
    current_position: Sequence[float],
    current_orientation: Sequence[float] | None,
    target_position: Sequence[float],
    target_orientation: Sequence[float] | None,
    position_reference_point: Sequence[float] | None = None,
) -> Tuple[float, float | None]:
    current_position = np.asarray(current_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    if position_reference_point is not None:
        point = np.asarray(position_reference_point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError('Position reference point must be a finite local three-vector')
        if current_orientation is None or target_orientation is None:
            raise ValueError('A local position reference point requires both orientations')
        current_position = current_position + quat_rotate(current_orientation, point)
        target_position = target_position + quat_rotate(target_orientation, point)
    position_error = float(np.linalg.norm(current_position - target_position))
    if current_orientation is None or target_orientation is None:
        return position_error, None
    orientation_error = quat_angle_error(current_orientation, target_orientation)
    return position_error, orientation_error


def pose_within_tolerance(
    current_position: Sequence[float],
    current_orientation: Sequence[float] | None,
    target_position: Sequence[float],
    target_orientation: Sequence[float] | None,
    position_tolerance: float,
    orientation_tolerance: float | None = None,
    position_reference_point: Sequence[float] | None = None,
    *,
    orientation_axis: Sequence[float] | None = None,
    twist_tolerance: float | None = None,
) -> bool:
    position_error, orientation_error = pose_error(
        current_position=current_position,
        current_orientation=current_orientation,
        target_position=target_position,
        target_orientation=target_orientation,
        position_reference_point=position_reference_point,
    )
    if orientation_axis is not None or twist_tolerance is not None:
        if (orientation_axis is None or twist_tolerance is None or orientation_tolerance is None
                or isinstance(twist_tolerance, bool) or not math.isfinite(float(twist_tolerance))
                or not 0 < float(twist_tolerance) <= math.pi):
            raise ValueError('Axis rotation needs an orientation axis, swing tolerance and twist tolerance in (0, pi]')
        swing, twist = axis_rotation_error(current_orientation, target_orientation, orientation_axis)
        return bool(np.isfinite(position_error) and position_error <= float(position_tolerance)
                    and swing <= float(orientation_tolerance) and twist <= float(twist_tolerance))
    if position_error > float(position_tolerance):
        return False
    if orientation_tolerance is None or orientation_error is None:
        return True
    return orientation_error <= float(orientation_tolerance)


def euler_xyz_to_quat(euler_xyz: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in euler_xyz]
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return normalize_quat(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def euler_xyz_intrinsic_to_quat(euler_xyz: Sequence[float]) -> np.ndarray:
    """Convert intrinsic/local XYZ Euler angles to a quaternion.

    This matches the Rotate XYZ values users see and edit in the Isaac Sim UI
    property panel for camera prims.
    """

    roll, pitch, yaw = [float(value) for value in euler_xyz]
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = np.array([cr, sr, 0.0, 0.0], dtype=float)
    qy = np.array([cp, 0.0, sp, 0.0], dtype=float)
    qz = np.array([cy, 0.0, 0.0, sy], dtype=float)
    return normalize_quat(quat_multiply(quat_multiply(qx, qy), qz))


def compose_pose(
    base_position: Sequence[float],
    base_orientation: Sequence[float],
    local_position: Sequence[float],
    local_orientation: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    base_position = np.asarray(base_position, dtype=float)
    local_position = np.asarray(local_position, dtype=float)
    world_position = base_position + quat_rotate(base_orientation, local_position)
    world_orientation = quat_multiply(base_orientation, local_orientation)
    return world_position, normalize_quat(world_orientation)


def relative_pose(
    base_position: Sequence[float],
    base_orientation: Sequence[float],
    world_position: Sequence[float],
    world_orientation: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    base_position = np.asarray(base_position, dtype=float)
    world_position = np.asarray(world_position, dtype=float)
    inv_orientation = quat_conjugate(base_orientation)
    local_position = quat_rotate(inv_orientation, world_position - base_position)
    local_orientation = quat_multiply(inv_orientation, world_orientation)
    return local_position, normalize_quat(local_orientation)


def sample_position(base_position: Sequence[float], random_xy: Iterable[float] | None, rng: Random) -> np.ndarray:
    position = np.asarray(base_position, dtype=float).copy()
    if random_xy is None:
        return position
    span = list(random_xy)
    if len(span) >= 1:
        position[0] += rng.uniform(-float(span[0]), float(span[0]))
    if len(span) >= 2:
        position[1] += rng.uniform(-float(span[1]), float(span[1]))
    return position


def pose_dict(position: Sequence[float], orientation: Sequence[float]) -> dict:
    return {
        'position': np.asarray(position, dtype=float).tolist(),
        'orientation': normalize_quat(orientation).tolist(),
    }
