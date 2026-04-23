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


def pose_error(
    current_position: Sequence[float],
    current_orientation: Sequence[float] | None,
    target_position: Sequence[float],
    target_orientation: Sequence[float] | None,
) -> Tuple[float, float | None]:
    current_position = np.asarray(current_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
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
) -> bool:
    position_error, orientation_error = pose_error(
        current_position=current_position,
        current_orientation=current_orientation,
        target_position=target_position,
        target_orientation=target_orientation,
    )
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
