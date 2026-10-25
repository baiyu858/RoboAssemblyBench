"""A material grasp point captured once, not a moving contact centroid.

All positions are in metres. Current contact measurements are used only during
registration; the hold check transforms the same object-local point thereafter.
"""

from __future__ import annotations

import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import (
    normalize_quat,
    quat_conjugate,
    quat_rotate,
    relative_pose,
)


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain {size} finite values.') from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain {size} finite values.')
    if size == 4:
        if np.linalg.norm(vector) < 1e-12:
            raise ValueError(f'{name} cannot be a zero quaternion.')
        return normalize_quat(vector)
    return vector


def capture_grasp_reference(
    *, object_position, object_orientation, tcp_position, tcp_orientation,
    contact_metrics: dict, target_prim_path: str, step: int,
) -> dict:
    """Freeze the midpoint of the two force-weighted target contact patches.

    Each finger contributes one centroid, with equal weight between fingers.
    This avoids shifting the midpoint toward whichever finger is squeezing more.
    Missing/truncated/invalid contact details must never fall back to geometry.
    """
    object_position = _finite_vector(object_position, 3, 'object_position')
    object_orientation = _finite_vector(object_orientation, 4, 'object_orientation')
    tcp_position = _finite_vector(tcp_position, 3, 'tcp_position')
    tcp_orientation = _finite_vector(tcp_orientation, 4, 'tcp_orientation')
    centroids = {}
    for side in ('left_finger', 'right_finger'):
        finger = contact_metrics.get(side) or {}
        observation = finger.get('force_observation') or {}
        if not (
            finger.get('force_probe_valid') and finger.get('force_contact')
            and observation.get('valid') and observation.get('details_valid')
            and not observation.get('details_at_capacity')
            and observation.get('source') == 'filtered_contact_probe'
            and observation.get('other_body') == target_prim_path
        ):
            raise ValueError(f'{side}: valid complete target-pair contact details required.')
        points, weights = [], []
        for contact in observation.get('contacts') or []:
            force = contact.get('normal_force')
            if not isinstance(force, (int, float)) or not np.isfinite(force) or force < 0:
                raise ValueError(f'{side}: invalid contact normal force.')
            if force > 0:
                points.append(_finite_vector(contact.get('point_world'), 3, f'{side}.point_world'))
                weights.append(float(force))
        if not weights:
            raise ValueError(f'{side}: no loaded target contact point.')
        centroids[side] = np.average(np.asarray(points), axis=0, weights=weights)
    point_world = 0.5 * (centroids['left_finger'] + centroids['right_finger'])
    point_in_object = quat_rotate(quat_conjugate(object_orientation), point_world - object_position)
    point_in_tcp = quat_rotate(quat_conjugate(tcp_orientation), point_world - tcp_position)
    origin_in_tcp, orientation_in_tcp = relative_pose(
        tcp_position, tcp_orientation, object_position, object_orientation
    )
    return {
        'source': 'target_pair_force_patch_midpoint',
        'captured_step': int(step),
        'point_in_object': point_in_object.tolist(),
        'point_in_tcp_at_attach': point_in_tcp.tolist(),
        'object_origin_in_tcp_at_attach': origin_in_tcp.tolist(),
        'object_orientation_in_tcp_at_attach': orientation_in_tcp.tolist(),
        'point_world_at_attach': point_world.tolist(),
        'finger_centroids_world_at_attach': {side: point.tolist() for side, point in centroids.items()},
    }


def measure_grasp_point_slip(reference: dict, *, relative_position, relative_orientation) -> dict:
    """Measure one fixed material point. This function never updates reference."""
    if not isinstance(reference, dict):
        raise ValueError('Missing fixed grasp reference; cannot recapture during hold.')
    point = _finite_vector(reference.get('point_in_object'), 3, 'point_in_object')
    anchor = _finite_vector(reference.get('point_in_tcp_at_attach'), 3, 'point_in_tcp_at_attach')
    origin_anchor = _finite_vector(reference.get('object_origin_in_tcp_at_attach'), 3,
                                   'object_origin_in_tcp_at_attach')
    orientation_anchor = _finite_vector(reference.get('object_orientation_in_tcp_at_attach'), 4,
                                        'object_orientation_in_tcp_at_attach')
    position = _finite_vector(relative_position, 3, 'relative_position')
    orientation = _finite_vector(relative_orientation, 4, 'relative_orientation')
    current = position + quat_rotate(orientation, point)
    return {
        'position_slip_reference': 'grasp_point',
        'position_slip': float(np.linalg.norm(current - anchor)),
        'object_origin_position_slip': float(np.linalg.norm(position - origin_anchor)),
        'grasp_point_in_tcp': current.tolist(),
        'grasp_point_in_tcp_at_attach': anchor.tolist(),
        'grasp_point_in_object': point.tolist(),
        'grasp_reference_capture_step': reference.get('captured_step'),
        'orientation_anchor': orientation_anchor.tolist(),
    }
