"""Part-0 opposite-entry experiment with an equivalent physical pickup pose."""
from __future__ import annotations

import copy
import math

import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import quat_conjugate, quat_multiply, quat_rotate


def apply_part0_reverse_insertion(task: dict, spec: dict) -> dict | None:
    """Reflect the axial path and use the mesh's half-turn symmetry at pickup.

    The audited part-0 mesh has a constant Y section and an X-symmetric profile.
    A half-turn around its X/Y centre leaves its world geometry unchanged. Use
    that equivalent initial body orientation and the matching mirrored grasp,
    so the physical pickup TCP remains identical to baseline. At the assembly
    end the contact is on the exposed +Y tail and insertion travels towards -Y.
    Final assembly poses, contact criteria and other parts remain unchanged.
    """
    enabled = spec.get('physical_part0_reverse_insertion', False)
    if type(enabled) is not bool:
        raise ValueError('physical_part0_reverse_insertion must be an explicit boolean.')
    if not enabled:
        return None
    if spec.get('assembly') != 'plumbers_block' or spec.get('continuous_physics') is not True:
        raise ValueError('physical_part0_reverse_insertion requires continuous physical plumbers_block.')
    weak_side_close = spec.get('physical_part0_reverse_weak_side_close', False)
    if not isinstance(weak_side_close, bool):
        raise ValueError('physical_part0_reverse_weak_side_close must be an explicit boolean.')
    steps = [s for s in task['assembly_steps'] if str(s['move_part']) == '0']
    if len(steps) != 1 or str(steps[0]['socket_part']) != '2' or str(task['base_part']) != '2':
        raise ValueError('Reverse part-0 insertion requires the audited part-0/base-2 pair.')
    step = steps[0]
    path = step['disassembly_path']
    if len(path) < 2:
        raise ValueError('Reverse part-0 insertion requires a nonempty straight axial path.')
    positions = [[float(x) for x in p['position']] for p in path]
    orientations = [[float(x) for x in p['orientation']] for p in path]
    if (any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in positions)
            or any(len(q) != 4 or not all(math.isfinite(x) for x in q) for q in orientations)):
        raise ValueError('Reverse part-0 insertion requires finite 3D positions and quaternions.')
    final = positions[0]
    norm0 = math.sqrt(sum(x * x for x in orientations[0]))
    if norm0 <= 1e-12:
        raise ValueError('Reverse part-0 insertion has a zero final quaternion.')
    for p, q in zip(positions, orientations):
        norm = math.sqrt(sum(x * x for x in q))
        if (abs(p[0] - final[0]) > 1e-6 or abs(p[2] - final[2]) > 1e-6 or norm <= 1e-12
                or abs(sum(a * b for a, b in zip(q, orientations[0]))) / (norm * norm0) < 1 - 1e-10):
            raise ValueError('Reverse part-0 insertion only supports straight local-Y translation at fixed orientation.')
    if (positions[-1][1] >= final[1] - 1e-6
            or any(b[1] > a[1] + 1e-9 for a, b in zip(positions, positions[1:]))):
        raise ValueError('Reverse part-0 insertion expects the original monotone -Y disassembly path.')
    sources = [g for g in step['move_grasp_candidates'] if int(g['grasp_id']) == 856]
    if len(sources) != 1 or sources[0].get('target_gripper') != 'robotiq-85':
        raise ValueError('Reverse part-0 insertion requires Robotiq source candidate 856.')
    part = next(p for p in task['parts'] if str(p['part_id']) == '0')
    base = next(p for p in task['parts'] if str(p['part_id']) == '2')
    pivot = np.array([.5 * (part['bbox_min'][0] + part['bbox_max'][0]), 0., 0.])
    half_turn = [0., 0., 0., 1.]
    translation = pivot - quat_rotate(half_turn, pivot)
    source = sources[0]
    grasp = copy.deepcopy(source)
    position = translation + quat_rotate(half_turn, source['tcp_in_assembly_position'])
    center = translation + quat_rotate(half_turn, source['grasp_center_m'])
    orientation = quat_multiply(half_turn, source['tcp_in_assembly_orientation'])
    inverse = quat_conjugate(orientation)
    if not base['bbox_max'][1] < center[1] < part['bbox_max'][1]:
        raise ValueError('Reverse part-0 contact must remain on its exposed +Y end.')
    old_pickup_position = list(part['pickup_position'])
    old_pickup_orientation = list(part['pickup_orientation'])
    # T leaves the mesh unchanged; T*T is identity, so (pickup*T)*(T*grasp)
    # reproduces the original world-space pickup TCP, including its depth.
    part['pickup_position'] = (np.asarray(old_pickup_position)
                               + quat_rotate(old_pickup_orientation, translation)).tolist()
    part['pickup_orientation'] = quat_multiply(old_pickup_orientation, half_turn).tolist()
    derivation = {
        'source_grasp_id': 856, 'object_z_half_turn_degrees': 180.,
        'symmetry_pivot': pivot.tolist(), 'symmetry_translation': translation.tolist(),
        'source_tcp_in_assembly_orientation': list(source['tcp_in_assembly_orientation']),
        'source_tcp_in_assembly_position': list(source['tcp_in_assembly_position']),
        'source_grasp_center_m': list(source['grasp_center_m']),
        'source_pickup_position': old_pickup_position,
        'source_pickup_orientation': old_pickup_orientation,
        'preserves_physical_pickup_tcp': True,
        'initial_body_pose_uses_mesh_symmetry': True,
        'source_collision_count_applies_to_source_only': True,
        'requires_native_contact_validation': True,
    }
    grasp.update(
        grasp_id=1000856, is_planner_grasp=False, panda_compatible=False,
        grasp_center_m=center.tolist(), tcp_in_assembly_position=position.tolist(),
        tcp_in_assembly_orientation=orientation.tolist(),
        object_in_tcp_orientation=inverse.tolist(),
        object_in_tcp_position=(-quat_rotate(inverse, position)).tolist(),
        assembly_approach_direction=quat_rotate(half_turn, source['assembly_approach_direction']).tolist(),
        reverse_entry_derivation=derivation,
    )
    for index, (point, p) in enumerate(zip(path, positions)):
        if index:
            point['position'] = [p[0], 2 * final[1] - p[1], p[2]]
    step['move_grasp_candidates'].append(grasp)
    return {
        'enabled': True, 'part': '0', 'socket': '2', 'axis_frame': 'assembly',
        'original_insertion_axis': [0., 1., 0.], 'insertion_axis': [0., -1., 0.],
        'original_entry_position': positions[-1], 'entry_position': list(path[-1]['position']),
        'final_position': list(path[0]['position']), 'final_orientation': list(path[0]['orientation']),
        'grasp_id': 1000856, 'source_grasp_id': 856,
        'initial_body_pose_uses_mesh_symmetry': True, 'physical_pickup_tcp_matches_baseline': True,
        'weak_side_close': weak_side_close,
        'geometry_status': 'half-turn-symmetric part and through-slot; requires native contact validation',
    }
