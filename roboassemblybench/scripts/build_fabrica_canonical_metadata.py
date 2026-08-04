#!/usr/bin/env python3
"""Build runtime-safe metadata for the seven canonical Fabrica task bundles.

This is the only RoboAssemblyBench utility that reads Fabrica pickle files.
Run it in the ``fabrica`` environment with ``third_part/Fabrica`` on
``PYTHONPATH``; task loading itself only consumes the generated JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from planning.robot.geometry import get_gripper_open_ratio
from planning.robot.util_grasp import (
    get_grasp_info_from_gripper_state,
    get_gripper_pos_quat,
)
from planning.robot.workcell import get_assembly_center
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLES_ROOT = REPO_ROOT / 'roboassemblybench/assets/Fabrica/canonical_7_bundles/task_bundles'
DEFAULT_OUTPUT = REPO_ROOT / 'roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json'
FABRICA_TO_ISAAC_ROBOTIQ_ROTATION = Rotation.from_euler('z', np.pi / 2.0)
MINIMUM_BASE_INTERIOR_CLEARANCE_SCORE = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _as_floats(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _wxyz_from_xyzw(quaternion: Any) -> list[float]:
    x, y, z, w = _as_floats(quaternion)
    return [w, x, y, z]


def _pickup_orientation(scene_part: dict[str, Any]) -> list[float]:
    euler_degrees = scene_part['scene_pickup_rotation_xyz_degrees']
    return _wxyz_from_xyzw(Rotation.from_euler('xyz', euler_degrees, degrees=True).as_quat())


def _lookup_move_grasp(grasps: dict[str, Any], part_id: str, grasp_id: int):
    for grasp_path in grasps[part_id]['move']:
        if int(grasp_path[0].grasp_id) == int(grasp_id):
            return grasp_path[0]
    raise KeyError(f'Cannot find move grasp {grasp_id} for part {part_id}.')


def _lookup_hold_grasp(grasps: dict[str, Any], part_id: str, grasp_id: int):
    for grasp in grasps[part_id]['hold']:
        if int(grasp.grasp_id) == int(grasp_id):
            return grasp
    raise KeyError(f'Cannot find hold grasp {grasp_id} for part {part_id}.')


def _convert_panda_grasp(grasp: Any, *, assembly_center_cm: np.ndarray) -> dict[str, Any]:
    panda_position_cm = np.asarray(grasp.pos, dtype=float)
    panda_orientation_wxyz = np.asarray(grasp.quat, dtype=float)
    panda_open_ratio = float(grasp.open_ratio)
    grasp_info = get_grasp_info_from_gripper_state(
        'panda',
        panda_position_cm,
        panda_orientation_wxyz,
        panda_open_ratio,
    )

    grasp_width_cm = panda_open_ratio * 8.0
    half_width = grasp_width_cm * 0.5
    antipodal_points = np.stack(
        [
            grasp_info['grasp_center'] - grasp_info['l2r_direction'] * half_width,
            grasp_info['grasp_center'] + grasp_info['l2r_direction'] * half_width,
        ]
    )
    robotiq_open_ratio = get_gripper_open_ratio('robotiq-85', antipodal_points)
    if robotiq_open_ratio is None:
        raise ValueError(f'Panda grasp {grasp.grasp_id} is wider than the Robotiq 2F-85 workspace.')

    gripper_position_cm, gripper_orientation_wxyz = get_gripper_pos_quat(
        'robotiq-85',
        grasp_info['grasp_center'],
        grasp_info['base_direction'],
        grasp_info['l2r_direction'],
        robotiq_open_ratio,
    )
    gripper_position_m = (np.asarray(gripper_position_cm) - assembly_center_cm) * 0.01
    fabrica_gripper_orientation_wxyz = np.asarray(gripper_orientation_wxyz, dtype=float)
    fabrica_gripper_rotation = Rotation.from_quat(fabrica_gripper_orientation_wxyz[[1, 2, 3, 0]])
    gripper_rotation = fabrica_gripper_rotation * FABRICA_TO_ISAAC_ROBOTIQ_ROTATION
    gripper_orientation_wxyz = _wxyz_from_xyzw(gripper_rotation.as_quat())
    object_in_tcp_position = gripper_rotation.inv().apply(-gripper_position_m)
    object_in_tcp_orientation = _wxyz_from_xyzw(gripper_rotation.inv().as_quat())

    return {
        'source_gripper': 'panda',
        'target_gripper': 'robotiq-85',
        'target_gripper_asset': 'isaac_official_robotiq_2f85',
        'gripper_frame_conversion': 'fabrica_minus_x_to_isaac_plus_y',
        'gripper_frame_rotation_wxyz': _wxyz_from_xyzw(FABRICA_TO_ISAAC_ROBOTIQ_ROTATION.as_quat()),
        'grasp_id': int(grasp.grasp_id),
        'panda_open_ratio': panda_open_ratio,
        'robotiq_open_ratio': float(robotiq_open_ratio),
        'grasp_width_m': grasp_width_cm * 0.01,
        'tcp_in_assembly_position': _as_floats(gripper_position_m),
        'tcp_in_assembly_orientation': gripper_orientation_wxyz,
        'object_in_tcp_position': _as_floats(object_in_tcp_position),
        'object_in_tcp_orientation': object_in_tcp_orientation,
        'assembly_approach_direction': _as_floats(grasp_info['base_direction']),
    }


def _source_collision_count(grasp: Any) -> int:
    hold_collisions = getattr(grasp, 'parts_in_collision_hold', {}) or {}
    return sum(
        len(values or [])
        for values in (
            getattr(grasp, 'parts_in_collision', []),
            getattr(grasp, 'parts_in_collision_move', []),
            hold_collisions.get('move', []),
            hold_collisions.get('fix', []),
        )
    )


def _grasp_geometry(
    grasp: Any,
    *,
    assembly_center_cm: np.ndarray,
    bbox_min_m: np.ndarray,
    bbox_max_m: np.ndarray,
) -> dict[str, Any]:
    grasp_info = get_grasp_info_from_gripper_state(
        'panda',
        np.asarray(grasp.pos, dtype=float),
        np.asarray(grasp.quat, dtype=float),
        float(grasp.open_ratio),
    )
    grasp_center_m = (np.asarray(grasp_info['grasp_center'], dtype=float) - assembly_center_cm) * 0.01
    bbox_min_m = np.asarray(bbox_min_m, dtype=float)
    bbox_max_m = np.asarray(bbox_max_m, dtype=float)
    bbox_half_size_m = np.maximum((bbox_max_m - bbox_min_m) * 0.5, 1e-9)
    center_margins_m = np.minimum(
        grasp_center_m - bbox_min_m,
        bbox_max_m - grasp_center_m,
    )
    return {
        'grasp_center_m': _as_floats(grasp_center_m),
        'interior_clearance_score': float(np.min(center_margins_m / bbox_half_size_m)),
        'source_collision_count': _source_collision_count(grasp),
    }


def _build_base_grasp_candidates(
    grasps: dict[str, Any],
    *,
    part_id: str,
    planner_grasp: Any,
    assembly_center_cm: np.ndarray,
    bbox_min_m: np.ndarray,
    bbox_max_m: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for grasp in grasps[part_id]['hold']:
        try:
            converted = _convert_panda_grasp(
                grasp,
                assembly_center_cm=assembly_center_cm,
            )
        except ValueError:
            continue
        geometry = _grasp_geometry(
            grasp,
            assembly_center_cm=assembly_center_cm,
            bbox_min_m=bbox_min_m,
            bbox_max_m=bbox_max_m,
        )
        converted.update(
            {
                'selection_method': 'compiler_joint_pickup_yaw_base_grasp_selection',
                'planner_grasp_id': int(planner_grasp.grasp_id),
                'assembly_approach_cosine': float(converted['assembly_approach_direction'][2]),
                **geometry,
            }
        )
        candidates.append(converted)

    if not candidates:
        raise ValueError(f'No Robotiq-compatible hold grasp found for base part {part_id}.')
    valid_candidates = [
        item for item in candidates if item['interior_clearance_score'] >= MINIMUM_BASE_INTERIOR_CLEARANCE_SCORE
    ]
    if not valid_candidates:
        best_interior_score = max(item['interior_clearance_score'] for item in candidates)
        raise ValueError(
            f'No interior-safe Robotiq hold grasp found for base part {part_id}; '
            f'best normalized interior clearance is {best_interior_score:.3f}, '
            f'required {MINIMUM_BASE_INTERIOR_CLEARANCE_SCORE:.3f}.'
        )
    for candidate in valid_candidates:
        candidate.update(
            {
                'interior_clearance_minimum': MINIMUM_BASE_INTERIOR_CLEARANCE_SCORE,
                'valid_candidate_count': len(valid_candidates),
            }
        )
    return sorted(valid_candidates, key=lambda item: item['grasp_id'])


def _build_move_grasp_candidates(
    grasps: dict[str, Any],
    *,
    part_id: str,
    planner_grasp: Any,
    assembly_center_cm: np.ndarray,
    bbox_min_m: np.ndarray,
    bbox_max_m: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_grasp_ids: set[int] = set()
    for grasp_path in grasps[part_id]['move']:
        if not grasp_path:
            continue
        grasp = grasp_path[0]
        grasp_id = int(grasp.grasp_id)
        if grasp_id in seen_grasp_ids:
            continue
        seen_grasp_ids.add(grasp_id)
        try:
            converted = _convert_panda_grasp(
                grasp,
                assembly_center_cm=assembly_center_cm,
            )
        except ValueError:
            continue
        geometry = _grasp_geometry(
            grasp,
            assembly_center_cm=assembly_center_cm,
            bbox_min_m=bbox_min_m,
            bbox_max_m=bbox_max_m,
        )
        grasp_center = np.asarray(geometry['grasp_center_m'], dtype=float)
        tcp_position = np.asarray(converted['tcp_in_assembly_position'], dtype=float)
        converted.update(
            {
                'selection_method': 'compiler_move_grasp_candidate_conversion',
                'planner_grasp_id': int(planner_grasp.grasp_id),
                'is_planner_grasp': grasp_id == int(planner_grasp.grasp_id),
                'grasp_lever_arm_m': float(np.linalg.norm(tcp_position - grasp_center)),
                **geometry,
            }
        )
        candidates.append(converted)

    if not candidates:
        raise ValueError(f'No Robotiq-compatible move grasp found for moving part {part_id}.')
    if not any(candidate['is_planner_grasp'] for candidate in candidates):
        raise ValueError(
            f'Planner move grasp {planner_grasp.grasp_id} for part {part_id} is not ' 'Robotiq-compatible.'
        )
    for candidate in candidates:
        candidate['valid_candidate_count'] = len(candidates)
    return sorted(candidates, key=lambda item: item['grasp_id'])


def _ordered_tree_edges(tree: Any) -> list[dict[str, Any]]:
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f'Expected one optimized-tree root, found {len(roots)}.')

    result: list[dict[str, Any]] = []
    parent = roots[0]
    while tree.out_degree(parent):
        children = list(tree.successors(parent))
        if len(children) != 1:
            raise ValueError('The optimized Fabrica tree must be a chain.')
        child = children[0]
        result.append(dict(tree.edges[parent, child]))
        parent = child
    return result


def _insertion_path(plan_entry: dict[str, Any]) -> list[dict[str, list[float]]]:
    result = []
    for pose in plan_entry['path']:
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f'Expected a seven-element insertion pose, got {pose.size}.')
        result.append(
            {
                'position': _as_floats(pose[:3]),
                'orientation': _wxyz_from_xyzw(pose[3:]),
            }
        )
    return result


def _relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _build_task(bundle_dir: Path) -> tuple[str, dict[str, Any]]:
    scene_spec_path = bundle_dir / 'scene/scene_spec.json'
    metadata_dir = bundle_dir / 'metadata'
    scene_spec = json.loads(scene_spec_path.read_text(encoding='utf-8'))
    assembly = str(scene_spec['assembly'])

    tree_path = metadata_dir / 'tree_opt.pkl'
    grasps_path = metadata_dir / 'grasps.pkl'
    plan_info_path = metadata_dir / 'plan_info.pkl'
    with tree_path.open('rb') as handle:
        tree = pickle.load(handle)
    with grasps_path.open('rb') as handle:
        grasp_payload = pickle.load(handle)
    with plan_info_path.open('rb') as handle:
        plan_info = pickle.load(handle)

    if grasp_payload['gripper'] != 'panda':
        raise ValueError(f'{assembly}: expected Panda source grasps.')
    grasps = grasp_payload['grasps']
    assembly_center_cm = np.asarray(get_assembly_center(grasp_payload['arm']), dtype=float)
    disassembly_edges = _ordered_tree_edges(tree)
    moved_parts = [str(edge['move_part']) for edge in disassembly_edges]
    part_ids = list(scene_spec['scene']['parts'])
    base_parts = sorted(set(part_ids) - set(moved_parts), key=lambda value: int(value))
    if len(base_parts) != 1:
        raise ValueError(f'{assembly}: expected one base part, found {base_parts}.')
    base_part = base_parts[0]

    final_edge = disassembly_edges[-1]
    if str(final_edge['hold_part']) != base_part:
        raise ValueError(f'{assembly}: final optimized hold part {final_edge["hold_part"]} is not base {base_part}.')
    planner_base_grasp = _lookup_hold_grasp(
        grasps,
        base_part,
        int(final_edge['hold_grasp_id']),
    )
    base_grasp_candidates = _build_base_grasp_candidates(
        grasps,
        part_id=base_part,
        planner_grasp=planner_base_grasp,
        assembly_center_cm=assembly_center_cm,
        bbox_min_m=np.asarray(scene_spec['scene']['parts'][base_part]['bbox_min_m'], dtype=float),
        bbox_max_m=np.asarray(scene_spec['scene']['parts'][base_part]['bbox_max_m'], dtype=float),
    )

    plan_by_move: dict[str, tuple[str, dict[str, Any]]] = {}
    for (move_part, socket_part), plan_entry in plan_info.items():
        move_part = str(move_part)
        if move_part in plan_by_move:
            raise ValueError(f'{assembly}: duplicate plan entry for moving part {move_part}.')
        plan_by_move[move_part] = (str(socket_part), plan_entry)

    disassembly_steps: list[dict[str, Any]] = []
    for edge in disassembly_edges:
        move_part = str(edge['move_part'])
        optimizer_hold_part = str(edge['hold_part'])
        move_grasp = _lookup_move_grasp(grasps, move_part, int(edge['move_grasp_id']))
        hold_grasp = _lookup_hold_grasp(
            grasps,
            optimizer_hold_part,
            int(edge['hold_grasp_id']),
        )
        socket_part, plan_entry = plan_by_move[move_part]
        scene_part = scene_spec['scene']['parts'][move_part]
        move_grasp_candidates = _build_move_grasp_candidates(
            grasps,
            part_id=move_part,
            planner_grasp=move_grasp,
            assembly_center_cm=assembly_center_cm,
            bbox_min_m=np.asarray(scene_part['bbox_min_m'], dtype=float),
            bbox_max_m=np.asarray(scene_part['bbox_max_m'], dtype=float),
        )
        planner_move_grasp = next(candidate for candidate in move_grasp_candidates if candidate['is_planner_grasp'])
        disassembly_steps.append(
            {
                'move_part': move_part,
                'socket_part': socket_part,
                'optimizer_hold_part': optimizer_hold_part,
                'move_grasp': planner_move_grasp,
                'move_grasp_candidates': move_grasp_candidates,
                'hold_grasp': _convert_panda_grasp(
                    hold_grasp,
                    assembly_center_cm=assembly_center_cm,
                ),
                'disassembly_path': _insertion_path(plan_entry),
            }
        )

    if set(plan_by_move) != set(moved_parts):
        raise ValueError(
            f'{assembly}: plan/tree moved-part mismatch: ' f'{sorted(plan_by_move)} != {sorted(moved_parts)}.'
        )

    parts = []
    for part_id, scene_part in scene_spec['scene']['parts'].items():
        relative_usd_path = bundle_dir / scene_part['relative_usd_path']
        parts.append(
            {
                'part_id': str(part_id),
                'name': f'fabrica_{assembly}_{part_id}',
                'usd_path': _relative_to_repo(relative_usd_path),
                'pickup_position': _as_floats(scene_part['scene_pickup_translation_m_local_to_board']),
                'pickup_orientation': _pickup_orientation(scene_part),
                'bbox_min': _as_floats(scene_part['bbox_min_m']),
                'bbox_max': _as_floats(scene_part['bbox_max_m']),
                'bbox_size': _as_floats(scene_part['bbox_size_m']),
            }
        )

    fixture_path = bundle_dir / f'assets/fabrica_official_fixture/{assembly}/fixture_raw_sdf512.usda'
    board_path = bundle_dir / 'assets/fabrica_official_support/optical_board_raw_sdf512.usda'
    fixture_spec = scene_spec['assets']['fixture']
    board_spec = scene_spec['assets']['optical_board']
    return assembly, {
        'bundle_path': _relative_to_repo(bundle_dir),
        'scene_spec_path': _relative_to_repo(scene_spec_path),
        'parts': parts,
        'base_part': base_part,
        'base_grasp_candidates': base_grasp_candidates,
        'fixture': {
            'usd_path': _relative_to_repo(fixture_path),
            'bbox_min': _as_floats(fixture_spec['bbox_min_m']),
            'bbox_max': _as_floats(fixture_spec['bbox_max_m']),
            'bbox_size': _as_floats(fixture_spec['bbox_size_m']),
        },
        'optical_board': {
            'usd_path': _relative_to_repo(board_path),
            'bbox_min': _as_floats(board_spec['bbox_min_m']),
            'bbox_max': _as_floats(board_spec['bbox_max_m']),
            'bbox_size': _as_floats(board_spec['bbox_size_m']),
        },
        'disassembly_steps': disassembly_steps,
        'assembly_steps': list(reversed(disassembly_steps)),
        'source_hashes': {
            'scene_spec.json': _sha256(scene_spec_path),
            'tree_opt.pkl': _sha256(tree_path),
            'grasps.pkl': _sha256(grasps_path),
            'plan_info.pkl': _sha256(plan_info_path),
        },
    }


def build_metadata(bundles_root: Path) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for bundle_dir in sorted(path for path in bundles_root.iterdir() if path.is_dir()):
        scene_spec_path = bundle_dir / 'scene/scene_spec.json'
        if not scene_spec_path.exists():
            continue
        assembly, task = _build_task(bundle_dir)
        if assembly in tasks:
            raise ValueError(f'Duplicate canonical Fabrica assembly {assembly!r}.')
        tasks[assembly] = task
    if len(tasks) != 7:
        raise ValueError(f'Expected seven canonical Fabrica tasks, found {len(tasks)}.')
    return {
        'schema_version': 'roboassemblybench.fabrica_canonical/v3',
        'source_note': (
            'Generated from canonical Fabrica bundles. Pickles are build-time inputs only; '
            'RoboAssemblyBench task loading reads this JSON.'
        ),
        'tasks': tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles-root', type=Path, default=DEFAULT_BUNDLES_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_metadata(args.bundles_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(f'Wrote {len(payload["tasks"])} tasks to {args.output}')


if __name__ == '__main__':
    main()
