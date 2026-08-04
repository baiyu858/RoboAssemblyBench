from __future__ import annotations

import copy
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from roboassemblybench.core.paths import BENCHMARK_ROOT
from roboassemblybench.core.scene_profiles import deep_merge
from toolkits.factory_dual_franka_assembly.ur5e_skill_api import UR5eAssemblySkillAPI

CANONICAL_METADATA_PATH = BENCHMARK_ROOT / 'assets/Fabrica/canonical_7_bundles/canonical_tasks.json'
CANONICAL_SCHEMA_VERSION = 'roboassemblybench.fabrica_canonical/v3'
PICKUP_TCP_ORIENTATION_REFERENCE_WXYZ = [0.0, 1.0, 0.0, 0.0]
DEFAULT_PICKUP_YAW_CANDIDATES_DEGREES = [0.0, 90.0, -90.0, 180.0]


def _vector(value: Any, *, size: int, name: str) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f'{name} must contain {size} finite numbers.')
    return result


def _add(left: list[float], right: list[float]) -> list[float]:
    return [float(a + b) for a, b in zip(left, right)]


def _quat_conjugate(quaternion: list[float]) -> list[float]:
    w, x, y, z = quaternion
    return [w, -x, -y, -z]


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1e-12:
        raise ValueError('Cannot normalize a zero quaternion.')
    return [float(value / norm) for value in result]


def _quat_rotate(quaternion: list[float], vector: list[float]) -> list[float]:
    vector_quaternion = [0.0, *vector]
    rotated = _quat_multiply_raw(
        _quat_multiply_raw(quaternion, vector_quaternion),
        _quat_conjugate(quaternion),
    )
    return [float(value) for value in rotated[1:]]


def _quat_multiply_raw(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _compose_pose(
    base_position: list[float],
    base_orientation: list[float],
    local_position: list[float],
    local_orientation: list[float],
) -> tuple[list[float], list[float]]:
    return (
        _add(base_position, _quat_rotate(base_orientation, local_position)),
        _quat_multiply(base_orientation, local_orientation),
    )


def _yaw_quaternion(degrees: float) -> list[float]:
    half_angle = math.radians(float(degrees)) * 0.5
    return [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]


def _orientation_continuity(
    orientation: list[float],
    reference: list[float],
) -> float:
    return abs(float(sum(left * right for left, right in zip(orientation, reference))))


def _normalized_direction(values: list[float], *, name: str) -> list[float]:
    direction = _vector(values, size=3, name=name)
    norm = math.sqrt(sum(value * value for value in direction))
    if norm <= 1e-9:
        raise ValueError(f'{name} must have non-zero length.')
    return [value / norm for value in direction]


def _projected_box_minimum_lateral_extent(
    *,
    bbox_size: list[float],
    orientation: list[float],
    axis: list[float],
) -> float:
    """Return the minimum width of an oriented box perpendicular to ``axis``."""

    normalized_axis = np.asarray(
        _normalized_direction(axis, name='box lateral-extent axis'),
        dtype=float,
    )
    world_axes = [
        np.asarray(
            _quat_rotate(
                orientation,
                [1.0 if local_axis == axis_index else 0.0 for local_axis in range(3)],
            ),
            dtype=float,
        )
        for axis_index in range(3)
    ]
    lateral_directions: list[np.ndarray] = []
    for world_axis in world_axes:
        direction = np.cross(normalized_axis, world_axis)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            lateral_directions.append(direction / norm)
    if not lateral_directions:
        raise ValueError('Cannot derive a lateral direction for the oriented box.')
    return min(
        sum(float(size) * abs(float(np.dot(world_axis, direction))) for size, world_axis in zip(bbox_size, world_axes))
        for direction in lateral_directions
    )


@lru_cache(maxsize=4)
def _load_metadata_cached(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f'Unsupported canonical Fabrica metadata schema ' f'{payload.get("schema_version")!r}.')
    tasks = payload.get('tasks')
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError('Canonical Fabrica metadata does not define any tasks.')
    return payload


def load_fabrica_canonical_metadata(path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(path or CANONICAL_METADATA_PATH).expanduser().resolve()
    return copy.deepcopy(_load_metadata_cached(str(resolved_path)))


def _merge_named_entries(generated: list[dict], existing: list[dict]) -> list[dict]:
    existing_by_name = {
        str(entry['name']): entry for entry in existing if isinstance(entry, dict) and entry.get('name') is not None
    }
    result = []
    generated_names = set()
    for entry in generated:
        name = str(entry['name'])
        generated_names.add(name)
        result.append(deep_merge(entry, existing_by_name.get(name, {})))
    result.extend(
        copy.deepcopy(entry)
        for entry in existing
        if not isinstance(entry, dict) or entry.get('name') is None or str(entry['name']) not in generated_names
    )
    return result


def _asset_path(relative_path: str) -> str:
    return str((BENCHMARK_ROOT.parent / relative_path).resolve())


def _contact_box(part: dict[str, Any]) -> tuple[list[float], list[float]]:
    bbox_min = _vector(part['bbox_min'], size=3, name='part bbox_min')
    bbox_max = _vector(part['bbox_max'], size=3, name='part bbox_max')
    center = [(lower + upper) * 0.5 for lower, upper in zip(bbox_min, bbox_max)]
    scale = [max(upper - lower, 0.008) for lower, upper in zip(bbox_min, bbox_max)]
    return center, scale


def _tcp_pose_for_object_pose(
    *,
    object_position: list[float],
    object_orientation: list[float],
    grasp: dict[str, Any],
) -> tuple[list[float], list[float]]:
    tcp_orientation = _quat_multiply(
        object_orientation,
        _quat_conjugate(
            _vector(
                grasp['object_in_tcp_orientation'],
                size=4,
                name='grasp object_in_tcp_orientation',
            )
        ),
    )
    tcp_position = _add(
        object_position,
        [
            -value
            for value in _quat_rotate(
                tcp_orientation,
                _vector(
                    grasp['object_in_tcp_position'],
                    size=3,
                    name='grasp object_in_tcp_position',
                ),
            )
        ],
    )
    return tcp_position, tcp_orientation


def _point_aabb_distance(
    point: list[float],
    lower: list[float],
    upper: list[float],
) -> float:
    squared_distance = 0.0
    for value, minimum, maximum in zip(point, lower, upper):
        delta = max(minimum - value, 0.0, value - maximum)
        squared_distance += delta * delta
    return math.sqrt(squared_distance)


def _pickup_fixture_body_clearance(
    *,
    grasp_center: list[float],
    tcp_position: list[float],
    pickup_origin: list[float],
    pickup_orientation: list[float],
    fixture_bbox_min: list[float],
    fixture_bbox_max: list[float],
    footprint_margin: float,
    body_start_fraction: float,
    body_sample_count: int,
) -> float:
    """Estimate gripper-body clearance above a tabletop pickup fixture.

    The fingertips may enter a fixture pocket, so only the established gripper-body
    segment is sampled.  Measurements are made in fixture coordinates and the
    footprint is expanded to account for the body's radial extent.
    """

    inverse_pickup_orientation = _quat_conjugate(pickup_orientation)
    clearance = math.inf
    for sample_index in range(body_sample_count):
        interpolation = body_start_fraction + (
            (1.0 - body_start_fraction) * sample_index / max(body_sample_count - 1, 1)
        )
        sample = [center + interpolation * (tcp - center) for center, tcp in zip(grasp_center, tcp_position)]
        fixture_local = _quat_rotate(
            inverse_pickup_orientation,
            [value - origin for value, origin in zip(sample, pickup_origin)],
        )
        if (
            fixture_bbox_min[0] - footprint_margin <= fixture_local[0] <= fixture_bbox_max[0] + footprint_margin
            and fixture_bbox_min[1] - footprint_margin <= fixture_local[1] <= fixture_bbox_max[1] + footprint_margin
        ):
            clearance = min(clearance, fixture_local[2] - fixture_bbox_max[2])
    return 1.0 if not math.isfinite(clearance) else clearance


@lru_cache(maxsize=2)
def _ur5e_ik_chain(urdf_path: str):
    from ikpy.chain import Chain

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        chain = Chain.from_urdf_file(urdf_path)
    chain.active_links_mask = np.asarray(
        [link.joint_type != 'fixed' for link in chain.links],
        dtype=bool,
    )
    return chain


def _rotation_matrix(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = _vector(quaternion, size=4, name='rotation quaternion')
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError('rotation quaternion cannot be zero.')
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _ur5e_initial_ik_configuration(chain, robot: dict[str, Any]) -> np.ndarray:
    configured = robot.get('initial_joint_positions') or {}
    if not isinstance(configured, dict):
        raise ValueError('UR5e initial_joint_positions must be a mapping.')
    result = np.zeros(len(chain.links), dtype=float)
    for index, link in enumerate(chain.links):
        if link.name in configured:
            result[index] = float(configured[link.name])
    return result


def _pose_in_robot_frame(
    *,
    position: list[float],
    orientation: list[float],
    robot_position: list[float],
    robot_orientation: list[float],
) -> tuple[list[float], list[float]]:
    inverse_robot_orientation = _quat_conjugate(robot_orientation)
    return (
        _quat_rotate(
            inverse_robot_orientation,
            [value - origin for value, origin in zip(position, robot_position)],
        ),
        _quat_multiply(inverse_robot_orientation, orientation),
    )


def _solve_ur5e_ik_pose(
    *,
    chain,
    position: list[float],
    orientation: list[float],
    robot_position: list[float],
    robot_orientation: list[float],
    warm_start: np.ndarray,
    fallback_start: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, float, float]:
    local_position, local_orientation = _pose_in_robot_frame(
        position=position,
        orientation=orientation,
        robot_position=robot_position,
        robot_orientation=robot_orientation,
    )
    target = np.eye(4, dtype=float)
    target[:3, :3] = _rotation_matrix(local_orientation)
    target[:3, 3] = np.asarray(local_position, dtype=float)
    best: tuple[np.ndarray, float, float] | None = None
    for seed in (warm_start, fallback_start):
        try:
            solution = np.asarray(
                chain.inverse_kinematics_frame(
                    target,
                    initial_position=np.asarray(seed, dtype=float),
                    orientation_mode='all',
                    max_iter=max_iterations,
                ),
                dtype=float,
            )
            actual = np.asarray(chain.forward_kinematics(solution), dtype=float)
        except Exception:
            continue
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        delta_rotation = actual[:3, :3].T @ target[:3, :3]
        cosine = min(max((float(np.trace(delta_rotation)) - 1.0) * 0.5, -1.0), 1.0)
        orientation_error = math.acos(cosine)
        candidate = (solution, position_error, orientation_error)
        if best is None or (position_error, orientation_error) < (best[1], best[2]):
            best = candidate
    if best is None:
        return fallback_start.copy(), math.inf, math.inf
    return best


def _ur5e_jacobian_metrics(
    chain,
    configuration: np.ndarray,
    *,
    finite_difference_step: float = 1e-5,
) -> tuple[float, float]:
    active_indices = np.flatnonzero(chain.active_links_mask)
    current = np.asarray(chain.forward_kinematics(configuration), dtype=float)
    jacobian = np.zeros((6, len(active_indices)), dtype=float)
    for column, link_index in enumerate(active_indices):
        shifted = np.asarray(configuration, dtype=float).copy()
        shifted[link_index] += finite_difference_step
        pose = np.asarray(chain.forward_kinematics(shifted), dtype=float)
        jacobian[:3, column] = (pose[:3, 3] - current[:3, 3]) / finite_difference_step
        rotation_delta = pose[:3, :3] @ current[:3, :3].T
        jacobian[3:, column] = np.asarray(
            [
                rotation_delta[2, 1] - rotation_delta[1, 2],
                rotation_delta[0, 2] - rotation_delta[2, 0],
                rotation_delta[1, 0] - rotation_delta[0, 1],
            ],
            dtype=float,
        ) / (2.0 * finite_difference_step)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    minimum_singular_value = float(singular_values[-1])
    condition_number = (
        math.inf if minimum_singular_value <= 1e-12 else float(singular_values[0] / minimum_singular_value)
    )
    return minimum_singular_value, condition_number


def _move_grasp_ik_metrics(
    *,
    candidate: dict[str, Any],
    part: dict[str, Any],
    step: dict[str, Any],
    pickup_origin: list[float],
    pickup_orientation: list[float],
    assembly_origin: list[float],
    assembly_orientation: list[float],
    workspace_offset: list[float],
    robot: dict[str, Any],
    transport_tcp_height: float,
    lift_distance: float,
    position_tolerance: float,
    orientation_tolerance: float,
    minimum_manipulability: float,
    max_iterations: int,
) -> dict[str, Any]:
    from internutopia_extension.configs.robots.ur5e import arm_ik_cfg

    chain = _ur5e_ik_chain(str(arm_ik_cfg.robot_urdf_path))
    initial_configuration = _ur5e_initial_ik_configuration(chain, robot)
    robot_position = _vector(
        robot.get('position', [0.0, 0.0, 0.0]),
        size=3,
        name='UR5e IK robot position',
    )
    if bool(robot.get('apply_workspace_offset', True)):
        robot_position = _add(robot_position, workspace_offset)
    robot_orientation = _vector(
        robot.get('orientation', [1.0, 0.0, 0.0, 0.0]),
        size=4,
        name='UR5e IK robot orientation',
    )

    pickup_part_position, pickup_part_orientation = _part_pickup_pose(
        part,
        pickup_origin=pickup_origin,
        pickup_orientation=pickup_orientation,
    )
    pickup_tcp_position, pickup_tcp_orientation = _tcp_pose_for_object_pose(
        object_position=pickup_part_position,
        object_orientation=pickup_part_orientation,
        grasp=candidate,
    )
    pickup_tcp_position = _add(pickup_tcp_position, workspace_offset)

    path = list(reversed(step['disassembly_path']))
    assembly_waypoints = [
        _compose_pose(
            assembly_origin,
            assembly_orientation,
            _vector(waypoint['position'], size=3, name='IK waypoint position'),
            _vector(waypoint['orientation'], size=4, name='IK waypoint orientation'),
        )
        for waypoint in path
    ]
    first_object_position, first_object_orientation = assembly_waypoints[0]
    assembly_clearance_object_position = _object_position_at_tcp_height(
        object_position=first_object_position,
        object_orientation=first_object_orientation,
        grasp=candidate,
        tcp_height=transport_tcp_height,
    )
    assembly_clearance_tcp_position, assembly_clearance_tcp_orientation = _tcp_pose_for_object_pose(
        object_position=assembly_clearance_object_position,
        object_orientation=first_object_orientation,
        grasp=candidate,
    )
    targets = [
        (
            'pickup_approach',
            _add(pickup_tcp_position, [0.0, 0.0, lift_distance]),
            pickup_tcp_orientation,
        ),
        ('pickup', pickup_tcp_position, pickup_tcp_orientation),
        (
            'lift',
            [
                pickup_tcp_position[0],
                pickup_tcp_position[1],
                max(
                    pickup_tcp_position[2] + lift_distance,
                    transport_tcp_height + workspace_offset[2],
                ),
            ],
            pickup_tcp_orientation,
        ),
        (
            'assembly_clearance',
            _add(assembly_clearance_tcp_position, workspace_offset),
            assembly_clearance_tcp_orientation,
        ),
    ]
    for waypoint_index, (object_position, object_orientation) in enumerate(assembly_waypoints):
        tcp_position, tcp_orientation = _tcp_pose_for_object_pose(
            object_position=object_position,
            object_orientation=object_orientation,
            grasp=candidate,
        )
        target_name = (
            'insertion_final' if waypoint_index == len(assembly_waypoints) - 1 else f'insertion_{waypoint_index:02d}'
        )
        targets.append(
            (
                target_name,
                _add(tcp_position, workspace_offset),
                tcp_orientation,
            )
        )
    configuration = initial_configuration.copy()
    errors: dict[str, dict[str, float]] = {}
    feasible = True
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    minimum_path_manipulability = math.inf
    maximum_path_condition_number = 0.0
    for name, position, orientation in targets:
        configuration, position_error, orientation_error = _solve_ur5e_ik_pose(
            chain=chain,
            position=position,
            orientation=orientation,
            robot_position=robot_position,
            robot_orientation=robot_orientation,
            warm_start=configuration,
            fallback_start=configuration,
            max_iterations=max_iterations,
        )
        manipulability, condition_number = _ur5e_jacobian_metrics(
            chain,
            configuration,
        )
        errors[name] = {
            'position_error': position_error,
            'orientation_error': orientation_error,
            'minimum_jacobian_singular_value': manipulability,
            'jacobian_condition_number': condition_number,
        }
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_orientation_error = max(maximum_orientation_error, orientation_error)
        minimum_path_manipulability = min(
            minimum_path_manipulability,
            manipulability,
        )
        maximum_path_condition_number = max(
            maximum_path_condition_number,
            condition_number,
        )
        if (
            position_error > position_tolerance
            or orientation_error > orientation_tolerance
            or manipulability < minimum_manipulability
        ):
            feasible = False
            break
    return {
        'ik_feasible': feasible,
        'ik_position_tolerance': position_tolerance,
        'ik_orientation_tolerance': orientation_tolerance,
        'ik_minimum_manipulability_threshold': minimum_manipulability,
        'ik_maximum_position_error': maximum_position_error,
        'ik_maximum_orientation_error': maximum_orientation_error,
        'ik_minimum_path_manipulability': minimum_path_manipulability,
        'ik_maximum_path_condition_number': maximum_path_condition_number,
        'ik_errors_by_target': errors,
    }


def _move_grasp_candidate_metrics(
    *,
    assembly: str,
    candidate: dict[str, Any],
    part: dict[str, Any],
    step: dict[str, Any],
    assembled_parts: list[dict[str, Any]],
    pickup_origin: list[float],
    pickup_orientation: list[float],
    assembly_origin: list[float],
    assembly_orientation: list[float],
    workspace_offset: list[float],
    fixture_bbox_min: list[float],
    fixture_bbox_max: list[float],
    fixture_footprint_margin: float,
    robot_position: list[float],
    orientation_reference: list[float],
    approach_height: float,
    body_start_fraction: float,
    body_sample_count: int,
) -> dict[str, Any]:
    part_id = str(part['part_id'])
    pickup_part_position, pickup_part_orientation = _part_pickup_pose(
        part,
        pickup_origin=pickup_origin,
        pickup_orientation=pickup_orientation,
    )
    pickup_tcp_position, pickup_tcp_orientation = _tcp_pose_for_object_pose(
        object_position=pickup_part_position,
        object_orientation=pickup_part_orientation,
        grasp=candidate,
    )
    pickup_grasp_center = _add(
        pickup_part_position,
        _quat_rotate(
            pickup_part_orientation,
            _vector(
                candidate['grasp_center_m'],
                size=3,
                name=f'{assembly} part {part_id} pickup grasp_center_m',
            ),
        ),
    )
    pickup_fixture_body_clearance = _pickup_fixture_body_clearance(
        grasp_center=pickup_grasp_center,
        tcp_position=pickup_tcp_position,
        pickup_origin=pickup_origin,
        pickup_orientation=pickup_orientation,
        fixture_bbox_min=fixture_bbox_min,
        fixture_bbox_max=fixture_bbox_max,
        footprint_margin=fixture_footprint_margin,
        body_start_fraction=body_start_fraction,
        body_sample_count=body_sample_count,
    )
    pickup_approach_position = _add(
        _add(pickup_tcp_position, [0.0, 0.0, approach_height]),
        workspace_offset,
    )
    pickup_reach = math.sqrt(sum((pickup_approach_position[axis] - robot_position[axis]) ** 2 for axis in range(3)))
    pickup_orientation_continuity = _orientation_continuity(
        pickup_tcp_orientation,
        orientation_reference,
    )

    path = list(reversed(step['disassembly_path']))
    if not path:
        raise ValueError(f'{assembly}: part {part_id} has an empty insertion path.')
    local_positions = [
        _vector(
            waypoint['position'],
            size=3,
            name=f'{assembly} part {part_id} insertion waypoint position',
        )
        for waypoint in path
    ]
    local_orientations = [
        _vector(
            waypoint['orientation'],
            size=4,
            name=f'{assembly} part {part_id} insertion waypoint orientation',
        )
        for waypoint in path
    ]
    insertion_delta = [final - initial for initial, final in zip(local_positions[0], local_positions[-1])]
    if math.sqrt(sum(value * value for value in insertion_delta)) <= 1e-9:
        insertion_axis = _normalized_direction(
            candidate['assembly_approach_direction'],
            name=f'{assembly} part {part_id} fallback insertion axis',
        )
    else:
        insertion_axis = _normalized_direction(
            insertion_delta,
            name=f'{assembly} part {part_id} insertion axis',
        )
    final_approach_direction = _quat_rotate(
        local_orientations[-1],
        _normalized_direction(
            candidate['assembly_approach_direction'],
            name=f'{assembly} part {part_id} assembly approach direction',
        ),
    )
    insertion_axis_alignment = -sum(
        direction * axis for direction, axis in zip(final_approach_direction, insertion_axis)
    )

    tcp_in_assembly_position = _vector(
        candidate['tcp_in_assembly_position'],
        size=3,
        name=f'{assembly} part {part_id} tcp_in_assembly_position',
    )
    tcp_in_assembly_orientation = _vector(
        candidate['tcp_in_assembly_orientation'],
        size=4,
        name=f'{assembly} part {part_id} tcp_in_assembly_orientation',
    )
    grasp_center = _vector(
        candidate['grasp_center_m'],
        size=3,
        name=f'{assembly} part {part_id} grasp_center_m',
    )
    obstacle_boxes = [
        (
            _vector(
                obstacle['bbox_min'],
                size=3,
                name=f'{assembly} obstacle bbox_min',
            ),
            _vector(
                obstacle['bbox_max'],
                size=3,
                name=f'{assembly} obstacle bbox_max',
            ),
        )
        for obstacle in assembled_parts
    ]
    insertion_body_clearance = math.inf
    assembly_reaches: list[float] = []
    for local_position, local_orientation in zip(local_positions, local_orientations):
        local_tcp_position, _ = _compose_pose(
            local_position,
            local_orientation,
            tcp_in_assembly_position,
            tcp_in_assembly_orientation,
        )
        local_grasp_center = _add(
            local_position,
            _quat_rotate(local_orientation, grasp_center),
        )
        world_object_position, world_object_orientation = _compose_pose(
            assembly_origin,
            assembly_orientation,
            local_position,
            local_orientation,
        )
        world_tcp_position, _ = _tcp_pose_for_object_pose(
            object_position=world_object_position,
            object_orientation=world_object_orientation,
            grasp=candidate,
        )
        world_tcp_position = _add(world_tcp_position, workspace_offset)
        assembly_reaches.append(
            math.sqrt(sum((world_tcp_position[axis] - robot_position[axis]) ** 2 for axis in range(3)))
        )
        for sample_index in range(body_sample_count):
            interpolation = body_start_fraction + (
                (1.0 - body_start_fraction) * sample_index / max(body_sample_count - 1, 1)
            )
            sample = [
                center + interpolation * (tcp - center) for center, tcp in zip(local_grasp_center, local_tcp_position)
            ]
            for lower, upper in obstacle_boxes:
                insertion_body_clearance = min(
                    insertion_body_clearance,
                    _point_aabb_distance(sample, lower, upper),
                )
    if not math.isfinite(insertion_body_clearance):
        insertion_body_clearance = 1.0
    maximum_reach = max([pickup_reach, *assembly_reaches])
    source_collision_count = int(candidate.get('source_collision_count', 0))
    interior_clearance_score = float(candidate.get('interior_clearance_score', 0.0))
    physical_score = (
        insertion_body_clearance
        + 0.04 * insertion_axis_alignment
        + 0.01 * pickup_orientation_continuity
        + 0.02 * min(max(interior_clearance_score, 0.0), 1.0)
        + 0.02 * min(max(pickup_fixture_body_clearance, 0.0), 0.10) / 0.10
        - 0.005 * maximum_reach
        - 0.001 * min(source_collision_count, 3)
    )
    return {
        'grasp_id': int(candidate['grasp_id']),
        'is_planner_grasp': bool(candidate.get('is_planner_grasp', False)),
        'pickup_orientation_continuity': pickup_orientation_continuity,
        'pickup_tcp_reach': pickup_reach,
        'pickup_fixture_body_clearance': pickup_fixture_body_clearance,
        'maximum_tcp_reach': maximum_reach,
        'insertion_axis': insertion_axis,
        'insertion_axis_alignment': insertion_axis_alignment,
        'insertion_body_clearance': insertion_body_clearance,
        'interior_clearance_score': interior_clearance_score,
        'source_collision_count': source_collision_count,
        'physical_score': physical_score,
    }


def _select_move_grasps_for_layout(
    *,
    assembly: str,
    task: dict[str, Any],
    pickup_origin: list[float],
    pickup_orientation: list[float],
    assembly_origin: list[float],
    assembly_orientation: list[float],
    workspace_offset: list[float],
    fixture_bbox_min: list[float],
    fixture_bbox_max: list[float],
    fixture_footprint_margin: float,
    robot: dict[str, Any],
    robot_position: list[float],
    orientation_reference: list[float],
    minimum_orientation_continuity: float,
    maximum_tcp_reach: float,
    approach_height: float,
    body_start_fraction: float,
    body_sample_count: int,
    reselection_minimum_gain: float,
    preferred_interior_clearance: float,
    minimum_relative_interior_clearance: float,
    minimum_fixture_clearance: float,
    minimum_relative_orientation_continuity: float,
    transport_tcp_height: float,
    ik_lift_distance: float,
    ik_position_tolerance: float,
    ik_orientation_tolerance: float,
    ik_minimum_manipulability: float,
    ik_max_iterations: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool]:
    parts_by_id = {str(part['part_id']): part for part in task['parts']}
    assembled_parts = [parts_by_id[str(task['base_part'])]]
    selected_by_part: dict[str, dict[str, Any]] = {}
    diagnostics_by_part: dict[str, dict[str, Any]] = {}
    all_steps_feasible = True
    for step in task['assembly_steps']:
        part_id = str(step['move_part'])
        candidates = step.get('move_grasp_candidates')
        if not isinstance(candidates, list) or not candidates:
            legacy_grasp = step.get('move_grasp')
            candidates = [legacy_grasp] if isinstance(legacy_grasp, dict) else []
        evaluated = [
            {
                'candidate': candidate,
                'metrics': _move_grasp_candidate_metrics(
                    assembly=assembly,
                    candidate=candidate,
                    part=parts_by_id[part_id],
                    step=step,
                    assembled_parts=assembled_parts,
                    pickup_origin=pickup_origin,
                    pickup_orientation=pickup_orientation,
                    assembly_origin=assembly_origin,
                    assembly_orientation=assembly_orientation,
                    workspace_offset=workspace_offset,
                    fixture_bbox_min=fixture_bbox_min,
                    fixture_bbox_max=fixture_bbox_max,
                    fixture_footprint_margin=fixture_footprint_margin,
                    robot_position=robot_position,
                    orientation_reference=orientation_reference,
                    approach_height=approach_height,
                    body_start_fraction=body_start_fraction,
                    body_sample_count=body_sample_count,
                ),
            }
            for candidate in candidates
        ]
        kinematically_feasible = [
            entry
            for entry in evaluated
            if entry['metrics']['pickup_orientation_continuity'] >= minimum_orientation_continuity
            and entry['metrics']['maximum_tcp_reach'] <= maximum_tcp_reach
        ]
        if kinematically_feasible:
            minimum_source_collision_count = min(
                entry['metrics']['source_collision_count'] for entry in kinematically_feasible
            )
            source_collision_feasible = [
                entry
                for entry in kinematically_feasible
                if entry['metrics']['source_collision_count'] == minimum_source_collision_count
            ]
            maximum_fixture_clearance = max(
                entry['metrics']['pickup_fixture_body_clearance'] for entry in source_collision_feasible
            )
            required_fixture_clearance = minimum_fixture_clearance
            fixture_feasible = [
                entry
                for entry in source_collision_feasible
                if entry['metrics']['pickup_fixture_body_clearance'] >= required_fixture_clearance
            ]
            if fixture_feasible:
                maximum_orientation_continuity = max(
                    entry['metrics']['pickup_orientation_continuity'] for entry in fixture_feasible
                )
                required_orientation_continuity = max(
                    minimum_orientation_continuity,
                    maximum_orientation_continuity * minimum_relative_orientation_continuity,
                )
                orientation_feasible = [
                    entry
                    for entry in fixture_feasible
                    if entry['metrics']['pickup_orientation_continuity'] >= required_orientation_continuity
                ]
                maximum_interior_clearance = max(
                    entry['metrics']['interior_clearance_score'] for entry in orientation_feasible
                )
                required_interior_clearance = min(
                    preferred_interior_clearance,
                    maximum_interior_clearance * minimum_relative_interior_clearance,
                )
                feasible = [
                    entry
                    for entry in orientation_feasible
                    if entry['metrics']['interior_clearance_score'] >= required_interior_clearance
                ]
            else:
                maximum_orientation_continuity = 0.0
                required_orientation_continuity = minimum_orientation_continuity
                orientation_feasible = []
                maximum_interior_clearance = 0.0
                required_interior_clearance = 0.0
                feasible = []
        else:
            minimum_source_collision_count = None
            source_collision_feasible = []
            maximum_fixture_clearance = 0.0
            required_fixture_clearance = 0.0
            fixture_feasible = []
            maximum_orientation_continuity = 0.0
            required_orientation_continuity = minimum_orientation_continuity
            orientation_feasible = []
            maximum_interior_clearance = 0.0
            required_interior_clearance = 0.0
            feasible = []
        if not feasible:
            all_steps_feasible = False
            diagnostics_by_part[part_id] = {
                'candidate_count': len(evaluated),
                'kinematically_feasible_candidate_count': len(kinematically_feasible),
                'minimum_source_collision_count': minimum_source_collision_count,
                'source_collision_feasible_candidate_count': len(source_collision_feasible),
                'fixture_feasible_candidate_count': len(fixture_feasible),
                'feasible_candidate_count': 0,
                'minimum_orientation_continuity': minimum_orientation_continuity,
                'maximum_tcp_reach': maximum_tcp_reach,
                'required_fixture_clearance': required_fixture_clearance,
                'maximum_fixture_clearance': maximum_fixture_clearance,
                'required_orientation_continuity': (required_orientation_continuity),
                'maximum_orientation_continuity': (maximum_orientation_continuity),
                'required_interior_clearance': required_interior_clearance,
                'maximum_interior_clearance': maximum_interior_clearance,
                'ik_evaluated_candidate_count': 0,
                'ik_feasible_candidate_count': 0,
            }
            assembled_parts.append(parts_by_id[part_id])
            continue
        ranked = sorted(
            feasible,
            key=lambda entry: (
                entry['metrics']['physical_score'],
                entry['metrics']['pickup_fixture_body_clearance'],
                entry['metrics']['insertion_body_clearance'],
                entry['metrics']['insertion_axis_alignment'],
                -entry['metrics']['maximum_tcp_reach'],
                -entry['metrics']['grasp_id'],
            ),
            reverse=True,
        )
        planner = next(
            (entry for entry in feasible if entry['metrics']['is_planner_grasp']),
            None,
        )

        def evaluate_ik(entry: dict[str, Any]) -> None:
            if 'ik_feasible' in entry['metrics']:
                return
            entry['metrics'].update(
                _move_grasp_ik_metrics(
                    candidate=entry['candidate'],
                    part=parts_by_id[part_id],
                    step=step,
                    pickup_origin=pickup_origin,
                    pickup_orientation=pickup_orientation,
                    assembly_origin=assembly_origin,
                    assembly_orientation=assembly_orientation,
                    workspace_offset=workspace_offset,
                    robot=robot,
                    transport_tcp_height=transport_tcp_height,
                    lift_distance=ik_lift_distance,
                    position_tolerance=ik_position_tolerance,
                    orientation_tolerance=ik_orientation_tolerance,
                    minimum_manipulability=ik_minimum_manipulability,
                    max_iterations=ik_max_iterations,
                )
            )

        best = None
        for entry in ranked:
            evaluate_ik(entry)
            if entry['metrics']['ik_feasible']:
                best = entry
                break
        if planner is not None:
            evaluate_ik(planner)

        ik_evaluated = [entry for entry in feasible if 'ik_feasible' in entry['metrics']]
        ik_feasible = [entry for entry in ik_evaluated if entry['metrics']['ik_feasible']]
        if best is None:
            all_steps_feasible = False
            diagnostics_by_part[part_id] = {
                'candidate_count': len(evaluated),
                'kinematically_feasible_candidate_count': len(kinematically_feasible),
                'minimum_source_collision_count': minimum_source_collision_count,
                'source_collision_feasible_candidate_count': len(source_collision_feasible),
                'fixture_feasible_candidate_count': len(fixture_feasible),
                'geometry_feasible_candidate_count': len(feasible),
                'feasible_candidate_count': 0,
                'minimum_orientation_continuity': minimum_orientation_continuity,
                'maximum_tcp_reach': maximum_tcp_reach,
                'required_fixture_clearance': required_fixture_clearance,
                'maximum_fixture_clearance': maximum_fixture_clearance,
                'required_orientation_continuity': (required_orientation_continuity),
                'maximum_orientation_continuity': (maximum_orientation_continuity),
                'required_interior_clearance': required_interior_clearance,
                'maximum_interior_clearance': maximum_interior_clearance,
                'ik_position_tolerance': ik_position_tolerance,
                'ik_orientation_tolerance': ik_orientation_tolerance,
                'ik_minimum_manipulability': ik_minimum_manipulability,
                'ik_evaluated_candidate_count': len(ik_evaluated),
                'ik_feasible_candidate_count': 0,
                'ik_rejections': [copy.deepcopy(entry['metrics']) for entry in ik_evaluated],
            }
            assembled_parts.append(parts_by_id[part_id])
            continue

        selected = best
        if (
            planner is not None
            and planner['metrics']['ik_feasible']
            and best['metrics']['physical_score'] - planner['metrics']['physical_score'] < reselection_minimum_gain
        ):
            selected = planner
        selected_grasp = copy.deepcopy(selected['candidate'])
        selected_grasp.update(
            {
                'selection_method': 'runtime_physical_move_grasp_selection',
                **selected['metrics'],
            }
        )
        selected_by_part[part_id] = selected_grasp
        diagnostics_by_part[part_id] = {
            'candidate_count': len(evaluated),
            'kinematically_feasible_candidate_count': len(kinematically_feasible),
            'minimum_source_collision_count': minimum_source_collision_count,
            'source_collision_feasible_candidate_count': len(source_collision_feasible),
            'fixture_feasible_candidate_count': len(fixture_feasible),
            'geometry_feasible_candidate_count': len(feasible),
            'feasible_candidate_count': len(ik_feasible),
            'ik_evaluated_candidate_count': len(ik_evaluated),
            'ik_feasible_candidate_count': len(ik_feasible),
            'ik_position_tolerance': ik_position_tolerance,
            'ik_orientation_tolerance': ik_orientation_tolerance,
            'ik_minimum_manipulability': ik_minimum_manipulability,
            'selected_grasp_id': int(selected_grasp['grasp_id']),
            'planner_grasp_id': int(selected_grasp.get('planner_grasp_id', selected_grasp['grasp_id'])),
            'reselection_minimum_gain': reselection_minimum_gain,
            'required_fixture_clearance': required_fixture_clearance,
            'maximum_fixture_clearance': maximum_fixture_clearance,
            'required_orientation_continuity': required_orientation_continuity,
            'maximum_orientation_continuity': maximum_orientation_continuity,
            'required_interior_clearance': required_interior_clearance,
            'maximum_interior_clearance': maximum_interior_clearance,
            'selected': copy.deepcopy(selected['metrics']),
            'best': copy.deepcopy(best['metrics']),
            'planner': (copy.deepcopy(planner['metrics']) if planner is not None else None),
        }
        assembled_parts.append(parts_by_id[part_id])
    return selected_by_part, diagnostics_by_part, all_steps_feasible


def _resolve_pickup_layout_and_grasps(
    *,
    assembly: str,
    task: dict[str, Any],
    spec: dict[str, Any],
    configured_pickup_origin: list[float],
    configured_pickup_orientation: list[float],
    board_origin: list[float],
    assembly_origin: list[float],
    assembly_orientation: list[float],
    workspace_offset: list[float],
    robots: list[dict[str, Any]],
) -> tuple[list[float], list[float], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    raw_yaw_candidates = spec.get(
        'pickup_yaw_candidates_degrees',
        DEFAULT_PICKUP_YAW_CANDIDATES_DEGREES,
    )
    if not isinstance(raw_yaw_candidates, list) or not raw_yaw_candidates:
        raise ValueError('pickup_yaw_candidates_degrees must be a non-empty list.')
    yaw_candidates = [float(value) for value in raw_yaw_candidates]
    if not all(math.isfinite(value) for value in yaw_candidates):
        raise ValueError('pickup_yaw_candidates_degrees must contain finite numbers.')

    minimum_vertical_clearance = float(spec.get('base_grasp_minimum_vertical_clearance', 0.70))
    minimum_interior_clearance = float(spec.get('base_grasp_minimum_interior_clearance', 0.20))
    minimum_support_clearance_ratio = float(spec.get('base_grasp_minimum_support_clearance_ratio', 0.70))
    minimum_orientation_continuity = float(spec.get('minimum_pickup_orientation_continuity', 0.50))
    maximum_pickup_tcp_reach = float(spec.get('pickup_tcp_maximum_reach', 0.82))
    move_grasp_body_start_fraction = float(spec.get('move_grasp_body_start_fraction', 0.30))
    move_grasp_body_sample_count = int(spec.get('move_grasp_body_sample_count', 24))
    move_grasp_reselection_minimum_gain = float(spec.get('move_grasp_reselection_minimum_gain', 0.015))
    move_grasp_preferred_interior_clearance = float(spec.get('move_grasp_preferred_interior_clearance', 0.40))
    move_grasp_minimum_relative_interior_clearance = float(
        spec.get('move_grasp_minimum_relative_interior_clearance', 0.75)
    )
    move_grasp_fixture_footprint_margin = float(spec.get('move_grasp_fixture_footprint_margin', 0.035))
    move_grasp_minimum_fixture_clearance = float(spec.get('move_grasp_minimum_fixture_clearance', 0.001))
    move_grasp_minimum_relative_orientation_continuity = float(
        spec.get('move_grasp_minimum_relative_orientation_continuity', 0.90)
    )
    move_grasp_ik_position_tolerance = float(spec.get('move_grasp_ik_position_tolerance', 0.01))
    move_grasp_ik_orientation_tolerance = float(spec.get('move_grasp_ik_orientation_tolerance', 0.03))
    move_grasp_ik_max_iterations = int(spec.get('move_grasp_ik_max_iterations', 200))
    move_grasp_ik_minimum_manipulability = float(spec.get('move_grasp_ik_minimum_manipulability', 0.08))
    move_grasp_ik_lift_distance = float(spec.get('move_grasp_ik_lift_distance', spec.get('approach_height', 0.10)))
    orientation_reference = _vector(
        spec.get(
            'pickup_tcp_orientation_reference',
            PICKUP_TCP_ORIENTATION_REFERENCE_WXYZ,
        ),
        size=4,
        name='pickup_tcp_orientation_reference',
    )
    reference_norm = math.sqrt(sum(value * value for value in orientation_reference))
    if reference_norm <= 1e-12:
        raise ValueError('pickup_tcp_orientation_reference cannot be zero.')
    orientation_reference = [value / reference_norm for value in orientation_reference]
    if (
        not 0.0 <= minimum_vertical_clearance <= 1.0
        or not 0.0 <= minimum_interior_clearance <= 1.0
        or not 0.0 <= minimum_support_clearance_ratio <= 1.0
        or not 0.0 <= minimum_orientation_continuity <= 1.0
        or not math.isfinite(maximum_pickup_tcp_reach)
        or maximum_pickup_tcp_reach <= 0.0
        or not math.isfinite(move_grasp_body_start_fraction)
        or not 0.0 <= move_grasp_body_start_fraction < 1.0
        or move_grasp_body_sample_count < 2
        or not math.isfinite(move_grasp_reselection_minimum_gain)
        or move_grasp_reselection_minimum_gain < 0.0
        or not 0.0 <= move_grasp_preferred_interior_clearance <= 1.0
        or not 0.0 <= move_grasp_minimum_relative_interior_clearance <= 1.0
        or not math.isfinite(move_grasp_fixture_footprint_margin)
        or move_grasp_fixture_footprint_margin < 0.0
        or not math.isfinite(move_grasp_minimum_fixture_clearance)
        or move_grasp_minimum_fixture_clearance < 0.0
        or not 0.0 <= move_grasp_minimum_relative_orientation_continuity <= 1.0
        or not math.isfinite(move_grasp_ik_position_tolerance)
        or move_grasp_ik_position_tolerance <= 0.0
        or not math.isfinite(move_grasp_ik_orientation_tolerance)
        or move_grasp_ik_orientation_tolerance <= 0.0
        or move_grasp_ik_max_iterations <= 0
        or not math.isfinite(move_grasp_ik_minimum_manipulability)
        or move_grasp_ik_minimum_manipulability <= 0.0
        or not math.isfinite(move_grasp_ik_lift_distance)
        or move_grasp_ik_lift_distance <= 0.0
    ):
        raise ValueError(
            'Canonical grasp selection scores must be in [0, 1], and maximum reach '
            'must be finite and positive; move-grasp body sampling and reselection '
            'gain must also be valid; fixture clearance must be finite and non-negative; '
            'IK tolerances, lift distance, and iteration limit must be positive.'
        )

    parts_by_id = {str(part['part_id']): part for part in task['parts']}
    base_part = str(task['base_part'])
    base_candidates = task.get('base_grasp_candidates')
    if not isinstance(base_candidates, list) or not base_candidates:
        legacy_base_grasp = task.get('base_grasp')
        if not isinstance(legacy_base_grasp, dict):
            raise ValueError(f'{assembly}: metadata has no base grasp candidates.')
        base_candidates = [legacy_base_grasp]
    base_bbox_min = _vector(
        parts_by_id[base_part]['bbox_min'],
        size=3,
        name=f'{assembly} base bbox_min',
    )
    base_support_clearances = [
        float(
            _vector(
                candidate['grasp_center_m'],
                size=3,
                name=f'{assembly} base grasp_center_m',
            )[2]
            - base_bbox_min[2]
        )
        for candidate in base_candidates
    ]
    maximum_support_clearance = max(base_support_clearances)
    if not math.isfinite(maximum_support_clearance) or maximum_support_clearance <= 0.0:
        raise ValueError(f'{assembly}: base grasp candidates have no positive support clearance.')

    fixture_bbox_min = _vector(task['fixture']['bbox_min'], size=3, name='fixture bbox_min')
    fixture_bbox_max = _vector(task['fixture']['bbox_max'], size=3, name='fixture bbox_max')
    default_rotation_pivot = [(lower + upper) * 0.5 for lower, upper in zip(fixture_bbox_min, fixture_bbox_max)]
    rotation_pivot = _vector(
        spec.get('pickup_rotation_pivot', default_rotation_pivot),
        size=3,
        name='pickup_rotation_pivot',
    )
    layout_offset = _vector(
        spec.get('pickup_layout_offset', [0.0, 0.0, 0.0]),
        size=3,
        name='pickup_layout_offset',
    )
    pivot_world = _add(
        configured_pickup_origin,
        _quat_rotate(configured_pickup_orientation, rotation_pivot),
    )
    board_bbox_min = _add(
        board_origin,
        _vector(task['optical_board']['bbox_min'], size=3, name='board bbox_min'),
    )
    board_bbox_max = _add(
        board_origin,
        _vector(task['optical_board']['bbox_max'], size=3, name='board bbox_max'),
    )

    robots_by_name = {
        str(robot['name']): robot for robot in robots if isinstance(robot, dict) and robot.get('name') is not None
    }
    base_robot = str(spec.get('base_robot', 'franka_right'))
    assembly_robot = str(spec.get('assembly_robot', 'franka_left'))
    robot_by_role: dict[str, dict[str, Any]] = {}
    robot_position_by_role: dict[str, list[float]] = {}
    for role, robot_name in [('base', base_robot), ('assembly', assembly_robot)]:
        robot = robots_by_name.get(robot_name)
        if robot is None:
            raise ValueError(f'{assembly}: robot {robot_name!r} is not configured.')
        robot_by_role[role] = robot
        robot_position = _vector(
            robot.get('position', [0.0, 0.0, 0.0]),
            size=3,
            name=f'{robot_name} position',
        )
        if bool(robot.get('apply_workspace_offset', True)):
            robot_position = _add(robot_position, workspace_offset)
        robot_position_by_role[role] = robot_position
    approach_height = float(spec.get('approach_height', 0.10))

    evaluated: list[dict[str, Any]] = []
    for yaw_index, yaw_degrees in enumerate(yaw_candidates):
        pickup_orientation = _quat_multiply(
            _yaw_quaternion(yaw_degrees),
            configured_pickup_orientation,
        )
        pickup_origin = _add(
            _add(pivot_world, layout_offset),
            [-value for value in _quat_rotate(pickup_orientation, rotation_pivot)],
        )
        fixture_corners = [
            _add(pickup_origin, _quat_rotate(pickup_orientation, [x, y, z]))
            for x in (fixture_bbox_min[0], fixture_bbox_max[0])
            for y in (fixture_bbox_min[1], fixture_bbox_max[1])
            for z in (fixture_bbox_min[2], fixture_bbox_max[2])
        ]
        fixture_on_board = all(
            board_bbox_min[axis] - 1e-9 <= corner[axis] <= board_bbox_max[axis] + 1e-9
            for corner in fixture_corners
            for axis in (0, 1)
        )
        (selected_move_grasps, move_grasp_diagnostics, move_grasps_feasible,) = _select_move_grasps_for_layout(
            assembly=assembly,
            task=task,
            pickup_origin=pickup_origin,
            pickup_orientation=pickup_orientation,
            assembly_origin=assembly_origin,
            assembly_orientation=assembly_orientation,
            workspace_offset=workspace_offset,
            fixture_bbox_min=fixture_bbox_min,
            fixture_bbox_max=fixture_bbox_max,
            fixture_footprint_margin=move_grasp_fixture_footprint_margin,
            robot=robot_by_role['assembly'],
            robot_position=robot_position_by_role['assembly'],
            orientation_reference=orientation_reference,
            minimum_orientation_continuity=minimum_orientation_continuity,
            maximum_tcp_reach=maximum_pickup_tcp_reach,
            approach_height=approach_height,
            body_start_fraction=move_grasp_body_start_fraction,
            body_sample_count=move_grasp_body_sample_count,
            reselection_minimum_gain=move_grasp_reselection_minimum_gain,
            preferred_interior_clearance=(move_grasp_preferred_interior_clearance),
            minimum_relative_interior_clearance=(move_grasp_minimum_relative_interior_clearance),
            minimum_fixture_clearance=move_grasp_minimum_fixture_clearance,
            minimum_relative_orientation_continuity=(move_grasp_minimum_relative_orientation_continuity),
            transport_tcp_height=_transport_tcp_height(
                spec,
                pickup_origin=pickup_origin,
                assembly_origin=assembly_origin,
            ),
            ik_lift_distance=move_grasp_ik_lift_distance,
            ik_position_tolerance=move_grasp_ik_position_tolerance,
            ik_orientation_tolerance=move_grasp_ik_orientation_tolerance,
            ik_minimum_manipulability=move_grasp_ik_minimum_manipulability,
            ik_max_iterations=move_grasp_ik_max_iterations,
        )
        move_grasps = {
            str(step['move_part']): selected_move_grasps.get(
                str(step['move_part']),
                step['move_grasp'],
            )
            for step in task['assembly_steps']
        }
        for base_candidate in base_candidates:
            grasp_by_part = {base_part: base_candidate, **move_grasps}
            continuity_by_part: dict[str, float] = {}
            pickup_tcp_orientation_by_part: dict[str, list[float]] = {}
            pickup_tcp_reach_by_part: dict[str, float] = {}
            for part_id, grasp in grasp_by_part.items():
                part_position, part_orientation = _part_pickup_pose(
                    parts_by_id[part_id],
                    pickup_origin=pickup_origin,
                    pickup_orientation=pickup_orientation,
                )
                tcp_orientation = _quat_multiply(
                    part_orientation,
                    _quat_conjugate(
                        _vector(
                            grasp['object_in_tcp_orientation'],
                            size=4,
                            name=f'{assembly} part {part_id} object_in_tcp_orientation',
                        )
                    ),
                )
                pickup_tcp_orientation_by_part[part_id] = tcp_orientation
                continuity_by_part[part_id] = _orientation_continuity(
                    tcp_orientation,
                    orientation_reference,
                )
                tcp_position = _add(
                    _add(
                        part_position,
                        [
                            -value
                            for value in _quat_rotate(
                                tcp_orientation,
                                _vector(
                                    grasp['object_in_tcp_position'],
                                    size=3,
                                    name=(f'{assembly} part {part_id} ' 'object_in_tcp_position'),
                                ),
                            )
                        ],
                    ),
                    [0.0, 0.0, approach_height],
                )
                tcp_position = _add(tcp_position, workspace_offset)
                robot_role = 'base' if part_id == base_part else 'assembly'
                robot_position = robot_position_by_role[robot_role]
                pickup_tcp_reach_by_part[part_id] = math.sqrt(
                    sum((tcp_position[axis] - robot_position[axis]) ** 2 for axis in range(3))
                )

            approach_direction = _vector(
                base_candidate['assembly_approach_direction'],
                size=3,
                name=f'{assembly} base grasp assembly_approach_direction',
            )
            base_pickup_orientation = _quat_multiply(
                pickup_orientation,
                _vector(
                    parts_by_id[base_part]['pickup_orientation'],
                    size=4,
                    name=f'{assembly} base pickup_orientation',
                ),
            )
            assembly_approach_cosine = _quat_rotate(
                assembly_orientation,
                approach_direction,
            )[2]
            pickup_approach_cosine = _quat_rotate(
                base_pickup_orientation,
                approach_direction,
            )[2]
            vertical_clearance = min(
                assembly_approach_cosine,
                pickup_approach_cosine,
            )
            interior_clearance = float(base_candidate['interior_clearance_score'])
            support_clearance = float(
                _vector(
                    base_candidate['grasp_center_m'],
                    size=3,
                    name=f'{assembly} base grasp_center_m',
                )[2]
                - base_bbox_min[2]
            )
            support_clearance_ratio = support_clearance / maximum_support_clearance
            worst_continuity = min(continuity_by_part.values())
            base_orientation_continuity = continuity_by_part[base_part]
            worst_pickup_tcp_reach = max(pickup_tcp_reach_by_part.values())
            selected_move_metrics = [
                diagnostics['selected']
                for diagnostics in move_grasp_diagnostics.values()
                if diagnostics.get('selected') is not None
            ]
            maximum_move_tcp_reach = max(
                (float(metrics['maximum_tcp_reach']) for metrics in selected_move_metrics),
                default=worst_pickup_tcp_reach,
            )
            minimum_move_physical_score = min(
                (float(metrics['physical_score']) for metrics in selected_move_metrics),
                default=-math.inf,
            )
            evaluated.append(
                {
                    'yaw_index': yaw_index,
                    'pickup_yaw_degrees': yaw_degrees,
                    'pickup_origin': pickup_origin,
                    'pickup_orientation': pickup_orientation,
                    'base_grasp': base_candidate,
                    'assembly_approach_cosine': assembly_approach_cosine,
                    'pickup_approach_cosine': pickup_approach_cosine,
                    'vertical_clearance_score': vertical_clearance,
                    'interior_clearance_score': interior_clearance,
                    'support_clearance': support_clearance,
                    'support_clearance_ratio': support_clearance_ratio,
                    'orientation_continuity_score': worst_continuity,
                    'base_orientation_continuity_score': (base_orientation_continuity),
                    'orientation_continuity_by_part': continuity_by_part,
                    'pickup_tcp_orientation_by_part': pickup_tcp_orientation_by_part,
                    'maximum_pickup_tcp_reach': worst_pickup_tcp_reach,
                    'pickup_tcp_reach_by_part': pickup_tcp_reach_by_part,
                    'maximum_move_tcp_reach': maximum_move_tcp_reach,
                    'minimum_move_physical_score': minimum_move_physical_score,
                    'selected_move_grasps': selected_move_grasps,
                    'move_grasp_diagnostics': move_grasp_diagnostics,
                    'move_grasps_feasible': move_grasps_feasible,
                    'fixture_on_optical_board': fixture_on_board,
                    'feasible': (
                        vertical_clearance >= minimum_vertical_clearance
                        and interior_clearance >= minimum_interior_clearance
                        and support_clearance_ratio >= minimum_support_clearance_ratio
                        and worst_continuity >= minimum_orientation_continuity
                        and worst_pickup_tcp_reach <= maximum_pickup_tcp_reach
                        and maximum_move_tcp_reach <= maximum_pickup_tcp_reach
                        and move_grasps_feasible
                        and fixture_on_board
                    ),
                }
            )

    feasible = [candidate for candidate in evaluated if candidate['feasible']]
    if not feasible:
        best = max(
            evaluated,
            key=lambda candidate: (
                candidate['orientation_continuity_score'],
                candidate['base_orientation_continuity_score'],
                candidate['vertical_clearance_score'],
                candidate['support_clearance_ratio'],
                -candidate['maximum_pickup_tcp_reach'],
                candidate['interior_clearance_score'],
                candidate['minimum_move_physical_score'],
            ),
        )
        raise ValueError(
            f'{assembly}: no pickup yaw/base grasp pair satisfies the runtime thresholds; '
            f'best yaw={best["pickup_yaw_degrees"]:g}, '
            f'grasp={best["base_grasp"]["grasp_id"]}, '
            f'continuity={best["orientation_continuity_score"]:.3f}, '
            f'vertical={best["vertical_clearance_score"]:.3f}, '
            f'interior={best["interior_clearance_score"]:.3f}, '
            f'support_clearance_ratio={best["support_clearance_ratio"]:.3f}, '
            f'reach={best["maximum_pickup_tcp_reach"]:.3f}, '
            f'fixture_on_board={best["fixture_on_optical_board"]}.'
        )
    selected = max(
        feasible,
        key=lambda candidate: (
            candidate['orientation_continuity_score'],
            candidate['base_orientation_continuity_score'],
            candidate['vertical_clearance_score'],
            candidate['support_clearance_ratio'],
            -candidate['maximum_pickup_tcp_reach'],
            candidate['interior_clearance_score'],
            candidate['minimum_move_physical_score'],
            -candidate['yaw_index'],
            -int(candidate['base_grasp']['grasp_id']),
        ),
    )
    selected_grasp = copy.deepcopy(selected['base_grasp'])
    selected_grasp.update(
        {
            'selection_method': 'joint_pickup_yaw_base_grasp_runtime_selection',
            'pickup_yaw_degrees': selected['pickup_yaw_degrees'],
            'assembly_approach_cosine': selected['assembly_approach_cosine'],
            'pickup_approach_cosine': selected['pickup_approach_cosine'],
            'clearance_score': selected['vertical_clearance_score'],
            'support_clearance': selected['support_clearance'],
            'support_clearance_ratio': selected['support_clearance_ratio'],
            'maximum_pickup_tcp_reach': selected['maximum_pickup_tcp_reach'],
            'orientation_continuity_minimum': minimum_orientation_continuity,
            'orientation_continuity_score': selected['orientation_continuity_score'],
            'orientation_reference': copy.deepcopy(orientation_reference),
            'pickup_tcp_orientation': selected['pickup_tcp_orientation_by_part'][base_part],
        }
    )
    diagnostics = {
        'pickup_yaw_degrees': selected['pickup_yaw_degrees'],
        'configured_pickup_origin': configured_pickup_origin,
        'pickup_origin': selected['pickup_origin'],
        'pickup_orientation': selected['pickup_orientation'],
        'pickup_rotation_pivot': rotation_pivot,
        'pickup_layout_offset': layout_offset,
        'base_grasp_id': int(selected_grasp['grasp_id']),
        'base_grasp_candidate_count': len(base_candidates),
        'evaluated_pair_count': len(evaluated),
        'feasible_pair_count': len(feasible),
        'minimum_vertical_clearance': minimum_vertical_clearance,
        'minimum_interior_clearance': minimum_interior_clearance,
        'minimum_support_clearance_ratio': minimum_support_clearance_ratio,
        'maximum_support_clearance': maximum_support_clearance,
        'minimum_orientation_continuity': minimum_orientation_continuity,
        'orientation_reference': copy.deepcopy(orientation_reference),
        'pickup_tcp_maximum_reach': maximum_pickup_tcp_reach,
        'maximum_pickup_tcp_reach': selected['maximum_pickup_tcp_reach'],
        'pickup_tcp_reach_by_part': selected['pickup_tcp_reach_by_part'],
        'maximum_move_tcp_reach': selected['maximum_move_tcp_reach'],
        'minimum_move_physical_score': selected['minimum_move_physical_score'],
        'move_grasp_body_start_fraction': move_grasp_body_start_fraction,
        'move_grasp_body_sample_count': move_grasp_body_sample_count,
        'move_grasp_reselection_minimum_gain': move_grasp_reselection_minimum_gain,
        'move_grasp_preferred_interior_clearance': (move_grasp_preferred_interior_clearance),
        'move_grasp_minimum_relative_interior_clearance': (move_grasp_minimum_relative_interior_clearance),
        'move_grasp_fixture_footprint_margin': (move_grasp_fixture_footprint_margin),
        'move_grasp_minimum_fixture_clearance': move_grasp_minimum_fixture_clearance,
        'move_grasp_minimum_relative_orientation_continuity': (move_grasp_minimum_relative_orientation_continuity),
        'move_grasp_ik_position_tolerance': move_grasp_ik_position_tolerance,
        'move_grasp_ik_orientation_tolerance': (move_grasp_ik_orientation_tolerance),
        'move_grasp_ik_max_iterations': move_grasp_ik_max_iterations,
        'move_grasp_ik_minimum_manipulability': (move_grasp_ik_minimum_manipulability),
        'move_grasp_ik_lift_distance': move_grasp_ik_lift_distance,
        'move_grasp_selection': copy.deepcopy(selected['move_grasp_diagnostics']),
        'fixture_on_optical_board': selected['fixture_on_optical_board'],
        'vertical_clearance_score': selected['vertical_clearance_score'],
        'interior_clearance_score': selected['interior_clearance_score'],
        'support_clearance': selected['support_clearance'],
        'support_clearance_ratio': selected['support_clearance_ratio'],
        'orientation_continuity_score': selected['orientation_continuity_score'],
        'base_orientation_continuity_score': selected['base_orientation_continuity_score'],
        'orientation_continuity_by_part': selected['orientation_continuity_by_part'],
    }
    return (
        selected['pickup_origin'],
        selected['pickup_orientation'],
        selected_grasp,
        copy.deepcopy(selected['selected_move_grasps']),
        diagnostics,
    )


def _part_pickup_pose(
    part: dict[str, Any],
    *,
    pickup_origin: list[float],
    pickup_orientation: list[float],
) -> tuple[list[float], list[float]]:
    return _compose_pose(
        pickup_origin,
        pickup_orientation,
        _vector(part['pickup_position'], size=3, name='part pickup_position'),
        _vector(part['pickup_orientation'], size=4, name='part pickup_orientation'),
    )


def _part_object(
    assembly: str,
    part: dict[str, Any],
    *,
    pickup_origin: list[float],
    pickup_orientation: list[float],
) -> dict[str, Any]:
    position, orientation = _part_pickup_pose(
        part,
        pickup_origin=pickup_origin,
        pickup_orientation=pickup_orientation,
    )
    return {
        'name': f'fabrica_{assembly}_{part["part_id"]}',
        'kind': 'usd',
        'prim_path': f'/fabrica_{assembly}_{part["part_id"]}',
        'usd_path': _asset_path(part['usd_path']),
        'position': position,
        'orientation': orientation,
        'scale': [1.0, 1.0, 1.0],
        'collider': True,
        'auto_collider': False,
        'rigid_body': True,
        'tracked': True,
        'static_friction': 0.8,
        'dynamic_friction': 0.7,
        'restitution': 0.0,
        'linear_damping': 1.0,
        'angular_damping': 2.0,
        'sleep_threshold': 0.01,
        'stabilization_threshold': 0.001,
        'solver_position_iteration_count': 16,
        'solver_velocity_iteration_count': 4,
        'fabrica_part_id': str(part['part_id']),
        'fabrica_bbox_min': copy.deepcopy(part['bbox_min']),
        'fabrica_bbox_max': copy.deepcopy(part['bbox_max']),
    }


def _static_asset_object(
    *,
    name: str,
    usd_path: str,
    position: list[float],
    orientation: list[float] | None = None,
    friction: float,
) -> dict[str, Any]:
    return {
        'name': name,
        'kind': 'usd',
        'prim_path': f'/{name}',
        'usd_path': _asset_path(usd_path),
        'position': copy.deepcopy(position),
        'orientation': copy.deepcopy(orientation or [1.0, 0.0, 0.0, 0.0]),
        'scale': [1.0, 1.0, 1.0],
        'collider': True,
        'auto_collider': False,
        'rigid_body': False,
        'tracked': False,
        'static_friction': float(friction),
        'dynamic_friction': float(friction),
        'restitution': 0.0,
    }


def _target(name: str, position: list[float], orientation: list[float]) -> dict[str, Any]:
    return {
        'name': name,
        'reference': 'world',
        'position': copy.deepcopy(position),
        'orientation': copy.deepcopy(orientation),
    }


def _transport_tcp_height(
    spec: dict[str, Any],
    *,
    pickup_origin: list[float],
    assembly_origin: list[float],
) -> float:
    work_surface_height = max(float(pickup_origin[2]), float(assembly_origin[2]))
    configured_height = spec.get('transport_tcp_height')
    if configured_height is None:
        clearance_height = float(spec.get('transport_clearance_height', 0.34))
        value = work_surface_height + clearance_height
    else:
        clearance_height = None
        value = float(configured_height)
    if clearance_height is not None and (not math.isfinite(clearance_height) or clearance_height <= 0.0):
        raise ValueError('transport_clearance_height must be finite and positive.')
    if not math.isfinite(value) or value <= work_surface_height:
        raise ValueError('transport_tcp_height must be finite and above both pickup and assembly origins.')
    return value


def _object_position_at_tcp_height(
    *,
    object_position: list[float],
    object_orientation: list[float],
    grasp: dict[str, Any],
    tcp_height: float,
) -> list[float]:
    relative_position = _vector(
        grasp['object_in_tcp_position'],
        size=3,
        name='grasp object_in_tcp_position',
    )
    relative_orientation = _vector(
        grasp['object_in_tcp_orientation'],
        size=4,
        name='grasp object_in_tcp_orientation',
    )
    tcp_orientation = _quat_multiply(
        object_orientation,
        _quat_conjugate(relative_orientation),
    )
    object_to_tcp = [-value for value in _quat_rotate(tcp_orientation, relative_position)]
    result = copy.deepcopy(object_position)
    result[2] = float(tcp_height - object_to_tcp[2])
    return result


def _skill_phase(
    *,
    skill: str,
    robot: str,
    phase_name: str,
    timeout_steps: int,
    parameters: dict[str, Any],
    phase_actions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return UR5eAssemblySkillAPI.compile_call(
        {
            'skill': skill,
            'robot': robot,
            'phase_name': phase_name,
            'timeout_steps': timeout_steps,
            'phase_actions': phase_actions or {},
            **parameters,
        }
    )


def _grasp_parameters(object_name: str, grasp: dict[str, Any]) -> dict[str, Any]:
    return {
        'object': object_name,
        'grasp_relative_position': copy.deepcopy(grasp['object_in_tcp_position']),
        'grasp_relative_orientation': copy.deepcopy(grasp['object_in_tcp_orientation']),
        'lock_target_position': True,
        'lock_target_orientation': True,
        'guard_ik_branch_jump': True,
        'ik_reference_mode': 'current',
        'use_command_warm_start': True,
        'require_warm_start_ik': True,
    }


def _append_pick_and_attach_phases(
    phases: list[dict[str, Any]],
    *,
    prefix: str,
    robot: str,
    object_name: str,
    grasp: dict[str, Any],
    part: dict[str, Any],
    approach_height: float,
    move_above_timeout_steps: int,
    move_above_orientation_first: bool,
    move_above_orientation_first_steps: int,
    move_above_orientation_first_tolerance: float,
    preshape_timeout_steps: int,
    preshape_open_margin: float,
    preshape_gripper_position_tolerance: float,
    descend_position_tolerance: float,
    descend_relaxed_position_tolerance: float,
    descend_relaxed_after_steps: int,
    prealign_steps: int,
    prealign_joint_positions: list[float] | None,
    prealign_shoulder_pan: float | None,
) -> None:
    grasp_parameters = _grasp_parameters(object_name, grasp)
    preclose_openness = min(
        1.0,
        float(grasp['robotiq_open_ratio']) + float(preshape_open_margin),
    )
    closed_openness = max(0.0, float(grasp['robotiq_open_ratio']) - 0.035)
    closed_joint_position = 0.8 * (1.0 - closed_openness)
    contact_box_offset, contact_box_scale = _contact_box(part)
    contact_parameters = {
        'require_dual_finger_contact': True,
        # Official Fabrica grasps commonly cage a convex corner with the two
        # pads on adjacent faces. Both pads must still satisfy the strict
        # surface-gap check before the grasp can be registered.
        'allow_cross_axis_dual_finger_contact': True,
        'finger_contact_distance': 0.008,
        'physical_attach_surface_gap': 0.006,
        'contact_force_threshold': 0.15,
        'measure_force_contact': True,
        'contact_box_offset': contact_box_offset,
        'contact_box_scale': contact_box_scale,
    }

    move_above_parameters = {
        **grasp_parameters,
        'ik_reference_mode': 'hybrid',
        'ik_reference_command_max_tracking_error': 0.12,
        'cartesian_orientation_command_warm_start': True,
        'cartesian_orientation_command_lookahead': 0.36,
        'max_command_joint_step': 0.06,
        'max_command_tracking_error': 0.24,
        'max_wrist_command_tracking_error': 0.24,
        'offset': [0.0, 0.0, float(approach_height)],
        'offset_frame': 'world',
        'gripper_command': 'open',
        'position_tolerance': 0.012,
        'orientation_tolerance': 0.12,
        'orientation_first_before_translation': bool(move_above_orientation_first),
        'orientation_first_max_steps': int(move_above_orientation_first_steps),
        'orientation_first_tolerance': float(move_above_orientation_first_tolerance),
    }
    if prealign_steps > 0:
        if prealign_joint_positions is not None:
            move_above_parameters.update(
                {
                    'prealign_steps': int(prealign_steps),
                    'prealign_joint_positions': copy.deepcopy(prealign_joint_positions),
                    'prealign_max_joint_step': 0.035,
                }
            )
        elif prealign_shoulder_pan is not None:
            move_above_parameters.update(
                {
                    'prealign_steps': int(prealign_steps),
                    'prealign_shoulder_pan': float(prealign_shoulder_pan),
                    'prealign_max_joint_step': 0.035,
                }
            )
    phases.append(
        _skill_phase(
            skill='move_above_part',
            robot=robot,
            phase_name=f'{prefix}_move_above',
            timeout_steps=move_above_timeout_steps,
            parameters=move_above_parameters,
        )
    )
    phases.append(
        _skill_phase(
            skill='preshape_gripper',
            robot=robot,
            phase_name=f'{prefix}_preshape',
            timeout_steps=preshape_timeout_steps + 24,
            parameters={
                'object': object_name,
                'gripper_openness': preclose_openness,
                'hold_steps': 36,
                'preshape_timeout_steps': preshape_timeout_steps,
                'gripper_position_tolerance': float(preshape_gripper_position_tolerance),
            },
        )
    )
    phases.append(
        _skill_phase(
            skill='descend_to_grasp',
            robot=robot,
            phase_name=f'{prefix}_descend',
            timeout_steps=900,
            parameters={
                **grasp_parameters,
                'gripper_command': preclose_openness,
                'position_tolerance': float(descend_position_tolerance),
                'relaxed_position_tolerance': float(descend_relaxed_position_tolerance),
                'relaxed_position_tolerance_after_steps': int(descend_relaxed_after_steps),
                'orientation_tolerance': 0.10,
                'target_object_position_tolerance': 0.012,
                'target_object_orientation_tolerance': 0.15,
            },
        )
    )

    tcp_position = copy.deepcopy(grasp['tcp_in_assembly_position'])
    tcp_orientation = copy.deepcopy(grasp['tcp_in_assembly_orientation'])
    attach_spec = {
        'object': object_name,
        'robot': robot,
        'target': {
            'reference': object_name,
            'offset': tcp_position,
            'orientation': tcp_orientation,
            'ik_frame_compensation': 'none',
        },
        'attachment_mode': 'fixed_joint',
        'attachment_relative_pose_source': 'current',
        'disable_collision_on_attach': False,
        # Keep finger/object contacts active while the fixed joint transports the
        # part.  Insertion compliance can then remove only the joint near the
        # socket without introducing a new collision pair and a PhysX impulse.
        'filter_gripper_collisions_on_attach': False,
        'compliant_hold_linear_limit': 0.006,
        'compliant_hold_angular_limit_degrees': 6.0,
        'compliant_hold_linear_max_force': 20.0,
        'compliant_hold_linear_damping': 10.0,
        'compliant_hold_linear_stiffness': 500.0,
        'compliant_hold_angular_max_force': 2.0,
        'compliant_hold_angular_damping': 0.2,
        'compliant_hold_angular_stiffness': 5.0,
        'compliant_hold_gravity_force_multiplier': 6.0,
        'compliant_hold_drive_damping_ratio': 1.0,
        'compliant_hold_torque_force_fraction': 0.5,
        'compliant_hold_linear_force_cap': 120.0,
        'compliant_hold_angular_force_cap': 12.0,
        'require_contact': True,
        'require_physical_contact': True,
        'require_local_skill_complete_for_attach': True,
        'require_target_reached_for_attach': True,
        'allow_strict_contact_target_refinement': True,
        'strict_contact_target_refinement_max_distance': 0.025,
        'strict_contact_target_refinement_tracking_tolerance': 0.00035,
        'allow_noncontact_fixed_joint': False,
        'gripper_closed_threshold': min(0.8, closed_joint_position + 0.04),
        **contact_parameters,
        'min_attach_steps': 36,
        'position_tolerance': float(descend_position_tolerance),
        'orientation_tolerance': 0.10,
        'support_height_tolerance': None,
        'top_clearance': None,
    }
    close_phase = _skill_phase(
        skill='close_gripper',
        robot=robot,
        phase_name=f'{prefix}_close_and_attach',
        timeout_steps=480,
        parameters={
            **grasp_parameters,
            'preclose_openness': preclose_openness,
            'closed_openness': closed_openness,
            'close_until_contact': True,
            'close_until_contact_min_steps': 24,
            'close_until_contact_timeout_steps': 240,
            'close_contact_stable_steps': 12,
            'close_steps': 180,
            'close_ramp_steps': 120,
            'require_close_pose_gate': True,
            'close_position_tolerance': float(descend_position_tolerance),
            'close_orientation_tolerance': 0.10,
            'close_gate_hold_refined_command': True,
            'close_gate_track_object_during_close': True,
            'close_gate_recenter_single_finger_contact': True,
            'close_gate_recenter_contact_distance': 0.008,
            'close_gate_recenter_min_gap_imbalance': 0.004,
            'close_gate_recenter_stable_steps': 2,
            'close_gate_recenter_step': 0.00075,
            'close_gate_recenter_max_offset': 0.025,
            'close_gate_recenter_target_tolerance': 0.00035,
            'require_grasp_contact': True,
            'require_strict_physical_contact': True,
            **contact_parameters,
        },
        phase_actions={
            'unlock': [object_name],
            'gripper_commands': {robot: 'close'},
            'attach': [attach_spec],
        },
    )
    close_phase['advance'] = {
        'type': 'all_of',
        'min_steps': 36,
        'conditions': [
            {
                'type': 'local_skill_complete',
                'robot': robot,
                'skill': 'ur5e_close_gripper',
                'min_steps': 36,
            },
            {'type': 'object_attached', 'object': object_name, 'robot': robot},
        ],
    }
    phases.append(close_phase)
    phases.append(
        _skill_phase(
            skill='retreat_vertical',
            robot=robot,
            phase_name=f'{prefix}_lift',
            timeout_steps=1200,
            parameters={
                'object': object_name,
                'requires_held_object': True,
                'relative_to_current_tcp': True,
                'offset': [0.0, 0.0, 0.12],
                'offset_frame': 'world',
                'gripper_command': 'contact_hold',
                'position_tolerance': 0.012,
            },
        )
    )


def _append_transport_phase(
    phases: list[dict[str, Any]],
    *,
    phase_name: str,
    robot: str,
    object_name: str,
    target_name: str,
    insertion: bool,
    timeout_steps: int,
    strict_alignment: bool = False,
    insertion_relaxed_position_tolerance: float = 0.018,
    insertion_relaxed_after_steps: int = 600,
    insertion_axis: list[float] | None = None,
    insertion_lateral_position_tolerance: float | None = None,
    insertion_cartesian_position_step: float = 0.00025,
    insertion_lateral_alignment_cartesian_position_step: float = 0.00025,
    insertion_lateral_alignment_enter_tolerance: float | None = None,
    insertion_lateral_alignment_exit_tolerance: float | None = None,
    insertion_path_depth: float = 0.0,
    insertion_lateral_alignment_axial_clearance: float = 0.0,
    insertion_axial_recovery_cartesian_position_step: float = 0.001,
    insertion_axial_recovery_deadband: float = 0.0005,
    insertion_compliance_capture_max_linear_speed: float = 0.10,
    insertion_compliance_capture_max_angular_speed: float = 2.0,
    insertion_compliance_capture_stable_steps: int = 8,
    insertion_compliance_position_tolerance: float = 0.015,
    insertion_compliance_after_steps: int = 0,
    insertion_compliance_require_waypoint_proximity: bool = False,
    insertion_compliance_waypoint_position_tolerance: float = 0.010,
    insertion_compliance_waypoint_axial_position_tolerance: float | None = None,
    insertion_compliance_waypoint_lateral_position_tolerance: float | None = None,
    insertion_compliance_geometric_capture_after_steps: int = 0,
    insertion_compliance_minimum_gravity_alignment: float = 0.70,
    insertion_compliant_alignment_retraction_limit: float = 0.006,
    insertion_compliant_track_object_orientation: bool = False,
    target_object_entry_capture_max_steps: int = 0,
    final_target_name: str | None = None,
    settle_at_target: bool = True,
) -> None:
    position_tolerance = 0.006 if insertion or strict_alignment else 0.014
    target_object_position_tolerance = 0.008 if insertion or strict_alignment else 0.014
    relaxed_parameters = {}
    if insertion:
        relaxed_parameters = {
            'relaxed_position_tolerance': float(insertion_relaxed_position_tolerance),
            'relaxed_target_object_position_tolerance': float(insertion_relaxed_position_tolerance),
            'relaxed_position_tolerance_after_steps': int(insertion_relaxed_after_steps),
        }
        if insertion_axis is not None:
            relaxed_parameters['target_object_convergence_axis'] = _normalized_direction(
                insertion_axis,
                name=f'{phase_name} insertion axis',
            )
        if insertion_lateral_position_tolerance is not None:
            relaxed_parameters['target_object_lateral_position_tolerance'] = float(insertion_lateral_position_tolerance)
            relaxed_parameters['target_object_lateral_alignment_cartesian_position_step'] = float(
                insertion_lateral_alignment_cartesian_position_step
            )
            relaxed_parameters['target_object_lateral_alignment_axial_clearance'] = float(
                insertion_lateral_alignment_axial_clearance
            )
            relaxed_parameters['target_object_insertion_path_depth'] = float(insertion_path_depth)
            relaxed_parameters['target_object_lateral_alignment_stable_steps'] = 8 if settle_at_target else 1
            relaxed_parameters['target_object_axial_recovery_cartesian_position_step'] = float(
                insertion_axial_recovery_cartesian_position_step
            )
            relaxed_parameters['target_object_axial_recovery_deadband'] = float(insertion_axial_recovery_deadband)
            alignment_enter_tolerance = (
                0.5 * float(insertion_lateral_position_tolerance)
                if insertion_lateral_alignment_enter_tolerance is None
                else float(insertion_lateral_alignment_enter_tolerance)
            )
            alignment_exit_tolerance = (
                float(insertion_lateral_position_tolerance)
                if insertion_lateral_alignment_exit_tolerance is None
                else float(insertion_lateral_alignment_exit_tolerance)
            )
            relaxed_parameters['target_object_lateral_alignment_enter_tolerance'] = alignment_enter_tolerance
            relaxed_parameters['target_object_lateral_alignment_exit_tolerance'] = alignment_exit_tolerance
    phases.append(
        _skill_phase(
            skill='move_part_to_target',
            robot=robot,
            phase_name=phase_name,
            timeout_steps=int(timeout_steps),
            parameters={
                'object': object_name,
                'requires_held_object': True,
                'target_object_target': target_name,
                'gripper_command': 'contact_hold',
                'lock_target_orientation': True,
                'derive_tcp_orientation_from_target_object': True,
                'guard_ik_branch_jump': True,
                'ik_reference_mode': 'hybrid',
                'ik_reference_command_max_tracking_error': 0.12,
                'use_command_warm_start': True,
                'require_warm_start_ik': True,
                'cartesian_orientation_command_warm_start': True,
                'cartesian_orientation_command_lookahead': 0.36,
                'target_object_use_measured_orientation_for_position_servo': True,
                'max_command_joint_step': 0.035 if insertion else 0.06,
                'max_command_tracking_error': 0.12 if insertion else 0.24,
                'max_wrist_command_tracking_error': 0.10 if insertion else 0.24,
                'cartesian_position_step': (
                    float(insertion_cartesian_position_step) if insertion else 0.003 if strict_alignment else 0.006
                ),
                'cartesian_orientation_step': 0.012 if insertion else 0.025,
                'target_object_servo_position_command_warm_start': bool(insertion),
                'target_object_servo_position_command_gate_overdrive': bool(insertion),
                'target_object_servo_position_command_lookahead': 0.004,
                'target_object_servo_position_command_accumulation_step': 0.0001,
                'max_joint_step': 0.08 if insertion else 0.14,
                'max_object_tcp_slip': 0.04,
                'position_tolerance': position_tolerance,
                **relaxed_parameters,
                'orientation_tolerance': (0.08 if insertion else 0.10 if strict_alignment else 0.12),
                'require_target_object_pose_convergence': True,
                'target_object_position_tolerance': target_object_position_tolerance,
                'target_object_orientation_tolerance': 0.10 if insertion else 0.15,
                'require_target_object_static': bool(insertion),
                'target_object_max_linear_speed': 0.03,
                'target_object_max_angular_speed': 2.0,
                'target_object_allow_pose_stable_override': True,
                'target_object_stable_steps': 8 if settle_at_target else 1,
                'hold_for_target_object_settle': bool(insertion),
                'target_object_settle_hold_steps': 48,
                'target_object_settle_retry_servo_steps': 8,
                **(
                    {
                        'target_object_final_target': str(final_target_name),
                        'relax_fixed_attachment_within_final_position_tolerance': float(
                            insertion_compliance_position_tolerance
                        ),
                        'relax_fixed_attachment_after_steps': int(insertion_compliance_after_steps),
                        'relax_fixed_attachment_require_waypoint_proximity': bool(
                            insertion_compliance_require_waypoint_proximity
                        ),
                        'relax_fixed_attachment_waypoint_position_tolerance': float(
                            insertion_compliance_waypoint_position_tolerance
                        ),
                        'relax_fixed_attachment_geometric_capture_after_steps': int(
                            insertion_compliance_geometric_capture_after_steps
                        ),
                        'relax_fixed_attachment_minimum_gravity_alignment': float(
                            insertion_compliance_minimum_gravity_alignment
                        ),
                        **(
                            {
                                'relax_fixed_attachment_waypoint_axial_position_tolerance': float(
                                    insertion_compliance_waypoint_axial_position_tolerance
                                ),
                                'relax_fixed_attachment_waypoint_lateral_position_tolerance': float(
                                    insertion_compliance_waypoint_lateral_position_tolerance
                                ),
                            }
                            if insertion_compliance_waypoint_axial_position_tolerance is not None
                            and insertion_compliance_waypoint_lateral_position_tolerance is not None
                            else {}
                        ),
                        'relax_fixed_attachment_final_orientation_tolerance': 0.15,
                        'relax_fixed_attachment_max_linear_speed': float(insertion_compliance_capture_max_linear_speed),
                        'relax_fixed_attachment_max_angular_speed': float(
                            insertion_compliance_capture_max_angular_speed
                        ),
                        'relax_fixed_attachment_stable_steps': int(insertion_compliance_capture_stable_steps),
                        'relax_fixed_attachment_allow_pose_stable_override': True,
                        'compliant_servo_pause_linear_speed': 0.20,
                        'compliant_servo_pause_angular_speed': 5.0,
                        'compliant_servo_resume_linear_speed': 0.03,
                        'compliant_servo_resume_angular_speed': 2.0,
                        'compliant_servo_resume_stable_steps': 8,
                        'compliant_servo_allow_pose_stable_resume': True,
                        'target_object_allow_pose_history_velocity_override': True,
                        'pose_history_velocity_override_position_tolerance': 0.0005,
                        'pose_history_velocity_override_orientation_tolerance': 0.01,
                        'compliant_servo_settle_max_linear_speed': float(insertion_compliance_capture_max_linear_speed),
                        'compliant_servo_settle_max_angular_speed': min(
                            float(insertion_compliance_capture_max_angular_speed),
                            2.0,
                        ),
                        'compliant_servo_settle_stable_steps': 24,
                        'compliant_servo_velocity_rate_limit': True,
                        'compliant_servo_minimum_step_scale': 0.2,
                        'compliant_servo_max_position_step': 0.0005,
                        'compliant_servo_position_command_warm_start': True,
                        'compliant_servo_position_command_gate_overdrive': True,
                        'compliant_servo_position_command_lookahead': 0.004,
                        'compliant_servo_position_command_accumulation_step': 0.0001,
                        'compliant_servo_max_lateral_step': 0.0005,
                        'compliant_servo_max_alignment_retraction': float(
                            insertion_compliant_alignment_retraction_limit
                        ),
                        'compliant_servo_max_orientation_step': 0.002,
                        'compliant_servo_orientation_correction_deadband': 0.005,
                        'compliant_servo_hold_orientation_during_lateral_alignment': True,
                        'compliant_servo_track_object_orientation': bool(insertion_compliant_track_object_orientation),
                    }
                    if insertion and final_target_name is not None
                    else {}
                ),
                **(
                    {'target_object_entry_capture_max_steps': int(target_object_entry_capture_max_steps)}
                    if insertion and target_object_entry_capture_max_steps > 0
                    else {}
                ),
            },
        )
    )


def _append_release_and_retreat(
    phases: list[dict[str, Any]],
    *,
    prefix: str,
    robot: str,
    object_name: str,
    final_target: str,
    assembled_objects: list[str],
    retreat_offset: list[float],
    park_offset: list[float],
    park_workspace_center: list[float],
    park_minimum_planar_radius: float,
) -> None:
    phases.append(
        {
            'name': f'{prefix}_release_and_lock',
            'timeout_steps': 720,
            'robot_targets': {},
            'gripper_commands': {robot: 'open'},
            'lock': [
                {
                    'object': object_name,
                    'target': final_target,
                    'position_tolerance': 0.015,
                    'orientation_tolerance': 0.12,
                    'snap_on_open': False,
                    'freeze_current_pose': True,
                }
            ],
            'advance': {
                'type': 'objects_static',
                'objects': copy.deepcopy(assembled_objects),
                'min_steps': 72,
                'linear_velocity_threshold': 0.05,
                'angular_velocity_threshold': 2.0,
            },
        }
    )
    phases.append(
        _skill_phase(
            skill='retreat_vertical',
            robot=robot,
            phase_name=f'{prefix}_retreat',
            timeout_steps=1200,
            parameters={
                'relative_to_current_tcp': True,
                'offset': copy.deepcopy(retreat_offset),
                'offset_frame': 'world',
                'gripper_command': 'open',
                'position_tolerance': 0.008,
            },
        )
    )
    phases.append(
        _skill_phase(
            skill='retreat_vertical',
            robot=robot,
            phase_name=f'{prefix}_park',
            timeout_steps=1800,
            parameters={
                'relative_to_current_tcp': True,
                'offset': copy.deepcopy(park_offset),
                'offset_frame': 'world',
                'workspace_center': copy.deepcopy(park_workspace_center),
                'workspace_minimum_planar_radius': float(park_minimum_planar_radius),
                'lock_target_position': True,
                'lock_target_orientation': False,
                'gripper_command': 'open',
                'position_tolerance': 0.018,
                'cartesian_position_step': 0.010,
            },
        )
    )


def _compile_targets_and_phases(
    *,
    assembly: str,
    task: dict[str, Any],
    spec: dict[str, Any],
    pickup_origin: list[float],
    pickup_orientation: list[float],
    assembly_origin: list[float],
    assembly_orientation: list[float],
    robots: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict], list[str], list[str]]:
    targets: list[dict] = []
    phases: list[dict] = []
    success: list[dict] = []
    pickup_target_names: list[str] = []
    assembly_target_names: list[str] = []
    parts_by_id = {str(part['part_id']): part for part in task['parts']}
    base_robot = str(spec.get('base_robot', 'franka_right'))
    assembly_robot = str(spec.get('assembly_robot', 'franka_left'))
    release_retreat_distance = float(spec.get('release_retreat_distance', 0.06))
    post_release_park_distance = float(spec.get('post_release_park_distance', 0.35))
    post_release_park_vertical_offset = float(spec.get('post_release_park_vertical_offset', 0.02))
    post_release_park_minimum_planar_radius = float(spec.get('post_release_park_minimum_planar_radius', 0.28))
    if (
        not math.isfinite(release_retreat_distance)
        or release_retreat_distance <= 0.0
        or not math.isfinite(post_release_park_distance)
        or post_release_park_distance <= 0.0
        or not math.isfinite(post_release_park_vertical_offset)
        or post_release_park_vertical_offset < 0.0
        or not math.isfinite(post_release_park_minimum_planar_radius)
        or post_release_park_minimum_planar_radius <= 0.0
    ):
        raise ValueError(
            'Release retreat and park distances must be finite and positive, and the '
            'park vertical offset must be finite and non-negative; the park workspace '
            'radius must be finite and positive.'
        )
    robots_by_name = {
        str(robot['name']): robot for robot in robots if isinstance(robot, dict) and robot.get('name') is not None
    }

    def post_release_park_offset(robot_name: str) -> list[float]:
        robot = robots_by_name.get(robot_name)
        if robot is None:
            raise ValueError(f'Post-release parking references missing robot {robot_name!r}.')
        robot_position = _vector(
            robot.get('position', [0.0, 0.0, 0.0]),
            size=3,
            name=f'{robot_name} position',
        )
        direction = [
            robot_position[0] - assembly_origin[0],
            robot_position[1] - assembly_origin[1],
        ]
        direction_norm = math.hypot(*direction)
        if direction_norm <= 1e-9:
            raise ValueError(f'Cannot derive a post-release parking direction for {robot_name!r}.')
        return [
            post_release_park_distance * direction[0] / direction_norm,
            post_release_park_distance * direction[1] / direction_norm,
            post_release_park_vertical_offset,
        ]

    def post_release_park_workspace_center(robot_name: str) -> list[float]:
        robot = robots_by_name.get(robot_name)
        if robot is None:
            raise ValueError(f'Post-release parking references missing robot {robot_name!r}.')
        return _vector(
            robot.get('position', [0.0, 0.0, 0.0]),
            size=3,
            name=f'{robot_name} position',
        )

    def release_retreat_offset(insertion_axis: list[float]) -> list[float]:
        direction = _normalized_direction(insertion_axis, name='release retreat axis')
        return [-release_retreat_distance * value for value in direction]

    approach_height = float(spec.get('approach_height', 0.10))
    transport_hover_height = float(spec.get('transport_hover_height', 0.12))
    move_above_timeout_steps = int(spec.get('move_above_timeout_steps', 2200))
    move_above_orientation_first = bool(spec.get('move_above_orientation_first', True))
    move_above_orientation_first_steps = int(spec.get('move_above_orientation_first_steps', 720))
    move_above_orientation_first_tolerance = float(spec.get('move_above_orientation_first_tolerance', 0.08))
    preshape_timeout_steps = int(spec.get('preshape_timeout_steps', 720))
    preshape_open_margin = float(spec.get('preshape_open_margin', 0.20))
    preshape_gripper_position_tolerance = float(spec.get('preshape_gripper_position_tolerance', 0.025))
    descend_position_tolerance = float(spec.get('descend_position_tolerance', 0.007))
    descend_relaxed_position_tolerance = float(spec.get('descend_relaxed_position_tolerance', 0.012))
    descend_relaxed_after_steps = int(spec.get('descend_relaxed_after_steps', 600))
    insertion_relaxed_position_tolerance = float(spec.get('insertion_relaxed_position_tolerance', 0.018))
    base_support_release_position_tolerance = float(spec.get('base_support_release_position_tolerance', 0.012))
    base_support_lateral_position_tolerance = float(spec.get('base_support_lateral_position_tolerance', 0.015))
    base_support_lateral_alignment_enter_tolerance = float(
        spec.get('base_support_lateral_alignment_enter_tolerance', 0.002)
    )
    base_support_lateral_alignment_exit_tolerance = float(
        spec.get('base_support_lateral_alignment_exit_tolerance', 0.004)
    )
    base_support_lateral_alignment_cartesian_position_step = float(
        spec.get(
            'base_support_lateral_alignment_cartesian_position_step',
            0.002,
        )
    )
    insertion_relaxed_after_steps = int(spec.get('insertion_relaxed_after_steps', 600))
    insertion_lateral_position_tolerance = float(spec.get('insertion_lateral_position_tolerance', 0.001))
    insertion_lateral_tolerance_object_extent_scale = float(
        spec.get('insertion_lateral_tolerance_object_extent_scale', 0.04)
    )
    intermediate_insertion_lateral_position_tolerance = float(
        spec.get('intermediate_insertion_lateral_position_tolerance', 0.002)
    )
    intermediate_insertion_lateral_alignment_cartesian_position_step = float(
        spec.get(
            'intermediate_insertion_lateral_alignment_cartesian_position_step',
            0.001,
        )
    )
    insertion_cartesian_position_step = float(spec.get('insertion_cartesian_position_step', 0.00025))
    insertion_lateral_alignment_cartesian_position_step = float(
        spec.get('insertion_lateral_alignment_cartesian_position_step', 0.00025)
    )
    insertion_lateral_alignment_entry_clearance = float(spec.get('insertion_lateral_alignment_entry_clearance', 0.01))
    insertion_lateral_alignment_clearance_object_extent_scale = float(
        spec.get(
            'insertion_lateral_alignment_clearance_object_extent_scale',
            1.0,
        )
    )
    insertion_axial_recovery_cartesian_position_step = float(
        spec.get('insertion_axial_recovery_cartesian_position_step', 0.001)
    )
    insertion_axial_recovery_deadband = float(spec.get('insertion_axial_recovery_deadband', 0.0005))
    insertion_compliance_capture_max_linear_speed = float(
        spec.get('insertion_compliance_capture_max_linear_speed', 0.10)
    )
    insertion_compliance_capture_max_angular_speed = float(
        spec.get('insertion_compliance_capture_max_angular_speed', 2.0)
    )
    insertion_compliance_capture_stable_steps = int(spec.get('insertion_compliance_capture_stable_steps', 8))
    insertion_compliance_geometric_capture_after_steps = int(
        spec.get('insertion_compliance_geometric_capture_after_steps', 1200)
    )
    insertion_compliance_minimum_gravity_alignment = float(
        spec.get('insertion_compliance_minimum_gravity_alignment', 0.70)
    )
    insertion_compliant_alignment_retraction_limit = float(
        spec.get('insertion_compliant_alignment_retraction_limit', 0.006)
    )
    insertion_compliant_track_object_orientation = bool(spec.get('insertion_compliant_track_object_orientation', False))
    transport_timeout_steps = int(spec.get('transport_timeout_steps', 4800))
    insertion_timeout_steps = int(spec.get('insertion_timeout_steps', 3600))
    base_place_timeout_steps = int(spec.get('base_place_timeout_steps', 4800))
    move_above_prealign_steps = int(spec.get('move_above_prealign_steps', 0))
    prealign_joint_positions_by_robot = spec.get('prealign_joint_positions_by_robot') or {}
    if not isinstance(prealign_joint_positions_by_robot, dict):
        raise ValueError('prealign_joint_positions_by_robot must be a mapping.')
    for robot_name, positions in prealign_joint_positions_by_robot.items():
        prealign_joint_positions_by_robot[robot_name] = _vector(
            positions,
            size=6,
            name=f'{robot_name} prealign_joint_positions',
        )
    prealign_shoulder_pan_by_robot = spec.get('prealign_shoulder_pan_by_robot') or {}
    if not isinstance(prealign_shoulder_pan_by_robot, dict):
        raise ValueError('prealign_shoulder_pan_by_robot must be a mapping.')
    if (
        move_above_timeout_steps <= 0
        or preshape_timeout_steps <= 0
        or insertion_relaxed_after_steps <= 0
        or transport_timeout_steps <= 0
        or insertion_timeout_steps <= 0
        or base_place_timeout_steps <= 0
        or move_above_orientation_first_steps < 0
        or move_above_prealign_steps < 0
        or not math.isfinite(preshape_open_margin)
        or preshape_open_margin < 0.0
        or preshape_open_margin > 1.0
        or not math.isfinite(preshape_gripper_position_tolerance)
        or preshape_gripper_position_tolerance <= 0.0
        or preshape_gripper_position_tolerance > 0.1
        or not math.isfinite(descend_position_tolerance)
        or descend_position_tolerance <= 0.0
        or not math.isfinite(descend_relaxed_position_tolerance)
        or descend_relaxed_position_tolerance < descend_position_tolerance
        or descend_relaxed_after_steps <= 0
        or not math.isfinite(insertion_relaxed_position_tolerance)
        or insertion_relaxed_position_tolerance < 0.008
        or not math.isfinite(base_support_release_position_tolerance)
        or base_support_release_position_tolerance < 0.008
        or base_support_release_position_tolerance > 0.015
        or not math.isfinite(base_support_lateral_position_tolerance)
        or base_support_lateral_position_tolerance < base_support_release_position_tolerance
        or base_support_lateral_position_tolerance > 0.02
        or not math.isfinite(base_support_lateral_alignment_enter_tolerance)
        or base_support_lateral_alignment_enter_tolerance <= 0.0
        or base_support_lateral_alignment_enter_tolerance > base_support_release_position_tolerance
        or not math.isfinite(base_support_lateral_alignment_exit_tolerance)
        or base_support_lateral_alignment_exit_tolerance < base_support_lateral_alignment_enter_tolerance
        or base_support_lateral_alignment_exit_tolerance > base_support_lateral_position_tolerance
        or not math.isfinite(base_support_lateral_alignment_cartesian_position_step)
        or base_support_lateral_alignment_cartesian_position_step <= 0.0
        or base_support_lateral_alignment_cartesian_position_step > base_support_lateral_alignment_enter_tolerance
        or not math.isfinite(insertion_lateral_position_tolerance)
        or insertion_lateral_position_tolerance <= 0.0
        or insertion_lateral_position_tolerance > insertion_relaxed_position_tolerance
        or not math.isfinite(insertion_lateral_tolerance_object_extent_scale)
        or insertion_lateral_tolerance_object_extent_scale < 0.0
        or insertion_lateral_tolerance_object_extent_scale > 0.25
        or not math.isfinite(intermediate_insertion_lateral_position_tolerance)
        or intermediate_insertion_lateral_position_tolerance < insertion_lateral_position_tolerance
        or intermediate_insertion_lateral_position_tolerance > 0.005
        or not math.isfinite(intermediate_insertion_lateral_alignment_cartesian_position_step)
        or intermediate_insertion_lateral_alignment_cartesian_position_step <= 0.0
        or intermediate_insertion_lateral_alignment_cartesian_position_step
        > intermediate_insertion_lateral_position_tolerance
        or not math.isfinite(insertion_cartesian_position_step)
        or insertion_cartesian_position_step <= 0.0
        or insertion_cartesian_position_step > 0.002
        or not math.isfinite(insertion_lateral_alignment_cartesian_position_step)
        or insertion_lateral_alignment_cartesian_position_step <= 0.0
        or insertion_lateral_alignment_cartesian_position_step > insertion_lateral_position_tolerance
        or not math.isfinite(insertion_lateral_alignment_entry_clearance)
        or insertion_lateral_alignment_entry_clearance < 0.0
        or insertion_lateral_alignment_entry_clearance > 0.05
        or not math.isfinite(insertion_lateral_alignment_clearance_object_extent_scale)
        or insertion_lateral_alignment_clearance_object_extent_scale < 0.0
        or insertion_lateral_alignment_clearance_object_extent_scale > 2.0
        or not math.isfinite(insertion_axial_recovery_cartesian_position_step)
        or insertion_axial_recovery_cartesian_position_step <= 0.0
        or insertion_axial_recovery_cartesian_position_step > 0.002
        or not math.isfinite(insertion_axial_recovery_deadband)
        or insertion_axial_recovery_deadband < 0.0
        or insertion_axial_recovery_deadband >= insertion_axial_recovery_cartesian_position_step
        or not math.isfinite(insertion_compliance_capture_max_linear_speed)
        or insertion_compliance_capture_max_linear_speed <= 0.0
        or not math.isfinite(insertion_compliance_capture_max_angular_speed)
        or insertion_compliance_capture_max_angular_speed <= 0.0
        or insertion_compliance_capture_stable_steps <= 0
        or insertion_compliance_geometric_capture_after_steps < 0
        or insertion_compliance_geometric_capture_after_steps >= insertion_timeout_steps
        or not math.isfinite(insertion_compliance_minimum_gravity_alignment)
        or insertion_compliance_minimum_gravity_alignment < 0.0
        or insertion_compliance_minimum_gravity_alignment > 1.0
        or not math.isfinite(insertion_compliant_alignment_retraction_limit)
        or insertion_compliant_alignment_retraction_limit <= 0.0
        or insertion_compliant_alignment_retraction_limit > 0.05
        or not math.isfinite(move_above_orientation_first_tolerance)
        or move_above_orientation_first_tolerance < 0.0
    ):
        raise ValueError(
            'Move-above and preshape timeouts must be positive, prealign steps cannot be negative, '
            'preshape_open_margin must be in [0, 1], descend tolerances must be positive and ordered, '
            'preshape gripper tolerance must be in (0, 0.1], transport/insertion/base-place '
            'timeouts must '
            'be positive, and delayed descend/insertion tolerances must be positive and ordered, '
            'with insertion lateral tolerance no larger than the relaxed position tolerance; '
            'the insertion lateral-tolerance object-extent scale must be in [0, 0.25]; '
            'intermediate insertion lateral tolerance must be between the final tolerance '
            'and 0.005 m; '
            'the intermediate insertion lateral-alignment Cartesian step must be positive '
            'and no larger than the intermediate lateral tolerance; '
            'base support release tolerance must be in [0.008, 0.015]; '
            'base support lateral tolerance must be between the release tolerance '
            'and 0.02 m; '
            'base support lateral-alignment enter/exit tolerances must be positive, '
            'ordered, and no larger than the release/completion tolerances; '
            'the base support lateral-alignment Cartesian step must be positive and '
            'no larger than the alignment enter tolerance; '
            'the insertion Cartesian step must be in (0, 0.002]; '
            'the insertion lateral-alignment Cartesian step must be positive and no '
            'larger than the final insertion lateral tolerance; '
            'the insertion lateral-alignment entry clearance must be in [0, 0.05] m; '
            'the insertion lateral-alignment object-extent scale must be in [0, 2]; '
            'the insertion axial-recovery Cartesian step must be in (0, 0.002] and '
            'larger than its non-negative deadband; '
            'insertion-compliance capture speed limits and stable steps must be positive, '
            'and geometric capture delay must be non-negative and smaller than the '
            'insertion timeout; '
            'the insertion-compliance minimum gravity alignment must be in [0, 1]; '
            'the compliant alignment-retraction limit must be in (0, 0.05] m; '
            'orientation-first steps and tolerance '
            'must be non-negative.'
        )
    transport_tcp_height = _transport_tcp_height(
        spec,
        pickup_origin=pickup_origin,
        assembly_origin=assembly_origin,
    )

    def add_transport_clearance_targets(
        *,
        part_id: str,
        grasp: dict[str, Any],
        destination_position: list[float],
        destination_orientation: list[float],
    ) -> tuple[str, str]:
        source_position, source_orientation = _part_pickup_pose(
            parts_by_id[part_id],
            pickup_origin=pickup_origin,
            pickup_orientation=pickup_orientation,
        )
        pickup_clearance_name = f'part_{part_id}_pickup_clearance'
        targets.append(
            _target(
                pickup_clearance_name,
                _object_position_at_tcp_height(
                    object_position=source_position,
                    object_orientation=source_orientation,
                    grasp=grasp,
                    tcp_height=transport_tcp_height,
                ),
                source_orientation,
            )
        )
        pickup_target_names.append(pickup_clearance_name)

        assembly_clearance_name = f'part_{part_id}_assembly_clearance'
        targets.append(
            _target(
                assembly_clearance_name,
                _object_position_at_tcp_height(
                    object_position=destination_position,
                    object_orientation=destination_orientation,
                    grasp=grasp,
                    tcp_height=transport_tcp_height,
                ),
                destination_orientation,
            )
        )
        assembly_target_names.append(assembly_clearance_name)
        return pickup_clearance_name, assembly_clearance_name

    final_targets: dict[str, str] = {}
    fixture_pickup_targets: dict[str, str] = {}
    for part_id, part in parts_by_id.items():
        target_name = f'part_{part_id}_fixture_pickup'
        pickup_position, pickup_part_orientation = _part_pickup_pose(
            part,
            pickup_origin=pickup_origin,
            pickup_orientation=pickup_orientation,
        )
        fixture_pickup_targets[part_id] = target_name
        targets.append(
            _target(
                target_name,
                pickup_position,
                pickup_part_orientation,
            )
        )
        pickup_target_names.append(target_name)

    for part_id in parts_by_id:
        target_name = f'part_{part_id}_assembled'
        final_targets[part_id] = target_name
        targets.append(_target(target_name, assembly_origin, assembly_orientation))
        assembly_target_names.append(target_name)

    base_part_id = str(task['base_part'])
    base_object = f'fabrica_{assembly}_{base_part_id}'
    base_hover_name = f'part_{base_part_id}_assembly_hover'
    targets.append(
        _target(
            base_hover_name,
            _add(assembly_origin, [0.0, 0.0, transport_hover_height]),
            assembly_orientation,
        )
    )
    assembly_target_names.append(base_hover_name)
    base_pickup_clearance, base_assembly_clearance = add_transport_clearance_targets(
        part_id=base_part_id,
        grasp=task['base_grasp'],
        destination_position=assembly_origin,
        destination_orientation=assembly_orientation,
    )
    _append_pick_and_attach_phases(
        phases,
        prefix=f'base_{base_part_id}',
        robot=base_robot,
        object_name=base_object,
        grasp=task['base_grasp'],
        part=parts_by_id[base_part_id],
        approach_height=approach_height,
        move_above_timeout_steps=move_above_timeout_steps,
        move_above_orientation_first=move_above_orientation_first,
        move_above_orientation_first_steps=move_above_orientation_first_steps,
        move_above_orientation_first_tolerance=move_above_orientation_first_tolerance,
        preshape_timeout_steps=preshape_timeout_steps,
        preshape_open_margin=preshape_open_margin,
        preshape_gripper_position_tolerance=preshape_gripper_position_tolerance,
        descend_position_tolerance=descend_position_tolerance,
        descend_relaxed_position_tolerance=descend_relaxed_position_tolerance,
        descend_relaxed_after_steps=descend_relaxed_after_steps,
        prealign_steps=move_above_prealign_steps,
        prealign_joint_positions=prealign_joint_positions_by_robot.get(base_robot),
        prealign_shoulder_pan=prealign_shoulder_pan_by_robot.get(base_robot),
    )
    _append_transport_phase(
        phases,
        phase_name=f'base_{base_part_id}_pickup_clearance',
        robot=base_robot,
        object_name=base_object,
        target_name=base_pickup_clearance,
        insertion=False,
        timeout_steps=transport_timeout_steps,
    )
    _append_transport_phase(
        phases,
        phase_name=f'base_{base_part_id}_assembly_clearance',
        robot=base_robot,
        object_name=base_object,
        target_name=base_assembly_clearance,
        insertion=False,
        timeout_steps=transport_timeout_steps,
    )
    _append_transport_phase(
        phases,
        phase_name=f'base_{base_part_id}_transport_hover',
        robot=base_robot,
        object_name=base_object,
        target_name=base_hover_name,
        insertion=False,
        timeout_steps=transport_timeout_steps,
        strict_alignment=True,
    )
    _append_transport_phase(
        phases,
        phase_name=f'base_{base_part_id}_place',
        robot=base_robot,
        object_name=base_object,
        target_name=final_targets[base_part_id],
        insertion=True,
        timeout_steps=base_place_timeout_steps,
        insertion_relaxed_position_tolerance=min(
            base_support_release_position_tolerance,
            0.04,
        ),
        insertion_relaxed_after_steps=insertion_relaxed_after_steps,
        insertion_axis=_quat_rotate(
            assembly_orientation,
            [0.0, 0.0, -1.0],
        ),
        insertion_lateral_position_tolerance=(base_support_lateral_position_tolerance),
        insertion_cartesian_position_step=insertion_cartesian_position_step,
        insertion_lateral_alignment_cartesian_position_step=(base_support_lateral_alignment_cartesian_position_step),
        insertion_lateral_alignment_enter_tolerance=(base_support_lateral_alignment_enter_tolerance),
        insertion_lateral_alignment_exit_tolerance=(base_support_lateral_alignment_exit_tolerance),
        insertion_axial_recovery_cartesian_position_step=(insertion_axial_recovery_cartesian_position_step),
        insertion_axial_recovery_deadband=insertion_axial_recovery_deadband,
    )
    assembled_objects = [base_object]
    _append_release_and_retreat(
        phases,
        prefix=f'base_{base_part_id}',
        robot=base_robot,
        object_name=base_object,
        final_target=final_targets[base_part_id],
        assembled_objects=assembled_objects,
        retreat_offset=release_retreat_offset(_quat_rotate(assembly_orientation, [0.0, 0.0, -1.0])),
        park_offset=post_release_park_offset(base_robot),
        park_workspace_center=post_release_park_workspace_center(base_robot),
        park_minimum_planar_radius=post_release_park_minimum_planar_radius,
    )

    for step_index, step in enumerate(task['assembly_steps']):
        part_id = str(step['move_part'])
        object_name = f'fabrica_{assembly}_{part_id}'
        prefix = f'assemble_{step_index:02d}_part_{part_id}'
        path = list(reversed(step['disassembly_path']))
        if not path:
            raise ValueError(f'{assembly}: part {part_id} has an empty insertion path.')

        waypoint_names = []
        waypoint_positions: list[list[float]] = []
        waypoint_orientations: list[list[float]] = []
        for waypoint_index, waypoint in enumerate(path):
            position, orientation = _compose_pose(
                assembly_origin,
                assembly_orientation,
                _vector(waypoint['position'], size=3, name='insertion waypoint position'),
                _vector(waypoint['orientation'], size=4, name='insertion waypoint orientation'),
            )
            is_final_waypoint = waypoint_index == len(path) - 1
            if is_final_waypoint:
                waypoint_name = final_targets[part_id]
            else:
                waypoint_name = f'part_{part_id}_insert_{waypoint_index:02d}'
                targets.append(_target(waypoint_name, position, orientation))
                assembly_target_names.append(waypoint_name)
            waypoint_names.append(waypoint_name)
            waypoint_positions.append(position)
            waypoint_orientations.append(orientation)

        first_waypoint = next(target for target in targets if target['name'] == waypoint_names[0])
        hover_name = f'part_{part_id}_preinsert_hover'
        targets.append(
            _target(
                hover_name,
                _add(first_waypoint['position'], [0.0, 0.0, transport_hover_height]),
                first_waypoint['orientation'],
            )
        )
        assembly_target_names.append(hover_name)
        pickup_clearance, assembly_clearance = add_transport_clearance_targets(
            part_id=part_id,
            grasp=step['move_grasp'],
            destination_position=first_waypoint['position'],
            destination_orientation=first_waypoint['orientation'],
        )

        _append_pick_and_attach_phases(
            phases,
            prefix=prefix,
            robot=assembly_robot,
            object_name=object_name,
            grasp=step['move_grasp'],
            part=parts_by_id[part_id],
            approach_height=approach_height,
            move_above_timeout_steps=move_above_timeout_steps,
            move_above_orientation_first=move_above_orientation_first,
            move_above_orientation_first_steps=move_above_orientation_first_steps,
            move_above_orientation_first_tolerance=move_above_orientation_first_tolerance,
            preshape_timeout_steps=preshape_timeout_steps,
            preshape_open_margin=preshape_open_margin,
            preshape_gripper_position_tolerance=preshape_gripper_position_tolerance,
            descend_position_tolerance=descend_position_tolerance,
            descend_relaxed_position_tolerance=descend_relaxed_position_tolerance,
            descend_relaxed_after_steps=descend_relaxed_after_steps,
            prealign_steps=move_above_prealign_steps,
            prealign_joint_positions=prealign_joint_positions_by_robot.get(assembly_robot),
            prealign_shoulder_pan=prealign_shoulder_pan_by_robot.get(assembly_robot),
        )
        _append_transport_phase(
            phases,
            phase_name=f'{prefix}_pickup_clearance',
            robot=assembly_robot,
            object_name=object_name,
            target_name=pickup_clearance,
            insertion=False,
            timeout_steps=transport_timeout_steps,
            insertion_relaxed_position_tolerance=insertion_relaxed_position_tolerance,
            insertion_relaxed_after_steps=insertion_relaxed_after_steps,
        )
        _append_transport_phase(
            phases,
            phase_name=f'{prefix}_assembly_clearance',
            robot=assembly_robot,
            object_name=object_name,
            target_name=assembly_clearance,
            insertion=False,
            timeout_steps=transport_timeout_steps,
            insertion_relaxed_position_tolerance=insertion_relaxed_position_tolerance,
            insertion_relaxed_after_steps=insertion_relaxed_after_steps,
        )
        _append_transport_phase(
            phases,
            phase_name=f'{prefix}_transport_hover',
            robot=assembly_robot,
            object_name=object_name,
            target_name=hover_name,
            insertion=False,
            timeout_steps=transport_timeout_steps,
            strict_alignment=True,
            insertion_relaxed_position_tolerance=insertion_relaxed_position_tolerance,
            insertion_relaxed_after_steps=insertion_relaxed_after_steps,
        )
        for waypoint_index, waypoint_name in enumerate(waypoint_names):
            is_final_waypoint = waypoint_name == final_targets[part_id]
            is_entry_waypoint = waypoint_index == 0
            is_pre_final_waypoint = len(waypoint_names) > 1 and waypoint_index == len(waypoint_names) - 2
            if len(waypoint_positions) > 1:
                previous_index = max(waypoint_index - 1, 0)
                next_index = min(waypoint_index + 1, len(waypoint_positions) - 1)
                segment_start = waypoint_positions[previous_index]
                segment_end = waypoint_positions[next_index]
                insertion_axis = [segment_end[axis] - segment_start[axis] for axis in range(3)]
            else:
                insertion_axis = _quat_rotate(
                    assembly_orientation,
                    _vector(
                        step['move_grasp']['assembly_approach_direction'],
                        size=3,
                        name='assembly approach direction',
                    ),
                )
            normalized_insertion_axis = _normalized_direction(
                insertion_axis,
                name=f'{prefix}_insert_{waypoint_index:02d} insertion axis',
            )
            insertion_entry_delta = [
                waypoint_positions[waypoint_index][axis] - waypoint_positions[0][axis] for axis in range(3)
            ]
            part_bbox_size = _vector(
                parts_by_id[part_id]['bbox_size'],
                size=3,
                name=f'{assembly} part {part_id} bbox_size',
            )
            part_world_axes = [
                _quat_rotate(
                    waypoint_orientations[waypoint_index],
                    [1.0 if local_axis == axis else 0.0 for local_axis in range(3)],
                )
                for axis in range(3)
            ]
            projected_object_extent = sum(
                abs(
                    sum(
                        part_world_axes[axis][world_axis] * normalized_insertion_axis[world_axis]
                        for world_axis in range(3)
                    )
                )
                * part_bbox_size[axis]
                for axis in range(3)
            )
            minimum_lateral_extent = _projected_box_minimum_lateral_extent(
                bbox_size=part_bbox_size,
                orientation=waypoint_orientations[waypoint_index],
                axis=normalized_insertion_axis,
            )
            geometry_lateral_tolerance = min(
                intermediate_insertion_lateral_position_tolerance,
                max(
                    insertion_lateral_position_tolerance,
                    minimum_lateral_extent * insertion_lateral_tolerance_object_extent_scale,
                ),
            )
            gravity_axis_alignment = abs(normalized_insertion_axis[2])
            if gravity_axis_alignment < insertion_compliance_minimum_gravity_alignment:
                geometry_lateral_tolerance = intermediate_insertion_lateral_position_tolerance
            compliance_waypoint_axial_tolerance = min(
                insertion_relaxed_position_tolerance,
                0.010,
            )
            compliance_waypoint_lateral_tolerance = intermediate_insertion_lateral_position_tolerance
            insertion_path_depth = max(
                sum(insertion_entry_delta[axis] * normalized_insertion_axis[axis] for axis in range(3)),
                0.0,
            )
            lateral_alignment_axial_clearance = insertion_path_depth + max(
                insertion_lateral_alignment_entry_clearance,
                projected_object_extent * insertion_lateral_alignment_clearance_object_extent_scale,
            )
            compliance_position_tolerance = max(
                0.015,
                min(0.020, 0.6 * min(part_bbox_size)),
            )
            if not is_final_waypoint:
                compliance_position_tolerance += math.dist(
                    waypoint_positions[waypoint_index],
                    waypoint_positions[-1],
                )
            _append_transport_phase(
                phases,
                phase_name=f'{prefix}_insert_{waypoint_index:02d}',
                robot=assembly_robot,
                object_name=object_name,
                target_name=waypoint_name,
                insertion=True,
                timeout_steps=insertion_timeout_steps,
                insertion_relaxed_position_tolerance=(
                    min(insertion_relaxed_position_tolerance, 0.015)
                    if is_final_waypoint
                    else min(insertion_relaxed_position_tolerance, 0.010)
                ),
                insertion_relaxed_after_steps=insertion_relaxed_after_steps,
                insertion_axis=insertion_axis,
                insertion_lateral_position_tolerance=(
                    geometry_lateral_tolerance
                    if is_entry_waypoint or is_pre_final_waypoint or is_final_waypoint
                    else intermediate_insertion_lateral_position_tolerance
                ),
                insertion_cartesian_position_step=(insertion_cartesian_position_step),
                insertion_lateral_alignment_cartesian_position_step=(
                    insertion_lateral_alignment_cartesian_position_step
                    if is_final_waypoint
                    else intermediate_insertion_lateral_alignment_cartesian_position_step
                ),
                insertion_lateral_alignment_exit_tolerance=(
                    intermediate_insertion_lateral_position_tolerance if is_pre_final_waypoint else None
                ),
                insertion_lateral_alignment_axial_clearance=(lateral_alignment_axial_clearance),
                insertion_path_depth=(
                    insertion_path_depth
                    if gravity_axis_alignment < insertion_compliance_minimum_gravity_alignment
                    else 0.0
                ),
                insertion_axial_recovery_cartesian_position_step=(insertion_axial_recovery_cartesian_position_step),
                insertion_axial_recovery_deadband=insertion_axial_recovery_deadband,
                insertion_compliance_capture_max_linear_speed=(insertion_compliance_capture_max_linear_speed),
                insertion_compliance_capture_max_angular_speed=(insertion_compliance_capture_max_angular_speed),
                insertion_compliance_capture_stable_steps=(insertion_compliance_capture_stable_steps),
                insertion_compliance_position_tolerance=(compliance_position_tolerance),
                insertion_compliance_after_steps=(0),
                insertion_compliance_require_waypoint_proximity=True,
                insertion_compliance_waypoint_position_tolerance=(min(insertion_relaxed_position_tolerance, 0.010)),
                insertion_compliance_waypoint_axial_position_tolerance=(compliance_waypoint_axial_tolerance),
                insertion_compliance_waypoint_lateral_position_tolerance=(compliance_waypoint_lateral_tolerance),
                insertion_compliance_geometric_capture_after_steps=(insertion_compliance_geometric_capture_after_steps),
                insertion_compliance_minimum_gravity_alignment=(insertion_compliance_minimum_gravity_alignment),
                insertion_compliant_alignment_retraction_limit=(insertion_compliant_alignment_retraction_limit),
                insertion_compliant_track_object_orientation=(insertion_compliant_track_object_orientation),
                target_object_entry_capture_max_steps=(
                    max(4, insertion_compliance_capture_stable_steps + 4) if is_final_waypoint else 0
                ),
                final_target_name=(final_targets[part_id]),
                settle_at_target=is_final_waypoint,
            )
        assembled_objects.append(object_name)
        _append_release_and_retreat(
            phases,
            prefix=prefix,
            robot=assembly_robot,
            object_name=object_name,
            final_target=final_targets[part_id],
            assembled_objects=assembled_objects,
            retreat_offset=release_retreat_offset(insertion_axis),
            park_offset=post_release_park_offset(assembly_robot),
            park_workspace_center=post_release_park_workspace_center(assembly_robot),
            park_minimum_planar_radius=post_release_park_minimum_planar_radius,
        )

    base_release_phase = next(phase for phase in phases if phase.get('name') == f'base_{base_part_id}_release_and_lock')
    base_release_phase['lock'][0]['rebase_targets'] = list(dict.fromkeys(assembly_target_names))

    stabilize_fixture_parts = bool(spec.get('stabilize_fixture_parts', True))
    if stabilize_fixture_parts:
        phases[0]['lock'] = [
            {
                'object': f'fabrica_{assembly}_{part_id}',
                'target': fixture_pickup_targets[part_id],
                'position_tolerance': 0.03,
                'orientation_tolerance': 0.20,
                'snap_free_object': True,
                'free_snap_steps': 0,
            }
            for part_id in parts_by_id
            if part_id != base_part_id
        ]

    for part_id in parts_by_id:
        success.append(
            {
                'object': f'fabrica_{assembly}_{part_id}',
                'target': final_targets[part_id],
                'position_tolerance': 0.015,
                'orientation_tolerance': 0.15,
                'require_released': True,
                'require_static': True,
                'linear_velocity_threshold': 0.05,
                'angular_velocity_threshold': 2.0,
            }
        )
    return targets, phases, success, pickup_target_names, assembly_target_names


def _default_domain_randomization(
    *,
    spec: dict[str, Any],
    assembly: str,
    part_names: list[str],
    pickup_target_names: list[str],
    assembly_target_names: list[str],
    translation_constraints: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pickup_range = spec.get(
        'pickup_translation_range',
        {'x': [-0.03, 0.03], 'y': [-0.03, 0.03], 'z': [0.0, 0.0]},
    )
    assembly_range = spec.get(
        'assembly_translation_range',
        {'x': [-0.03, 0.03], 'y': [-0.03, 0.03], 'z': [0.0, 0.0]},
    )
    pickup_minimum_planar_distance = float(spec.get('pickup_translation_minimum_planar_distance', 0.0))
    pickup_maximum_planar_distance = float(spec.get('pickup_translation_maximum_planar_distance', math.inf))
    assembly_minimum_planar_distance = float(spec.get('assembly_translation_minimum_planar_distance', 0.0))
    assembly_maximum_planar_distance = float(spec.get('assembly_translation_maximum_planar_distance', math.inf))
    return {
        'enabled': False,
        'seed_namespace': f'fabrica_{assembly}_canonical_ur5e_domain_v1',
        'fixed_objects': ['optical_board'],
        'groups': {
            'start_parts': {
                'translation': copy.deepcopy(pickup_range),
                **(
                    {'minimum_planar_distance': pickup_minimum_planar_distance}
                    if pickup_minimum_planar_distance > 0.0
                    else {}
                ),
                **(
                    {'maximum_planar_distance': pickup_maximum_planar_distance}
                    if math.isfinite(pickup_maximum_planar_distance)
                    else {}
                ),
                'translation_constraints': copy.deepcopy(translation_constraints.get('start_parts', [])),
                'objects': ['fabrica_fixture', *part_names],
                'targets': copy.deepcopy(pickup_target_names),
            },
            'assembly_base': {
                'translation': copy.deepcopy(assembly_range),
                **(
                    {'minimum_planar_distance': assembly_minimum_planar_distance}
                    if assembly_minimum_planar_distance > 0.0
                    else {}
                ),
                **(
                    {'maximum_planar_distance': assembly_maximum_planar_distance}
                    if math.isfinite(assembly_maximum_planar_distance)
                    else {}
                ),
                'translation_constraints': copy.deepcopy(translation_constraints.get('assembly_base', [])),
                'objects': [],
                'targets': copy.deepcopy(assembly_target_names),
            },
        },
        'appearance': {
            'groups': {
                'table_surface': {
                    'objects': ['factory_tabletop_visual'],
                    'palette': [
                        [0.14, 0.14, 0.14],
                        [0.28, 0.31, 0.32],
                        [0.22, 0.30, 0.25],
                        [0.33, 0.24, 0.25],
                        [0.25, 0.27, 0.36],
                        [0.40, 0.36, 0.25],
                    ],
                },
                'background': {
                    'objects': [],
                    'lights': ['warehouse_dome_fill'],
                    'palette': [
                        [0.78, 0.84, 1.0],
                        [0.92, 0.82, 0.72],
                        [0.72, 0.88, 0.80],
                        [0.86, 0.76, 0.82],
                        [0.72, 0.82, 0.88],
                        [0.88, 0.88, 0.76],
                    ],
                },
            }
        },
    }


def _bbox_corners(
    *,
    position: list[float],
    orientation: list[float],
    bbox_min: list[float],
    bbox_max: list[float],
) -> list[list[float]]:
    return [
        _add(position, _quat_rotate(orientation, [x, y, z]))
        for x in (bbox_min[0], bbox_max[0])
        for y in (bbox_min[1], bbox_max[1])
        for z in (bbox_min[2], bbox_max[2])
    ]


def _tcp_position_for_object_pose(
    *,
    object_position: list[float],
    object_orientation: list[float],
    grasp: dict[str, Any],
) -> list[float]:
    tcp_position, _ = _tcp_pose_for_object_pose(
        object_position=object_position,
        object_orientation=object_orientation,
        grasp=grasp,
    )
    return tcp_position


def _canonical_translation_constraints(
    *,
    assembly: str,
    task: dict[str, Any],
    spec: dict[str, Any],
    generated_objects: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    pickup_target_names: list[str],
    assembly_target_names: list[str],
    workspace_offset: list[float],
    robots: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    objects_by_name = {str(entry['name']): entry for entry in generated_objects}
    targets_by_name = {str(entry['name']): entry for entry in targets}
    board = objects_by_name['optical_board']
    fixture = objects_by_name['fabrica_fixture']
    board_lower = _add(
        board['position'],
        _vector(task['optical_board']['bbox_min'], size=3, name='board bbox_min'),
    )
    board_upper = _add(
        board['position'],
        _vector(task['optical_board']['bbox_max'], size=3, name='board bbox_max'),
    )
    constraints: dict[str, list[dict[str, Any]]] = {
        'start_parts': [
            {
                'type': 'points_inside_bounds',
                'points': _bbox_corners(
                    position=fixture['position'],
                    orientation=fixture['orientation'],
                    bbox_min=_vector(
                        task['fixture']['bbox_min'],
                        size=3,
                        name='fixture bbox_min',
                    ),
                    bbox_max=_vector(
                        task['fixture']['bbox_max'],
                        size=3,
                        name='fixture bbox_max',
                    ),
                ),
                'lower': board_lower,
                'upper': board_upper,
                'axes': [0, 1],
            }
        ],
        'assembly_base': [],
    }

    final_assembly_corners = []
    for part in task['parts']:
        part_id = str(part['part_id'])
        final_target = targets_by_name[f'part_{part_id}_assembled']
        final_assembly_corners.extend(
            _bbox_corners(
                position=final_target['position'],
                orientation=final_target['orientation'],
                bbox_min=_vector(part['bbox_min'], size=3, name=f'{assembly} part {part_id} bbox_min'),
                bbox_max=_vector(part['bbox_max'], size=3, name=f'{assembly} part {part_id} bbox_max'),
            )
        )
    constraints['assembly_base'].append(
        {
            'type': 'points_inside_bounds',
            'points': final_assembly_corners,
            'lower': board_lower,
            'upper': board_upper,
            'axes': [0, 1],
        }
    )

    base_part = str(task['base_part'])
    grasp_by_part = {
        base_part: task['base_grasp'],
        **{str(step['move_part']): step['move_grasp'] for step in task['assembly_steps']},
    }
    robot_name_by_part = {
        part_id: str(
            spec.get('base_robot', 'franka_right')
            if part_id == base_part
            else spec.get('assembly_robot', 'franka_left')
        )
        for part_id in grasp_by_part
    }
    robots_by_name = {
        str(robot['name']): robot for robot in robots if isinstance(robot, dict) and robot.get('name') is not None
    }
    maximum_reach = float(spec.get('pickup_tcp_maximum_reach', 0.82))

    for group_name, target_names in (
        ('start_parts', pickup_target_names),
        ('assembly_base', assembly_target_names),
    ):
        tcp_points_by_robot: dict[str, list[list[float]]] = {}
        for target_name in target_names:
            tokens = str(target_name).split('_', 2)
            if len(tokens) < 3 or tokens[0] != 'part' or tokens[1] not in grasp_by_part:
                continue
            part_id = tokens[1]
            target = targets_by_name[str(target_name)]
            tcp_position = _tcp_position_for_object_pose(
                object_position=target['position'],
                object_orientation=target['orientation'],
                grasp=grasp_by_part[part_id],
            )
            tcp_points_by_robot.setdefault(robot_name_by_part[part_id], []).append(_add(tcp_position, workspace_offset))

        if group_name == 'start_parts':
            for part_id, grasp in grasp_by_part.items():
                part_object = objects_by_name[f'fabrica_{assembly}_{part_id}']
                tcp_position = _tcp_position_for_object_pose(
                    object_position=part_object['position'],
                    object_orientation=part_object['orientation'],
                    grasp=grasp,
                )
                tcp_points_by_robot.setdefault(robot_name_by_part[part_id], []).append(
                    _add(_add(tcp_position, [0.0, 0.0, 0.10]), workspace_offset)
                )

        for robot_name, points in tcp_points_by_robot.items():
            robot = robots_by_name.get(robot_name)
            if robot is None:
                raise ValueError(f'Domain randomization references missing robot {robot_name!r}.')
            robot_position = _vector(
                robot.get('position', [0.0, 0.0, 0.0]),
                size=3,
                name=f'{robot_name} position',
            )
            if bool(robot.get('apply_workspace_offset', True)):
                robot_position = _add(robot_position, workspace_offset)
            constraints[group_name].append(
                {
                    'type': 'points_within_distance',
                    'points': points,
                    'origin': robot_position,
                    'maximum_distance': maximum_reach,
                }
            )
    return constraints


def compile_fabrica_canonical_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload.get('fabrica_canonical')
    if spec is None:
        return payload
    if not isinstance(spec, dict):
        raise ValueError('fabrica_canonical must be a mapping.')

    assembly = str(spec.get('assembly', '')).strip()
    if not assembly:
        raise ValueError('fabrica_canonical.assembly is required.')
    metadata_path = spec.get('metadata_path')
    metadata = load_fabrica_canonical_metadata(metadata_path)
    task = metadata['tasks'].get(assembly)
    if task is None:
        raise ValueError(
            f'Unknown canonical Fabrica assembly {assembly!r}; ' f'available tasks: {sorted(metadata["tasks"])}.'
        )

    configured_pickup_origin = _vector(
        spec.get('pickup_origin', [0.47, -0.45, 0.0125]),
        size=3,
        name='pickup_origin',
    )
    configured_pickup_orientation = _vector(
        spec.get('pickup_orientation', [1.0, 0.0, 0.0, 0.0]),
        size=4,
        name='pickup_orientation',
    )
    assembly_origin = _vector(
        spec.get('assembly_origin', [0.70, -0.18, 0.0125]),
        size=3,
        name='assembly_origin',
    )
    assembly_orientation = _vector(
        spec.get('assembly_orientation', [1.0, 0.0, 0.0, 0.0]),
        size=4,
        name='assembly_orientation',
    )
    board_origin = _vector(
        spec.get('board_origin', [0.47, -0.14, 0.0125]),
        size=3,
        name='board_origin',
    )
    workspace_offset = _vector(
        payload.get('workspace_offset', [0.0, 0.0, 0.0]),
        size=3,
        name='workspace_offset',
    )
    robots = payload.get('robots', [])
    if not isinstance(robots, list):
        raise ValueError('robots must be a list.')
    task = copy.deepcopy(task)
    (
        pickup_origin,
        pickup_orientation,
        selected_base_grasp,
        selected_move_grasps,
        grasp_selection,
    ) = _resolve_pickup_layout_and_grasps(
        assembly=assembly,
        task=task,
        spec=spec,
        configured_pickup_origin=configured_pickup_origin,
        configured_pickup_orientation=configured_pickup_orientation,
        board_origin=board_origin,
        assembly_origin=assembly_origin,
        assembly_orientation=assembly_orientation,
        workspace_offset=workspace_offset,
        robots=robots,
    )
    task['base_grasp'] = selected_base_grasp
    for step in task['assembly_steps']:
        part_id = str(step['move_part'])
        step['move_grasp'] = copy.deepcopy(selected_move_grasps[part_id])

    generated_objects = [
        _static_asset_object(
            name='optical_board',
            usd_path=task['optical_board']['usd_path'],
            position=board_origin,
            friction=0.5,
        ),
        _static_asset_object(
            name='fabrica_fixture',
            usd_path=task['fixture']['usd_path'],
            position=pickup_origin,
            orientation=pickup_orientation,
            friction=0.8,
        ),
    ]
    generated_objects.extend(
        _part_object(
            assembly,
            part,
            pickup_origin=pickup_origin,
            pickup_orientation=pickup_orientation,
        )
        for part in task['parts']
    )

    (targets, phases, success, pickup_target_names, assembly_target_names,) = _compile_targets_and_phases(
        assembly=assembly,
        task=task,
        spec=spec,
        pickup_origin=pickup_origin,
        pickup_orientation=pickup_orientation,
        assembly_origin=assembly_origin,
        assembly_orientation=assembly_orientation,
        robots=robots,
    )
    part_names = [f'fabrica_{assembly}_{part["part_id"]}' for part in task['parts']]
    translation_constraints = _canonical_translation_constraints(
        assembly=assembly,
        task=task,
        spec=spec,
        generated_objects=generated_objects,
        targets=targets,
        pickup_target_names=pickup_target_names,
        assembly_target_names=assembly_target_names,
        workspace_offset=workspace_offset,
        robots=robots,
    )
    generated_randomization = _default_domain_randomization(
        spec=spec,
        assembly=assembly,
        part_names=part_names,
        pickup_target_names=pickup_target_names,
        assembly_target_names=assembly_target_names,
        translation_constraints=translation_constraints,
    )

    resolved = copy.deepcopy(payload)
    resolved['objects'] = _merge_named_entries(generated_objects, resolved.get('objects', []))
    resolved['targets'] = _merge_named_entries(targets, resolved.get('targets', []))
    resolved['phases'] = phases
    resolved['success'] = success
    resolved['domain_randomization'] = deep_merge(
        generated_randomization,
        resolved.get('domain_randomization', {}),
    )
    fixed_objects = {str(name) for name in resolved['domain_randomization'].get('fixed_objects', [])}
    fixed_objects.add('optical_board')
    resolved['domain_randomization']['fixed_objects'] = sorted(fixed_objects)
    for group_name, group in resolved['domain_randomization'].get('groups', {}).items():
        if 'optical_board' in (group.get('objects') or []):
            raise ValueError(f'Optical board is fixed and cannot belong to position group {group_name!r}.')

    resolved['max_steps'] = max(
        int(resolved.get('max_steps', 0)),
        22000,
        len(task['parts']) * 7000,
        len(phases) * 360,
    )
    resolved['fabrica_canonical_resolved'] = {
        'assembly': assembly,
        'metadata_schema_version': metadata['schema_version'],
        'metadata_path': str(Path(metadata_path or CANONICAL_METADATA_PATH).expanduser().resolve()),
        'bundle_path': _asset_path(task['bundle_path']),
        'base_part': str(task['base_part']),
        'transport_tcp_height': _transport_tcp_height(
            spec,
            pickup_origin=pickup_origin,
            assembly_origin=assembly_origin,
        ),
        'transport_timeout_steps': int(spec.get('transport_timeout_steps', 4800)),
        'insertion_timeout_steps': int(spec.get('insertion_timeout_steps', 3600)),
        'base_place_timeout_steps': int(spec.get('base_place_timeout_steps', 4800)),
        'insertion_lateral_position_tolerance': float(spec.get('insertion_lateral_position_tolerance', 0.001)),
        'insertion_lateral_tolerance_object_extent_scale': float(
            spec.get('insertion_lateral_tolerance_object_extent_scale', 0.04)
        ),
        'intermediate_insertion_lateral_position_tolerance': float(
            spec.get('intermediate_insertion_lateral_position_tolerance', 0.002)
        ),
        'intermediate_insertion_lateral_alignment_cartesian_position_step': float(
            spec.get(
                'intermediate_insertion_lateral_alignment_cartesian_position_step',
                0.001,
            )
        ),
        'insertion_cartesian_position_step': float(spec.get('insertion_cartesian_position_step', 0.00025)),
        'insertion_lateral_alignment_cartesian_position_step': float(
            spec.get('insertion_lateral_alignment_cartesian_position_step', 0.00025)
        ),
        'insertion_lateral_alignment_entry_clearance': float(
            spec.get('insertion_lateral_alignment_entry_clearance', 0.01)
        ),
        'insertion_lateral_alignment_clearance_object_extent_scale': float(
            spec.get(
                'insertion_lateral_alignment_clearance_object_extent_scale',
                1.0,
            )
        ),
        'insertion_axial_recovery_cartesian_position_step': float(
            spec.get('insertion_axial_recovery_cartesian_position_step', 0.001)
        ),
        'insertion_axial_recovery_deadband': float(spec.get('insertion_axial_recovery_deadband', 0.0005)),
        'insertion_compliance_capture_max_linear_speed': float(
            spec.get('insertion_compliance_capture_max_linear_speed', 0.10)
        ),
        'insertion_compliance_capture_max_angular_speed': float(
            spec.get('insertion_compliance_capture_max_angular_speed', 2.0)
        ),
        'insertion_compliance_capture_stable_steps': int(spec.get('insertion_compliance_capture_stable_steps', 8)),
        'insertion_compliance_geometric_capture_after_steps': int(
            spec.get('insertion_compliance_geometric_capture_after_steps', 1200)
        ),
        'insertion_compliance_minimum_gravity_alignment': float(
            spec.get('insertion_compliance_minimum_gravity_alignment', 0.70)
        ),
        'insertion_compliant_alignment_retraction_limit': float(
            spec.get('insertion_compliant_alignment_retraction_limit', 0.006)
        ),
        'insertion_compliant_track_object_orientation': bool(
            spec.get('insertion_compliant_track_object_orientation', False)
        ),
        'base_support_release_position_tolerance': float(spec.get('base_support_release_position_tolerance', 0.012)),
        'base_support_lateral_position_tolerance': float(spec.get('base_support_lateral_position_tolerance', 0.015)),
        'base_support_lateral_alignment_enter_tolerance': float(
            spec.get('base_support_lateral_alignment_enter_tolerance', 0.002)
        ),
        'base_support_lateral_alignment_exit_tolerance': float(
            spec.get('base_support_lateral_alignment_exit_tolerance', 0.004)
        ),
        'base_support_lateral_alignment_cartesian_position_step': float(
            spec.get(
                'base_support_lateral_alignment_cartesian_position_step',
                0.002,
            )
        ),
        'release_retreat_distance': float(spec.get('release_retreat_distance', 0.06)),
        'post_release_park_distance': float(spec.get('post_release_park_distance', 0.35)),
        'post_release_park_vertical_offset': float(spec.get('post_release_park_vertical_offset', 0.02)),
        'post_release_park_minimum_planar_radius': float(spec.get('post_release_park_minimum_planar_radius', 0.28)),
        'assembly_order': [
            {
                'move_part': str(step['move_part']),
                'socket_part': str(step['socket_part']),
                'optimizer_hold_part': str(step['optimizer_hold_part']),
            }
            for step in task['assembly_steps']
        ],
        'stabilize_fixture_parts': bool(spec.get('stabilize_fixture_parts', True)),
        'configured_pickup_origin': configured_pickup_origin,
        'selected_pickup_origin': pickup_origin,
        'configured_pickup_orientation': configured_pickup_orientation,
        'selected_pickup_orientation': pickup_orientation,
        'selected_base_grasp': selected_base_grasp,
        'selected_move_grasps': selected_move_grasps,
        'move_grasp_selection': grasp_selection['move_grasp_selection'],
        'pickup_layout_selection': grasp_selection,
        'optical_board_position_randomized': False,
    }
    return resolved
