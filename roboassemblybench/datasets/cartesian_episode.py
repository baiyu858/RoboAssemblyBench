from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roboassemblybench.datasets.media_codecs import ChunkedDepthWriter, FFmpegVideoWriter

ROBOT_NAMES = ('franka_left', 'franka_right')
CAMERA_KEYS = (
    'observation.images.front',
    'observation.images.left_wrist',
    'observation.images.right_wrist',
)


def expected_replay_joint_widths(robot_metadata) -> list[int] | None:
    """Return a strict articulation signature for supported robot platforms."""

    entries = list(robot_metadata or [])
    if entries and all(str(entry.get('type')) in {'FrankaRobot', 'franka', 'Franka'} for entry in entries):
        return [9] * len(entries)
    return None


def _cartesian_names(prefix: str) -> list[str]:
    return [
        f'{prefix}_eef_x',
        f'{prefix}_eef_y',
        f'{prefix}_eef_z',
        f'{prefix}_eef_qw',
        f'{prefix}_eef_qx',
        f'{prefix}_eef_qy',
        f'{prefix}_eef_qz',
        f'{prefix}_gripper_open',
    ]


STATE_NAMES = tuple(_cartesian_names('left') + _cartesian_names('right'))
ACTION_NAMES = tuple(
    name.replace('_eef_', '_target_').replace('_gripper_open', '_gripper_target') for name in STATE_NAMES
)
ACTION_SEMANTICS = 'next_sample_dual_arm_absolute_cartesian_pose_and_gripper_target'


def task_progress_annotations(
    phase_annotations: dict[str, dict[str, Any]],
    subtask_indices,
    substage_indices,
) -> dict[str, list[dict[str, Any]]]:
    subtask_names = sorted({str(item.get('subtask_id', 'unknown')) for item in phase_annotations.values()})
    subtask_labels = [
        {'subtask_index': index, 'subtask_id': name}
        for index, name in enumerate(subtask_names)
    ]
    substage_labels = []
    for phase_name, annotation in sorted(
        phase_annotations.items(), key=lambda item: int(item[1].get('phase_index', -1))
    ):
        substage_labels.append(
            {
                'substage_index': int(annotation.get('phase_index', -1)),
                'substage_id': str(annotation.get('substage_id') or annotation.get('name') or phase_name),
                'phase': str(annotation.get('name') or phase_name),
            }
        )

    def boundaries(values, *, index_key: str, label_key: str, label_by_index: dict[int, str]):
        values = np.asarray(values).reshape(-1)
        if values.size == 0:
            return []
        result = []
        start = 0
        previous = int(values[0])
        for frame_index, current_value in enumerate(values[1:], start=1):
            current = int(current_value)
            if current != previous:
                result.append(
                    {
                        index_key: previous,
                        label_key: label_by_index.get(previous, 'unknown'),
                        'start_frame': start,
                        'end_frame': frame_index - 1,
                    }
                )
                start = frame_index
                previous = current
        result.append(
            {
                index_key: previous,
                label_key: label_by_index.get(previous, 'unknown'),
                'start_frame': start,
                'end_frame': int(values.size) - 1,
            }
        )
        return result

    subtask_label_by_index = {item['subtask_index']: item['subtask_id'] for item in subtask_labels}
    substage_label_by_index = {item['substage_index']: item['substage_id'] for item in substage_labels}
    return {
        'subtask_labels': subtask_labels,
        'substage_labels': substage_labels,
        'subtask_boundaries': boundaries(
            subtask_indices,
            index_key='subtask_index',
            label_key='subtask_id',
            label_by_index=subtask_label_by_index,
        ),
        'substage_boundaries': boundaries(
            substage_indices,
            index_key='substage_index',
            label_key='substage_id',
            label_by_index=substage_label_by_index,
        ),
    }


def cartesian_trajectory_errors(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    simulation_steps: np.ndarray | None = None,
    frame_stride: int | None = None,
) -> list[str]:
    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.ndim != 2 or states.shape[1:] != (len(STATE_NAMES),):
        return ['state_shape']
    if actions.shape != states.shape:
        return ['action_shape']
    errors = []
    if not np.all(np.isfinite(states)):
        errors.append('state_nonfinite')
    if not np.all(np.isfinite(actions)):
        errors.append('action_nonfinite')
    if errors:
        return errors

    for values, prefix in ((states, 'state'), (actions, 'action')):
        grippers = values[:, [7, 15]]
        if np.any(grippers < -1e-5) or np.any(grippers > 1.0 + 1e-5):
            errors.append(f'{prefix}_gripper_range')
        for quaternion_slice in (slice(3, 7), slice(11, 15)):
            norms = np.linalg.norm(values[:, quaternion_slice], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
                errors.append(f'{prefix}_quaternion_norm')
                break

    pose_indices = [*range(0, 7), *range(8, 15)]
    expected_action_poses = states[:, pose_indices].copy()
    if len(states) > 1:
        expected_action_poses[:-1] = states[1:, pose_indices]
    if not np.allclose(actions[:, pose_indices], expected_action_poses, atol=1e-5, rtol=0.0):
        errors.append('absolute_next_sample_pose')

    if simulation_steps is not None:
        simulation_steps = np.asarray(simulation_steps)
        if simulation_steps.shape != (len(states),):
            errors.append('simulation_step_shape')
        elif len(simulation_steps) > 1 and frame_stride is not None:
            if not np.all(np.diff(simulation_steps) == int(frame_stride)):
                errors.append('simulation_step_stride')
    return sorted(set(errors))


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_quaternion(quaternion, reference=None) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quaternion = quaternion / norm
    if reference is not None and float(np.dot(quaternion, reference)) < 0.0:
        quaternion = -quaternion
    elif reference is None and quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _gripper_joint_position(robot_obs: dict[str, Any]) -> float | None:
    controller_obs = (robot_obs.get('controllers') or {}).get('gripper_controller') or {}
    gripper_position = controller_obs.get('gripper_pos')
    if gripper_position is None:
        return None
    values = np.asarray(gripper_position, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return float(values[0])


def _gripper_openness(task, robot_name: str, robot_obs: dict[str, Any]) -> float:
    gripper_q = _gripper_joint_position(robot_obs)
    robot = getattr(task, 'robots', {}).get(robot_name)
    config = getattr(robot, 'config', None)
    open_q = float(getattr(config, 'gripper_open_position', 0.0))
    closed_q = float(getattr(config, 'gripper_closed_position', 0.8))
    if gripper_q is None or abs(open_q - closed_q) <= 1e-8:
        return 1.0
    return float(np.clip((gripper_q - closed_q) / (open_q - closed_q), 0.0, 1.0))


def _gripper_command(actions: dict[str, Any], robot_name: str, fallback: float) -> float:
    robot_action = actions.get(robot_name) or {}
    command = robot_action.get('gripper_controller')
    if isinstance(command, (list, tuple, np.ndarray)):
        command = command[0] if len(command) else None
    if command is None:
        return float(fallback)
    if isinstance(command, str):
        lowered = command.strip().lower()
        if lowered == 'open':
            return 1.0
        if lowered == 'close':
            return 0.0
    try:
        return float(np.clip(float(command), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(fallback)


def _first_observation_value(robot_obs: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        value = robot_obs.get(key)
        if value is not None:
            return value
    return default


def _safe_joint_vector(task, robot_name: str, method_name: str, *, width: int = 7) -> tuple[np.ndarray, bool]:
    robot = getattr(task, 'robots', {}).get(robot_name)
    articulation = getattr(robot, 'articulation', None)
    if articulation is None:
        return np.zeros(width, dtype=np.float32), False
    try:
        method = getattr(articulation, method_name)
        values = np.asarray(method(), dtype=np.float32).reshape(-1)
        values = values[:width]
        if values.size < width:
            values = np.pad(values, (0, width - values.size))
        if not np.all(np.isfinite(values)):
            return np.zeros(width, dtype=np.float32), False
        return values.astype(np.float32, copy=False), True
    except Exception:
        return np.zeros(width, dtype=np.float32), False


def _safe_full_joint_vector(task, robot_name: str) -> np.ndarray:
    robot = getattr(task, 'robots', {}).get(robot_name)
    articulation = getattr(robot, 'articulation', None)
    if articulation is None:
        return np.empty(0, dtype=np.float32)
    try:
        values = np.asarray(articulation.get_joint_positions(), dtype=np.float32).reshape(-1)
    except Exception:
        return np.empty(0, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        return np.empty(0, dtype=np.float32)
    return values


def _safe_joint_effort_vector(task, robot_name: str, *, width: int = 7) -> tuple[np.ndarray, bool]:
    robot = getattr(task, 'robots', {}).get(robot_name)
    articulation = getattr(robot, 'articulation', None)
    if articulation is None:
        return np.zeros(width, dtype=np.float32), False
    try:
        unwrapped = articulation.unwrap()
    except Exception:
        unwrapped = None
    candidates = [articulation, unwrapped]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, 'get_joint_efforts'):
            continue
        try:
            values = np.asarray(candidate.get_joint_efforts(), dtype=np.float32).reshape(-1)[:width]
            if values.size < width:
                values = np.pad(values, (0, width - values.size))
            if np.all(np.isfinite(values)):
                return values.astype(np.float32, copy=False), True
        except Exception:
            continue
    return np.zeros(width, dtype=np.float32), False


def _safe_wrist_wrench(task, robot_name: str, robot_obs: dict[str, Any]) -> tuple[np.ndarray, bool]:
    """Read an optional 6D wrist wrench while keeping Isaac backends optional."""
    candidates = []
    for key in ('wrist_wrench', 'eef_wrench', 'wrench', 'force_torque', 'wrist_force_torque'):
        if robot_obs.get(key) is not None:
            candidates.append(robot_obs[key])
    robot = getattr(task, 'robots', {}).get(robot_name)
    articulation = getattr(robot, 'articulation', None)
    unwrapped = None
    if articulation is not None:
        try:
            unwrapped = articulation.unwrap()
        except Exception:
            pass
    for candidate in (articulation, unwrapped):
        if candidate is None:
            continue
        for method_name in ('get_wrist_wrench', 'get_eef_wrench', 'get_net_contact_forces'):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    candidates.append(method())
                except Exception:
                    pass
    for value in candidates:
        try:
            values = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if values.size >= 6 and np.all(np.isfinite(values[:6])):
            return values[:6].astype(np.float32, copy=False), True
    return np.zeros(6, dtype=np.float32), False


def _phase_subtask_name(phase_name: str) -> str:
    tokens = [token for token in str(phase_name).split('_') if token]
    if len(tokens) >= 2 and tokens[0] in {'base', 'part', 'handoff', 'fixture'}:
        return '_'.join(tokens[:2])
    return tokens[0] if tokens else 'unknown'


def _safe_task_method(task, method_name: str, default: Any) -> Any:
    method = getattr(task, method_name, None)
    if not callable(method):
        return default
    try:
        return method()
    except Exception:
        return default


def runtime_scene_integrity(task) -> dict[str, Any]:
    """Snapshot the composed scene, robot, and camera prims for one episode."""
    result: dict[str, Any] = {
        'checked': False,
        'valid': False,
        'scene': {},
        'robots': {},
        'objects': {},
        'cameras': {},
        'runtime_root': '',
        'isolated_camera_paths': False,
        'errors': [],
    }
    try:
        scene_wrapper = getattr(task, '_scene', None)
        scene_prim = getattr(scene_wrapper, 'scene_prim', None)
        if scene_prim is None or not scene_prim.IsValid():
            result['errors'].append('scene-prim-missing')
            return result
        stage = scene_prim.GetStage()
        scene_path = str(scene_prim.GetPath())
        runtime_root = scene_path.rsplit('/scene', 1)[0] if '/scene' in scene_path else ''
        result['runtime_root'] = runtime_root

        def prim_status(prim_path: str, *, inspect_geometry: bool = False) -> dict[str, Any]:
            prim = stage.GetPrimAtPath(prim_path) if prim_path else None
            valid = bool(prim and prim.IsValid())
            active = bool(valid and prim.IsActive())
            loaded = bool(valid and prim.IsLoaded())
            visibility = None
            if valid:
                visibility_attr = prim.GetAttribute('visibility')
                if visibility_attr and visibility_attr.IsValid():
                    visibility = visibility_attr.Get()
            visible = visibility not in {'invisible'}
            status = {
                'prim_path': str(prim_path),
                'valid': valid,
                'active': active,
                'loaded': loaded,
                'visible': bool(visible),
            }
            if inspect_geometry:
                from pxr import Usd, UsdGeom

                geometry_count = 0
                renderable_geometry_count = 0
                if valid:
                    for descendant in Usd.PrimRange(prim):
                        if not descendant.IsA(UsdGeom.Gprim):
                            continue
                        geometry_count += 1
                        imageable = UsdGeom.Imageable(descendant)
                        purpose = str(imageable.ComputePurpose()) if imageable else 'default'
                        descendant_visible = str(imageable.ComputeVisibility()) if imageable else 'inherited'
                        if purpose in {'default', 'render'} and descendant_visible != 'invisible':
                            renderable_geometry_count += 1
                status['geometry_count'] = geometry_count
                status['renderable_geometry_count'] = renderable_geometry_count
            return status

        scene_status = prim_status(scene_path)
        descendant_count = 0
        pending = list(scene_prim.GetChildren())
        while pending:
            child = pending.pop()
            descendant_count += 1
            pending.extend(child.GetChildren())
        scene_status['descendant_count'] = descendant_count
        result['scene'] = scene_status

        for robot_name, robot in (getattr(task, 'robots', {}) or {}).items():
            robot_path = str(getattr(getattr(robot, 'config', None), 'prim_path', '') or '')
            result['robots'][str(robot_name)] = prim_status(robot_path)
            for sensor_name, sensor in (getattr(robot, 'sensors', {}) or {}).items():
                camera_path = str(getattr(sensor, 'camera_prim_path', '') or '')
                if camera_path:
                    result['cameras'][str(sensor_name)] = prim_status(camera_path)

        for object_name, scene_object in (getattr(task, 'objects', {}) or {}).items():
            object_path = str(getattr(getattr(scene_object, 'config', None), 'prim_path', '') or '')
            result['objects'][str(object_name)] = prim_status(object_path, inspect_geometry=True)

        camera_paths = [status['prim_path'] for status in result['cameras'].values()]
        result['isolated_camera_paths'] = bool(
            runtime_root
            and camera_paths
            and all(path.startswith(runtime_root.rstrip('/') + '/') for path in camera_paths)
        )
        required_statuses = [
            result['scene'],
            *result['robots'].values(),
            *result['cameras'].values(),
        ]
        if descendant_count <= 0:
            result['errors'].append('scene-composition-empty')
        if len(result['robots']) < 2:
            result['errors'].append('robot-count')
        if len(result['cameras']) < len(CAMERA_KEYS):
            result['errors'].append('camera-count')
        tracked_names = tuple(getattr(task.config, 'tracked_object_names', ()) or ())
        required_object_names = set(tracked_names) | {'fabrica_fixture', 'optical_board'}
        for object_name in sorted(required_object_names):
            status = result['objects'].get(object_name)
            if status is None:
                result['errors'].append(f'object-missing:{object_name}')
            else:
                required_statuses.append(status)
                if not status.get('valid') or not status.get('active') or not status.get('loaded'):
                    result['errors'].append(f'object-invalid:{object_name}')
                elif not status.get('visible') or int(status.get('renderable_geometry_count', 0)) <= 0:
                    result['errors'].append(f'object-not-renderable:{object_name}')
        if not result['isolated_camera_paths']:
            result['errors'].append('camera-path-not-episode-scoped')
        if any(
            not status.get('valid')
            or not status.get('active')
            or not status.get('loaded')
            or not status.get('visible')
            for status in required_statuses
        ):
            result['errors'].append('prim-invalid')
        result['checked'] = True
        result['errors'] = sorted(set(result['errors']))
        result['valid'] = not result['errors']
        return result
    except Exception as exc:
        result['errors'].append(f'check-failed:{type(exc).__name__}')
        return result


def _phase_annotation(task, phase_spec: dict[str, Any], phase_index: int) -> dict[str, Any]:
    name = str(phase_spec.get('name') or f'phase_{phase_index}')
    robot_names = set(getattr(task.config, 'robot_names', ROBOT_NAMES))
    referenced_robots: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {'robot', 'robot_name', 'owner'} and str(item) in robot_names:
                    referenced_robots.add(str(item))
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value in robot_names:
            referenced_robots.add(value)

    visit(phase_spec)
    roles = getattr(task.config, 'task_metadata', {}) or {}
    robot_roles = roles.get('robot_roles') if isinstance(roles, dict) else None
    if not isinstance(robot_roles, dict):
        robot_roles = {
            'franka_left': 'assembly_robot',
            'franka_right': 'base_robot',
        }
    lower_name = name.lower()
    return {
        'phase_index': int(phase_index),
        'name': name,
        'subtask_id': _phase_subtask_name(name),
        'substage_id': name,
        'robot_names': sorted(referenced_robots),
        'robot_roles': {robot: robot_roles.get(robot, 'coordinator') for robot in sorted(referenced_robots)},
        'preconditions': _jsonable(
            phase_spec.get('preconditions', phase_spec.get('conditions', phase_spec.get('requires', [])))
        ),
        'completion_conditions': _jsonable(
            phase_spec.get('completion_conditions', phase_spec.get('advance', phase_spec.get('conditions', [])))
        ),
        'waiting': bool(phase_spec.get('wait') or phase_spec.get('waiting') or 'wait' in lower_name),
        'handoff': bool(phase_spec.get('handoff') or 'handoff' in lower_name or 'transfer' in lower_name),
        'spec': _jsonable(phase_spec),
    }


class CartesianObservationEncoder:
    """Encode synchronized task observations into the LeRobot policy schema."""

    def __init__(
        self,
        *,
        task,
        output_resolution: tuple[int, int] = (640, 480),
        front_output_resolution: tuple[int, int] | None = None,
    ):
        if len(output_resolution) != 2 or any(int(value) <= 0 for value in output_resolution):
            raise ValueError(f'Invalid dataset output resolution: {output_resolution}.')
        self.output_resolution = tuple(int(value) for value in output_resolution)
        self.front_output_resolution = tuple(
            int(value) for value in (front_output_resolution or output_resolution)
        )
        if any(value <= 0 for value in self.front_output_resolution):
            raise ValueError(f'Invalid front output resolution: {self.front_output_resolution}.')
        self.camera_bindings = self.resolve_camera_bindings(task)
        self._last_quaternions: dict[str, np.ndarray] = {}

    @staticmethod
    def resolve_camera_bindings(task) -> dict[str, tuple[str, str]]:
        bindings: dict[str, tuple[str, str]] = {}
        for camera in getattr(task.config, 'camera_metadata', []):
            if not isinstance(camera, dict):
                continue
            owner = str(camera.get('owner') or camera.get('robot') or 'franka_left')
            sensor_name = str(camera.get('name') or '')
            view_type = str(camera.get('view_type') or '').lower()
            robot = str(camera.get('robot') or owner)
            if view_type == 'front':
                bindings['observation.images.front'] = (owner, sensor_name)
            elif view_type == 'wrist' and robot == 'franka_left':
                bindings['observation.images.left_wrist'] = (owner, sensor_name)
            elif view_type == 'wrist' and robot == 'franka_right':
                bindings['observation.images.right_wrist'] = (owner, sensor_name)
        missing = [key for key in CAMERA_KEYS if key not in bindings]
        if missing:
            raise ValueError(f'Task camera metadata is missing required dataset views: {missing}.')
        return bindings

    @staticmethod
    def rgb_frame(obs: dict[str, Any], binding: tuple[str, str]) -> np.ndarray | None:
        owner, sensor_name = binding
        sensor_obs = ((obs.get(owner) or {}).get('sensors') or {}).get(sensor_name)
        if not isinstance(sensor_obs, dict) or sensor_obs.get('rgba') is None:
            return None
        frame = np.asarray(sensor_obs['rgba'])
        if frame.ndim != 3 or frame.shape[-1] < 3:
            return None
        frame = frame[..., :3]
        if np.issubdtype(frame.dtype, np.floating):
            if frame.size and float(np.nanmax(frame)) <= 1.0 + 1e-6:
                frame = frame * 255.0
        return np.clip(np.nan_to_num(frame), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def depth_frame(obs: dict[str, Any], binding: tuple[str, str]) -> np.ndarray | None:
        owner, sensor_name = binding
        sensor_obs = ((obs.get(owner) or {}).get('sensors') or {}).get(sensor_name)
        if not isinstance(sensor_obs, dict):
            return None
        depth_value = None
        for key in ('depth', 'distance_to_image_plane', 'distance_to_camera', 'depth_linear'):
            if sensor_obs.get(key) is not None:
                depth_value = sensor_obs[key]
                break
        if depth_value is None:
            return None
        frame = np.asarray(depth_value)
        if frame.ndim == 3:
            frame = frame[..., 0]
        if frame.ndim != 2:
            return None
        frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(frame, 0.0)

    def letterbox_frame(self, frame: np.ndarray, camera_key: str | None = None) -> np.ndarray:
        target_width, target_height = (
            self.front_output_resolution
            if camera_key == 'observation.images.front'
            else self.output_resolution
        )
        source_height, source_width = frame.shape[:2]
        if (source_width, source_height) == (target_width, target_height):
            return frame
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(int(round(source_width * scale)), 1)
        resized_height = max(int(round(source_height * scale)), 1)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - resized_width) // 2
        y_offset = (target_height - resized_height) // 2
        canvas[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ] = resized
        return canvas

    def letterbox_depth(self, frame: np.ndarray) -> np.ndarray:
        target_width, target_height = self.output_resolution
        source_height, source_width = frame.shape[:2]
        if (source_width, source_height) == (target_width, target_height):
            return frame.astype(np.float32, copy=False)
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(int(round(source_width * scale)), 1)
        resized_height = max(int(round(source_height * scale)), 1)
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target_height, target_width), dtype=np.float32)
        x_offset = (target_width - resized_width) // 2
        y_offset = (target_height - resized_height) // 2
        canvas[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ] = resized
        return canvas

    def state(self, task, obs: dict[str, Any]) -> np.ndarray:
        values = []
        for robot_name in ROBOT_NAMES:
            robot_obs = obs.get(robot_name) or {}
            position = np.asarray(
                _first_observation_value(
                    robot_obs,
                    'eef_position',
                    'position',
                    default=[0.0, 0.0, 0.0],
                ),
                dtype=np.float64,
            ).reshape(-1)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f'Invalid {robot_name} Cartesian position: {position}.')
            quaternion = _normalize_quaternion(
                _first_observation_value(
                    robot_obs,
                    'eef_orientation',
                    'orientation',
                    default=[1.0, 0.0, 0.0, 0.0],
                ),
                reference=self._last_quaternions.get(robot_name),
            )
            self._last_quaternions[robot_name] = quaternion
            values.extend(position.tolist())
            values.extend(quaternion.tolist())
            values.append(_gripper_openness(task, robot_name, robot_obs))
        state = np.asarray(values, dtype=np.float32)
        if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
            raise ValueError(f'Invalid Cartesian state: {state}.')
        return state

    def encode(self, *, task, obs: dict[str, Any]) -> dict[str, np.ndarray] | None:
        frames = {key: self.rgb_frame(obs, self.camera_bindings[key]) for key in CAMERA_KEYS}
        if not all(frame is not None for frame in frames.values()):
            return None
        encoded = {key: self.letterbox_frame(frame, key) for key, frame in frames.items()}
        encoded['observation.state'] = self.state(task, obs)
        return encoded


class CompactCartesianEpisodeRecorder:
    """Record synchronized RGB, Cartesian state, and 30 Hz absolute actions.

    RGB is streamed directly to one MP4 per camera. Low-dimensional arrays stay
    in memory and are written once as a compressed NPZ, keeping a one-minute
    episode small enough for long-running 2k collection jobs.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        episode_idx: int,
        task,
        fps: int = 30,
        frame_stride: int = 8,
        simulation_fps: int | None = None,
        rendering_interval: int | None = None,
        output_resolution: tuple[int, int] = (640, 480),
        record_media: bool = True,
        video_codec: str = 'h264',
        video_crf: int = 23,
        video_preset: str = 'veryfast',
        depth_compression_level: int = 5,
        allow_replay_recipe_fingerprint_mismatch: bool = False,
        overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.episode_idx = int(episode_idx)
        self.fps = max(int(fps), 1)
        self.frame_stride = max(int(frame_stride), 1)
        self.simulation_fps = self.fps * self.frame_stride if simulation_fps is None else max(int(simulation_fps), 1)
        self.rendering_interval = self.frame_stride - 1 if rendering_interval is None else int(rendering_interval)
        self.record_media = bool(record_media)
        self.video_codec = str(video_codec)
        self.video_crf = int(video_crf)
        self.video_preset = str(video_preset)
        self.depth_compression_level = int(depth_compression_level)
        self._allow_replay_recipe_fingerprint_mismatch = bool(allow_replay_recipe_fingerprint_mismatch)
        configured_resolution = (
            int(os.environ.get('RAB_DATASET_OUTPUT_WIDTH', output_resolution[0])),
            int(os.environ.get('RAB_DATASET_OUTPUT_HEIGHT', output_resolution[1])),
        )
        configured_front_resolution = (
            int(os.environ.get('RAB_FRONT_OUTPUT_WIDTH', configured_resolution[0])),
            int(os.environ.get('RAB_FRONT_OUTPUT_HEIGHT', configured_resolution[1])),
        )
        if self.rendering_interval < 0:
            raise ValueError('rendering_interval must be non-negative.')
        if self.simulation_fps != self.fps * self.frame_stride:
            raise ValueError(
                'Raw Cartesian timing requires simulation_fps == fps * frame_stride; '
                f'got {self.simulation_fps} != {self.fps} * {self.frame_stride}.'
            )
        if self.rendering_interval + 1 != self.frame_stride:
            raise ValueError(
                'Raw Cartesian timing requires rendering_interval + 1 == frame_stride; '
                f'got {self.rendering_interval} + 1 != {self.frame_stride}.'
            )
        self._observation_encoder = CartesianObservationEncoder(
            task=task,
            output_resolution=configured_resolution,
            front_output_resolution=configured_front_resolution,
        )
        self.output_resolution = self._observation_encoder.output_resolution
        self.front_output_resolution = self._observation_encoder.front_output_resolution
        self.episode_dir = self.output_dir / f'episode_{self.episode_idx:06d}_cartesian_raw'
        if self.episode_dir.exists() and not overwrite:
            raise FileExistsError(f'Raw Cartesian episode already exists: {self.episode_dir}')
        self.video_dir = self.episode_dir / 'videos'
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if self.record_media:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        self._camera_bindings = self._observation_encoder.camera_bindings
        self._writers: dict[str, FFmpegVideoWriter] = {}
        self._video_encoding_metadata: dict[str, dict[str, Any]] = {}
        self._video_paths: dict[str, Path] = {}
        self._video_shapes: dict[str, tuple[int, int, int]] = {}
        self._source_video_shapes: dict[str, tuple[int, int, int]] = {}
        self._frame_counts = {key: 0 for key in CAMERA_KEYS}
        self._states: list[np.ndarray] = []
        self._gripper_commands: list[np.ndarray] = []
        self._simulation_steps: list[int] = []
        self._phase_indices: list[int] = []
        self._phase_steps: list[int] = []
        self._phase_names: list[str] = []
        self._phase_annotations = {
            str(phase.get('name') or f'phase_{index}'): _phase_annotation(task, phase, index)
            for index, phase in enumerate(getattr(task.config, 'phase_specs', []) or [])
            if isinstance(phase, dict)
        }
        self._subtask_indices: list[int] = []
        self._substage_indices: list[int] = []
        self._waiting_states: list[int] = []
        self._handoff_states: list[int] = []
        self._joint_states: list[np.ndarray] = []
        self._joint_velocities: list[np.ndarray] = []
        self._joint_state_available: list[int] = []
        self._joint_velocity_available: list[int] = []
        self._joint_efforts: list[np.ndarray] = []
        self._joint_effort_available: list[int] = []
        self._wrist_wrenches: list[np.ndarray] = []
        self._wrist_wrench_available: list[int] = []
        self._collision_signals: list[np.ndarray] = []
        self._replay_joint_states: list[list[np.ndarray]] = []
        self._tracked_object_names: list[str] | None = None
        self._tracked_object_positions: list[np.ndarray] = []
        self._tracked_object_orientations: list[np.ndarray] = []
        self._depth_handles: dict[str, ChunkedDepthWriter] = {}
        self._depth_metadata: dict[str, dict[str, Any]] = {}
        self._runtime_scene_integrity_start = runtime_scene_integrity(task)
        self._expected_recipe_fingerprint = str(getattr(task.config, 'recipe_fingerprint', '') or '')
        self._expected_replay_robot_names = list(getattr(task.config, 'robot_names', ROBOT_NAMES) or ROBOT_NAMES)
        self._expected_replay_joint_widths = expected_replay_joint_widths(
            getattr(task.config, 'robot_metadata', [])
        )
        self._source_actions: np.ndarray | None = None
        self._replay_source_metadata_path: str | None = None
        self._replay_source_recipe_fingerprint: str | None = None
        self._replay_recipe_fingerprint_mismatch = False
        self._finalized = False

    @staticmethod
    def _resolve_camera_bindings(task) -> dict[str, tuple[str, str]]:
        return CartesianObservationEncoder.resolve_camera_bindings(task)

    @staticmethod
    def _rgb_frame(obs: dict[str, Any], binding: tuple[str, str]) -> np.ndarray | None:
        return CartesianObservationEncoder.rgb_frame(obs, binding)

    def _letterbox_frame(self, frame: np.ndarray, camera_key: str | None = None) -> np.ndarray:
        return self._observation_encoder.letterbox_frame(frame, camera_key)

    def _state(self, task, obs: dict[str, Any]) -> np.ndarray:
        return self._observation_encoder.state(task, obs)

    def _record_depth(self, obs: dict[str, Any]) -> None:
        depth_root = self.episode_dir / 'sensors' / 'depth'
        for camera_key, binding in self._camera_bindings.items():
            depth = self._observation_encoder.depth_frame(obs, binding)
            if depth is None:
                continue
            depth = self._observation_encoder.letterbox_depth(depth)
            if camera_key not in self._depth_handles:
                depth_root.mkdir(parents=True, exist_ok=True)
                path = depth_root / f'{camera_key.replace(".", "_")}.u16.bshuf.zst'
                self._depth_handles[camera_key] = ChunkedDepthWriter(
                    path,
                    shape=tuple(int(value) for value in depth.shape),
                    compression_level=self.depth_compression_level,
                )
            self._depth_handles[camera_key].write(depth)

    def _record_replay_state(self, task) -> None:
        full_joint_state = [_safe_full_joint_vector(task, robot_name) for robot_name in ROBOT_NAMES]
        if any(values.size == 0 for values in full_joint_state):
            raise RuntimeError('Trajectory replay requires full articulation joint state for both robots.')
        if self._replay_joint_states:
            expected_widths = [len(values) for values in self._replay_joint_states[0]]
            actual_widths = [len(values) for values in full_joint_state]
            if actual_widths != expected_widths:
                raise RuntimeError(f'Robot joint widths changed from {expected_widths} to {actual_widths}.')
        self._replay_joint_states.append(full_joint_state)

        tracked = _safe_task_method(task, 'get_tracked_object_states', {}) or {}
        names = sorted(str(name) for name in tracked)
        if self._tracked_object_names is None:
            self._tracked_object_names = names
        elif names != self._tracked_object_names:
            raise RuntimeError(f'Tracked objects changed from {self._tracked_object_names} to {names}.')
        positions = []
        orientations = []
        for name in names:
            state = tracked[name]
            position = np.asarray(state.get('position'), dtype=np.float32).reshape(-1)
            orientation = _normalize_quaternion(state.get('orientation')).astype(np.float32)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise RuntimeError(f'Invalid tracked-object position for {name!r}: {position}.')
            positions.append(position)
            orientations.append(orientation)
        self._tracked_object_positions.append(
            np.stack(positions) if positions else np.empty((0, 3), dtype=np.float32)
        )
        self._tracked_object_orientations.append(
            np.stack(orientations) if orientations else np.empty((0, 4), dtype=np.float32)
        )

    @staticmethod
    def _phase_indices(phase_annotation: dict[str, Any], phase_annotations: dict[str, dict[str, Any]]) -> tuple[int, int]:
        phase_index = int(phase_annotation.get('phase_index', -1))
        subtask_names = sorted({item.get('subtask_id', 'unknown') for item in phase_annotations.values()})
        subtask_index = subtask_names.index(phase_annotation.get('subtask_id', 'unknown')) if phase_annotation else -1
        return subtask_index, phase_index

    @staticmethod
    def _collision_signal(task, *, phase_annotation: dict[str, Any]) -> np.ndarray:
        tracked = _safe_task_method(task, 'get_tracked_object_states', {}) or {}
        left_contact = any(
            state.get('attached_to') == 'franka_left' or state.get('grasped_by') == 'franka_left'
            for state in tracked.values()
            if isinstance(state, dict)
        )
        right_contact = any(
            state.get('attached_to') == 'franka_right' or state.get('grasped_by') == 'franka_right'
            for state in tracked.values()
            if isinstance(state, dict)
        )
        locked_count = sum(bool(state.get('locked_target')) for state in tracked.values() if isinstance(state, dict))
        collision_detected = bool(
            getattr(task, 'collision_detected', False)
            or getattr(task, 'collision_violation', False)
            or getattr(task, '_collision_detected', False)
        )
        return np.asarray(
            [float(collision_detected), float(left_contact), float(right_contact), float(locked_count)],
            dtype=np.float32,
        )

    def _record_low_dimensional_extras(self, task, obs: dict[str, Any], phase_annotation: dict[str, Any]) -> None:
        joint_states = []
        joint_velocities = []
        joint_efforts = []
        joint_state_available = True
        joint_velocity_available = True
        effort_available = True
        for robot_name in ROBOT_NAMES:
            q, q_available = _safe_joint_vector(task, robot_name, 'get_joint_positions')
            dq, dq_available = _safe_joint_vector(task, robot_name, 'get_joint_velocities')
            tau, tau_available = _safe_joint_effort_vector(task, robot_name)
            joint_states.append(q)
            joint_velocities.append(dq)
            joint_efforts.append(tau)
            joint_state_available = joint_state_available and q_available
            joint_velocity_available = joint_velocity_available and dq_available
            effort_available = effort_available and tau_available
        self._joint_states.append(np.concatenate(joint_states).astype(np.float32, copy=False))
        self._joint_velocities.append(np.concatenate(joint_velocities).astype(np.float32, copy=False))
        self._joint_state_available.append(int(joint_state_available))
        self._joint_velocity_available.append(int(joint_velocity_available))
        self._joint_efforts.append(np.concatenate(joint_efforts).astype(np.float32, copy=False))
        self._joint_effort_available.append(int(effort_available))
        wrench_values = []
        wrench_available = True
        for robot_name in ROBOT_NAMES:
            wrench, available = _safe_wrist_wrench(task, robot_name, obs.get(robot_name) or {})
            wrench_values.append(wrench)
            wrench_available = wrench_available and available
        self._wrist_wrenches.append(np.concatenate(wrench_values).astype(np.float32, copy=False))
        self._wrist_wrench_available.append(int(wrench_available))
        self._collision_signals.append(self._collision_signal(task, phase_annotation=phase_annotation))

    def _phase_annotation_for_task(self, task) -> dict[str, Any]:
        phase_index = getattr(task, 'phase_index', -1)
        try:
            phase_index = int(phase_index)
        except (TypeError, ValueError):
            phase_index = -1
        phase_name = str(getattr(task, 'phase', '') or '')
        if phase_name in self._phase_annotations:
            annotation = self._phase_annotations[phase_name]
        elif 0 <= phase_index < len(getattr(task.config, 'phase_specs', []) or []):
            phase_spec = task.config.phase_specs[phase_index]
            annotation = _phase_annotation(task, phase_spec, phase_index) if isinstance(phase_spec, dict) else {}
        else:
            annotation = {}
        annotation = dict(annotation)
        annotation.setdefault('phase_index', phase_index)
        annotation.setdefault('name', phase_name or f'phase_{phase_index}')
        annotation.setdefault('subtask_id', _phase_subtask_name(annotation['name']))
        annotation.setdefault('substage_id', annotation['name'])
        return annotation

    def _open_writer(self, camera_key: str, frame: np.ndarray) -> FFmpegVideoWriter:
        height, width = frame.shape[:2]
        path = self.video_dir / f'{camera_key.replace(".", "_")}.mp4'
        writer = FFmpegVideoWriter(
            path,
            width=int(width),
            height=int(height),
            fps=self.fps,
            codec=self.video_codec,
            crf=self.video_crf,
            preset=self.video_preset,
        )
        self._writers[camera_key] = writer
        self._video_paths[camera_key] = path
        self._video_shapes[camera_key] = (int(height), int(width), 3)
        return writer

    def _record_media_frame(self, obs: dict[str, Any]) -> bool:
        frames = {key: self._rgb_frame(obs, self._camera_bindings[key]) for key in CAMERA_KEYS}
        if not all(frame is not None for frame in frames.values()):
            return False
        for camera_key, frame in frames.items():
            self._source_video_shapes.setdefault(camera_key, tuple(int(value) for value in frame.shape))
            frame = self._letterbox_frame(frame, camera_key)
            writer = self._writers.get(camera_key) or self._open_writer(camera_key, frame)
            expected_shape = self._video_shapes[camera_key]
            if tuple(frame.shape) != expected_shape:
                raise ValueError(f'Camera {camera_key} changed shape from {expected_shape} to {tuple(frame.shape)}.')
            writer.write(frame)
            self._frame_counts[camera_key] += 1
        self._record_depth(obs)
        return True

    def prepare_replay_source(self, metadata_path: Path) -> int:
        """Load authoritative Stage-1 signals before rendering a visual variant."""

        if self._states:
            raise RuntimeError('Replay source must be loaded before any frames are recorded.')
        metadata_path = Path(metadata_path).resolve()
        source_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        source_fingerprint = str(source_metadata.get('recipe_fingerprint', '') or '')
        if self._expected_recipe_fingerprint and source_fingerprint != self._expected_recipe_fingerprint:
            if not self._allow_replay_recipe_fingerprint_mismatch:
                raise ValueError(
                    f'Replay source recipe fingerprint {source_fingerprint!r} does not match '
                    f'the runtime recipe {self._expected_recipe_fingerprint!r}: {metadata_path}.'
                )
            self._replay_recipe_fingerprint_mismatch = True
        self._replay_source_recipe_fingerprint = source_fingerprint
        trajectory_path = Path(source_metadata.get('trajectory_path') or '')
        if not trajectory_path.is_file():
            raise FileNotFoundError(f'Replay source trajectory is missing: {trajectory_path}.')
        with np.load(trajectory_path) as source:
            required = {
                'observation_state',
                'action',
                'expert_gripper_command',
                'simulation_step',
                'phase_index',
                'phase_step',
                'phase_name',
                'subtask_index',
                'substage_index',
                'waiting_state',
                'handoff_state',
                'joint_state',
                'joint_velocity',
                'joint_state_available',
                'joint_velocity_available',
                'joint_effort',
                'joint_effort_available',
                'wrist_wrench',
                'wrist_wrench_available',
                'collision_signal',
                'replay_robot_names',
                'replay_joint_widths',
                'replay_joint_state',
                'tracked_object_names',
                'tracked_object_position',
                'tracked_object_orientation',
            }
            missing = sorted(required.difference(source.files))
            if missing:
                raise ValueError(f'Replay source {trajectory_path} is missing arrays: {missing}.')
            frame_count = int(source_metadata.get('frame_count', 0))
            states = np.asarray(source['observation_state'], dtype=np.float32)
            if states.shape != (frame_count, len(STATE_NAMES)) or frame_count <= 0:
                raise ValueError(f'Invalid replay source state shape: {states.shape}.')
            self._states = list(states)
            self._source_actions = np.asarray(source['action'], dtype=np.float32)
            self._gripper_commands = list(np.asarray(source['expert_gripper_command'], dtype=np.float32))
            self._simulation_steps = list(np.asarray(source['simulation_step'], dtype=np.int64))
            self._phase_indices = list(np.asarray(source['phase_index'], dtype=np.int16))
            self._phase_steps = list(np.asarray(source['phase_step'], dtype=np.int32))
            self._phase_names = [str(value) for value in np.asarray(source['phase_name']).tolist()]
            self._subtask_indices = list(np.asarray(source['subtask_index'], dtype=np.int16))
            self._substage_indices = list(np.asarray(source['substage_index'], dtype=np.int16))
            self._waiting_states = list(np.asarray(source['waiting_state'], dtype=np.uint8))
            self._handoff_states = list(np.asarray(source['handoff_state'], dtype=np.uint8))
            self._joint_states = list(np.asarray(source['joint_state'], dtype=np.float32))
            self._joint_velocities = list(np.asarray(source['joint_velocity'], dtype=np.float32))
            self._joint_state_available = list(np.asarray(source['joint_state_available'], dtype=np.uint8))
            self._joint_velocity_available = list(np.asarray(source['joint_velocity_available'], dtype=np.uint8))
            self._joint_efforts = list(np.asarray(source['joint_effort'], dtype=np.float32))
            self._joint_effort_available = list(np.asarray(source['joint_effort_available'], dtype=np.uint8))
            self._wrist_wrenches = list(np.asarray(source['wrist_wrench'], dtype=np.float32))
            self._wrist_wrench_available = list(np.asarray(source['wrist_wrench_available'], dtype=np.uint8))
            self._collision_signals = list(np.asarray(source['collision_signal'], dtype=np.float32))
            widths = [int(value) for value in np.asarray(source['replay_joint_widths']).tolist()]
            replay_robot_names = [str(value) for value in np.asarray(source['replay_robot_names']).tolist()]
            if replay_robot_names != self._expected_replay_robot_names:
                raise ValueError(
                    f'Replay source robot names {replay_robot_names} do not match '
                    f'{self._expected_replay_robot_names}: {trajectory_path}.'
                )
            if self._expected_replay_joint_widths is not None and widths != self._expected_replay_joint_widths:
                raise ValueError(
                    f'Replay source joint widths {widths} do not match the Franka articulation '
                    f'signature {self._expected_replay_joint_widths}: {trajectory_path}.'
                )
            replay_joint_state = np.asarray(source['replay_joint_state'], dtype=np.float32)
            self._replay_joint_states = [
                list(np.split(frame, np.cumsum(widths)[:-1]))
                for frame in replay_joint_state
            ]
            self._tracked_object_names = [str(value) for value in np.asarray(source['tracked_object_names']).tolist()]
            self._tracked_object_positions = list(np.asarray(source['tracked_object_position'], dtype=np.float32))
            self._tracked_object_orientations = list(
                np.asarray(source['tracked_object_orientation'], dtype=np.float32)
            )
        arrays = [
            self._source_actions,
            self._gripper_commands,
            self._simulation_steps,
            self._phase_indices,
            self._replay_joint_states,
            self._tracked_object_positions,
        ]
        if any(len(values) != frame_count for values in arrays):
            raise ValueError(f'Replay source arrays are not aligned to {frame_count} frames.')
        self._phase_annotations = dict(source_metadata.get('phase_annotations') or self._phase_annotations)
        self._replay_source_metadata_path = str(metadata_path)
        return frame_count

    def record_replay_frame(self, *, obs: dict[str, Any], frame_index: int) -> bool:
        if self._replay_source_metadata_path is None:
            raise RuntimeError('prepare_replay_source() must be called before replay rendering.')
        expected_index = next(iter(self._frame_counts.values()))
        if int(frame_index) != expected_index:
            raise ValueError(f'Replay frames must be sequential: {frame_index} != {expected_index}.')
        if int(frame_index) >= len(self._states):
            raise IndexError(frame_index)
        return self._record_media_frame(obs)

    def record(self, *, task, obs: dict[str, Any], actions: dict[str, Any]) -> bool:
        if int(task.step_counter) % self.frame_stride != 0:
            return False

        if self.record_media:
            if not self._record_media_frame(obs):
                return False
        state = self._state(task, obs)
        gripper_commands = np.asarray(
            [
                _gripper_command(actions, robot_name, state[index * 8 + 7])
                for index, robot_name in enumerate(ROBOT_NAMES)
            ],
            dtype=np.float32,
        )
        self._states.append(state)
        self._gripper_commands.append(gripper_commands)
        self._simulation_steps.append(int(task.step_counter))
        phase_annotation = self._phase_annotation_for_task(task)
        phase_index = int(phase_annotation.get('phase_index', -1))
        phase_name = str(phase_annotation.get('name', ''))
        subtask_names = sorted({item.get('subtask_id', 'unknown') for item in self._phase_annotations.values()})
        subtask_id = str(phase_annotation.get('subtask_id', 'unknown'))
        subtask_index = subtask_names.index(subtask_id) if subtask_id in subtask_names else -1
        self._phase_indices.append(phase_index)
        self._phase_steps.append(int(getattr(task, 'phase_step_counter', -1)))
        self._phase_names.append(phase_name)
        self._subtask_indices.append(subtask_index)
        self._substage_indices.append(phase_index)
        self._waiting_states.append(int(bool(phase_annotation.get('waiting', False))))
        self._handoff_states.append(int(bool(phase_annotation.get('handoff', False))))
        self._record_low_dimensional_extras(task, obs, phase_annotation)
        self._record_replay_state(task)
        return True

    @staticmethod
    def _next_sample_actions(states: np.ndarray, gripper_commands: np.ndarray) -> np.ndarray:
        actions = states.copy()
        if len(actions) > 1:
            actions[:-1, 0:7] = states[1:, 0:7]
            actions[:-1, 8:15] = states[1:, 8:15]
            actions[:-1, 7] = gripper_commands[1:, 0]
            actions[:-1, 15] = gripper_commands[1:, 1]
        actions[-1, 7] = gripper_commands[-1, 0]
        actions[-1, 15] = gripper_commands[-1, 1]
        return actions.astype(np.float32, copy=False)

    def finalize(self, *, task, metrics: dict[str, Any]) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError(f'Raw Cartesian episode {self.episode_idx} was finalized twice.')
        self._finalized = True
        for camera_key, writer in self._writers.items():
            self._video_encoding_metadata[camera_key] = writer.close()
        for camera_key, handle in self._depth_handles.items():
            self._depth_metadata[camera_key] = handle.close()

        depth_index_path = self.episode_dir / 'sensors' / 'depth' / 'index.json'
        if self._depth_metadata:
            depth_index_path.parent.mkdir(parents=True, exist_ok=True)
            depth_index_path.write_text(json.dumps(_jsonable(self._depth_metadata), indent=2), encoding='utf-8')

        frame_counts = set(self._frame_counts.values())
        media_unaligned = self.record_media and frame_counts != {len(self._states)}
        if not self._states or media_unaligned:
            raise RuntimeError(
                f'Unaligned raw episode {self.episode_idx}: states={len(self._states)}, '
                f'camera_frames={self._frame_counts}.'
            )
        states = np.asarray(self._states, dtype=np.float32)
        gripper_commands = np.asarray(self._gripper_commands, dtype=np.float32)
        actions = (
            self._next_sample_actions(states, gripper_commands)
            if self._source_actions is None
            else np.asarray(self._source_actions, dtype=np.float32)
        )
        if actions.shape != states.shape:
            raise RuntimeError(f'Replay action shape {actions.shape} does not match states {states.shape}.')
        joint_state = np.asarray(self._joint_states, dtype=np.float32)
        joint_velocity = np.asarray(self._joint_velocities, dtype=np.float32)
        joint_effort = np.asarray(self._joint_efforts, dtype=np.float32)
        wrist_wrench = np.asarray(self._wrist_wrenches, dtype=np.float32)
        replay_joint_state = np.asarray(
            [np.concatenate(per_robot) for per_robot in self._replay_joint_states],
            dtype=np.float32,
        )
        replay_joint_widths = np.asarray(
            [len(values) for values in self._replay_joint_states[0]],
            dtype=np.int16,
        )
        tracked_object_position = np.asarray(self._tracked_object_positions, dtype=np.float32)
        tracked_object_orientation = np.asarray(self._tracked_object_orientations, dtype=np.float32)
        trajectory_path = self.episode_dir / 'trajectory.npz'
        np.savez_compressed(
            trajectory_path,
            observation_state=states,
            action=actions,
            expert_gripper_command=gripper_commands,
            simulation_step=np.asarray(self._simulation_steps, dtype=np.int64),
            phase_index=np.asarray(self._phase_indices, dtype=np.int16),
            phase_step=np.asarray(self._phase_steps, dtype=np.int32),
            phase_name=np.asarray(self._phase_names, dtype=np.str_),
            subtask_index=np.asarray(self._subtask_indices, dtype=np.int16),
            substage_index=np.asarray(self._substage_indices, dtype=np.int16),
            waiting_state=np.asarray(self._waiting_states, dtype=np.uint8),
            handoff_state=np.asarray(self._handoff_states, dtype=np.uint8),
            joint_state=joint_state,
            joint_velocity=joint_velocity,
            joint_state_available=np.asarray(self._joint_state_available, dtype=np.uint8),
            joint_velocity_available=np.asarray(self._joint_velocity_available, dtype=np.uint8),
            joint_effort=joint_effort,
            joint_effort_available=np.asarray(self._joint_effort_available, dtype=np.uint8),
            wrist_wrench=wrist_wrench,
            wrist_wrench_available=np.asarray(self._wrist_wrench_available, dtype=np.uint8),
            collision_signal=np.asarray(self._collision_signals, dtype=np.float32),
            replay_robot_names=np.asarray(ROBOT_NAMES, dtype=np.str_),
            replay_joint_widths=replay_joint_widths,
            replay_joint_state=replay_joint_state,
            tracked_object_names=np.asarray(self._tracked_object_names or [], dtype=np.str_),
            tracked_object_position=tracked_object_position,
            tracked_object_orientation=tracked_object_orientation,
        )

        phase_boundaries = []
        if self._phase_names:
            start = 0
            previous = self._phase_names[0]
            for index, current in enumerate(self._phase_names[1:], start=1):
                if current != previous:
                    phase_boundaries.append({'phase': previous, 'start_frame': start, 'end_frame': index - 1})
                    start = index
                    previous = current
            phase_boundaries.append({'phase': previous, 'start_frame': start, 'end_frame': len(self._phase_names) - 1})
        progress_annotations = task_progress_annotations(
            self._phase_annotations,
            self._subtask_indices,
            self._substage_indices,
        )

        config_metadata = getattr(task.config, 'task_metadata', {}) or {}
        default_robot_roles = {'franka_left': 'assembly_robot', 'franka_right': 'base_robot'}
        configured_robot_roles = config_metadata.get('robot_roles', {}) if isinstance(config_metadata, dict) else {}
        robot_roles = {**default_robot_roles, **configured_robot_roles}
        execution_order = [
            annotation['name']
            for annotation in sorted(self._phase_annotations.values(), key=lambda item: item.get('phase_index', -1))
        ]
        annotation_payload = {
            'schema_version': 'roboassemblybench_long_horizon_annotation_v1',
            'episode_idx': self.episode_idx,
            'task': str(task.config.task_description or task.config.prompt),
            'recipe': str(task.config.recipe),
            'task_goal': _jsonable(getattr(task.config, 'prompt', '')),
            'assembly_steps': _jsonable(getattr(task.config, 'phase_specs', []) or []),
            'phase_annotations': _jsonable(self._phase_annotations),
            'phase_boundaries': phase_boundaries,
            **_jsonable(progress_annotations),
            'phase_transition_history': _jsonable(getattr(task, 'phase_transition_history', []) or []),
            'runtime_phase_state': _jsonable(_safe_task_method(task, 'get_phase_runtime_state', {})),
            'robot_roles': _jsonable(robot_roles),
            'execution_order': _jsonable(
                config_metadata.get('execution_order', execution_order)
                if isinstance(config_metadata, dict)
                else execution_order
            ),
            'shared_workspace': _jsonable(
                config_metadata.get(
                    'shared_workspace',
                    {
                        'target_poses': getattr(task.config, 'target_poses', {}),
                        'domain_randomization': getattr(task.config, 'domain_randomization', {}),
                    },
                )
                if isinstance(config_metadata, dict)
                else {}
            ),
            'preconditions': _jsonable(config_metadata.get('preconditions', []) if isinstance(config_metadata, dict) else []),
            'completion_conditions': _jsonable(
                config_metadata.get('completion_conditions', getattr(task.config, 'success_criteria', []))
                if isinstance(config_metadata, dict)
                else getattr(task.config, 'success_criteria', [])
            ),
            'tracked_objects': _jsonable(_safe_task_method(task, 'get_tracked_object_states', {})),
            'waiting_frames': [index for index, value in enumerate(self._waiting_states) if value],
            'handoff_frames': [index for index, value in enumerate(self._handoff_states) if value],
            'task_result': _jsonable(metrics),
        }
        annotation_path = self.episode_dir / 'annotations' / f'episode_{self.episode_idx:06d}.json'
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(annotation_payload, indent=2), encoding='utf-8')

        metadata = {
            'schema_version': 'roboassemblybench_raw_cartesian_v1',
            'episode_idx': self.episode_idx,
            'seed': int(task.config.seed),
            'layout_seed': int(
                task.config.seed if getattr(task.config, 'layout_seed', None) is None else task.config.layout_seed
            ),
            'recipe': str(task.config.recipe),
            'recipe_fingerprint': str(getattr(task.config, 'recipe_fingerprint', '')),
            'task': str(task.config.task_description or task.config.prompt),
            'scene_profile': str(getattr(task.config, 'scene_profile', '') or ''),
            'scene_asset_path': str(getattr(task.config, 'scene_asset_path', '') or ''),
            'scene_asset_fallback_path': str(
                getattr(task.config, 'scene_asset_fallback_path', '') or ''
            ),
            'scene_asset_source': str(getattr(task.config, 'scene_asset_source', '') or ''),
            'scene_family': str(
                getattr(task.config, 'resolved_scene_family', '')
                or (getattr(task.config, 'scene_profile_metadata', {}) or {}).get('scene_family', '')
            ),
            'robot_metadata': _jsonable(getattr(task.config, 'robot_metadata', []) or []),
            'runtime_scene_integrity': {
                'start': _jsonable(self._runtime_scene_integrity_start),
                'end': _jsonable(runtime_scene_integrity(task)),
            },
            'fps': self.fps,
            'simulation_fps': self.simulation_fps,
            'frame_stride': self.frame_stride,
            'timing': {
                'physics_fps': self.simulation_fps,
                'control_fps': self.simulation_fps,
                'dataset_fps': self.fps,
                'dataset_frame_stride': self.frame_stride,
                'rendering_interval': self.rendering_interval,
                'camera_render_period_steps': self.rendering_interval + 1,
                'camera_fps': self.simulation_fps / (self.rendering_interval + 1),
                'camera_state_action_aligned': True,
            },
            'frame_count': len(states),
            'recording_mode': (
                'rendered_replay'
                if self._replay_source_metadata_path is not None
                else ('rendered' if self.record_media else 'trajectory_only')
            ),
            'replay_source_metadata_path': self._replay_source_metadata_path,
            'replay_source_recipe_fingerprint': self._replay_source_recipe_fingerprint,
            'replay_recipe_fingerprint_mismatch': self._replay_recipe_fingerprint_mismatch,
            'replay_recipe_fingerprint_mismatch_allowed': self._allow_replay_recipe_fingerprint_mismatch,
            'trajectory_path': str(trajectory_path),
            'state_names': list(STATE_NAMES),
            'action_names': list(ACTION_NAMES),
            'action_semantics': ACTION_SEMANTICS,
            'videos': {key: str(path) for key, path in self._video_paths.items()},
            'video_encoding': _jsonable(self._video_encoding_metadata),
            'video_shapes': {key: list(shape) for key, shape in self._video_shapes.items()},
            'source_video_shapes': {key: list(shape) for key, shape in self._source_video_shapes.items()},
            'image_preprocessing': {
                'method': 'aspect_ratio_preserving_letterbox',
                'output_width': self.output_resolution[0],
                'output_height': self.output_resolution[1],
                'front_output_width': self.front_output_resolution[0],
                'front_output_height': self.front_output_resolution[1],
                'padding_value': 0,
            },
            'video_frame_counts': dict(self._frame_counts),
            'depth': _jsonable(self._depth_metadata),
            'depth_index_path': str(depth_index_path) if self._depth_metadata else None,
            'extra_features': {
                'joint_state': {'shape': [14], 'dtype': 'float32'},
                'joint_velocity': {'shape': [14], 'dtype': 'float32'},
                'joint_effort': {'shape': [14], 'dtype': 'float32'},
                'wrist_wrench': {'shape': [12], 'dtype': 'float32'},
                'collision_signal': {'shape': [4], 'dtype': 'float32'},
                'subtask_index': {'shape': [1], 'dtype': 'int16'},
                'substage_index': {'shape': [1], 'dtype': 'int16'},
                'waiting_state': {'shape': [1], 'dtype': 'uint8'},
                'handoff_state': {'shape': [1], 'dtype': 'uint8'},
                'replay_joint_state': {'shape': [int(replay_joint_state.shape[1])], 'dtype': 'float32'},
                'tracked_object_position': {
                    'shape': list(tracked_object_position.shape[1:]),
                    'dtype': 'float32',
                },
                'tracked_object_orientation': {
                    'shape': list(tracked_object_orientation.shape[1:]),
                    'dtype': 'float32',
                },
            },
            'capabilities': {
                'rgb': bool(self._video_paths),
                'depth': bool(
                    set(self._depth_metadata) == set(CAMERA_KEYS)
                    and all(item['count'] == len(states) for item in self._depth_metadata.values())
                ),
                'joint_state': bool(self._joint_state_available and all(self._joint_state_available)),
                'joint_velocity': bool(self._joint_velocity_available and all(self._joint_velocity_available)),
                'joint_effort': bool(self._joint_effort_available and any(self._joint_effort_available)),
                'wrist_wrench': bool(self._wrist_wrench_available and any(self._wrist_wrench_available)),
                'collision_signal': bool(self._collision_signals),
                'trajectory_replay': bool(
                    len(replay_joint_state) == len(states)
                    and len(tracked_object_position) == len(states)
                ),
            },
            'collision_signal_names': [
                'collision_detected',
                'left_gripper_contact',
                'right_gripper_contact',
                'locked_object_count',
            ],
            'annotation_path': str(annotation_path),
            'phase_annotations': _jsonable(self._phase_annotations),
            'phase_boundaries': phase_boundaries,
            **_jsonable(progress_annotations),
            'domain_randomization': _jsonable(getattr(task.config, 'domain_randomization', {})),
            'metrics': _jsonable(metrics),
        }
        metadata_path = self.episode_dir / 'metadata.json'
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return {
            'episode_dir': str(self.episode_dir),
            'metadata_path': str(metadata_path),
            'trajectory_path': str(trajectory_path),
            'frame_count': len(states),
            'videos': metadata['videos'],
            'depth': metadata['depth'],
            'annotation_path': str(annotation_path),
            'success': bool(metrics.get('success', False)),
        }
