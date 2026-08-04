from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROBOT_NAMES = ('franka_left', 'franka_right')
CAMERA_KEYS = (
    'observation.images.front',
    'observation.images.left_wrist',
    'observation.images.right_wrist',
)


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


class CartesianObservationEncoder:
    """Encode synchronized task observations into the LeRobot policy schema."""

    def __init__(
        self,
        *,
        task,
        output_resolution: tuple[int, int] = (640, 480),
    ):
        if len(output_resolution) != 2 or any(int(value) <= 0 for value in output_resolution):
            raise ValueError(f'Invalid dataset output resolution: {output_resolution}.')
        self.output_resolution = tuple(int(value) for value in output_resolution)
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

    def letterbox_frame(self, frame: np.ndarray) -> np.ndarray:
        target_width, target_height = self.output_resolution
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
        encoded = {key: self.letterbox_frame(frame) for key, frame in frames.items()}
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
        overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.episode_idx = int(episode_idx)
        self.fps = max(int(fps), 1)
        self.frame_stride = max(int(frame_stride), 1)
        self.simulation_fps = self.fps * self.frame_stride if simulation_fps is None else max(int(simulation_fps), 1)
        self.rendering_interval = self.frame_stride - 1 if rendering_interval is None else int(rendering_interval)
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
            output_resolution=output_resolution,
        )
        self.output_resolution = self._observation_encoder.output_resolution
        self.episode_dir = self.output_dir / f'episode_{self.episode_idx:06d}_cartesian_raw'
        if self.episode_dir.exists() and not overwrite:
            raise FileExistsError(f'Raw Cartesian episode already exists: {self.episode_dir}')
        self.video_dir = self.episode_dir / 'videos'
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self._camera_bindings = self._observation_encoder.camera_bindings
        self._writers: dict[str, cv2.VideoWriter] = {}
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
        self._finalized = False

    @staticmethod
    def _resolve_camera_bindings(task) -> dict[str, tuple[str, str]]:
        return CartesianObservationEncoder.resolve_camera_bindings(task)

    @staticmethod
    def _rgb_frame(obs: dict[str, Any], binding: tuple[str, str]) -> np.ndarray | None:
        return CartesianObservationEncoder.rgb_frame(obs, binding)

    def _letterbox_frame(self, frame: np.ndarray) -> np.ndarray:
        return self._observation_encoder.letterbox_frame(frame)

    def _state(self, task, obs: dict[str, Any]) -> np.ndarray:
        return self._observation_encoder.state(task, obs)

    def _open_writer(self, camera_key: str, frame: np.ndarray) -> cv2.VideoWriter:
        height, width = frame.shape[:2]
        path = self.video_dir / f'{camera_key.replace(".", "_")}.mp4'
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            float(self.fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f'Failed to open dataset video writer: {path}')
        self._writers[camera_key] = writer
        self._video_paths[camera_key] = path
        self._video_shapes[camera_key] = (int(height), int(width), 3)
        return writer

    def record(self, *, task, obs: dict[str, Any], actions: dict[str, Any]) -> bool:
        if int(task.step_counter) % self.frame_stride != 0:
            return False

        frames = {key: self._rgb_frame(obs, self._camera_bindings[key]) for key in CAMERA_KEYS}
        if not all(frame is not None for frame in frames.values()):
            return False
        for camera_key, frame in frames.items():
            self._source_video_shapes.setdefault(camera_key, tuple(int(value) for value in frame.shape))
            frames[camera_key] = self._letterbox_frame(frame)

        state = self._state(task, obs)
        gripper_commands = np.asarray(
            [
                _gripper_command(actions, robot_name, state[index * 8 + 7])
                for index, robot_name in enumerate(ROBOT_NAMES)
            ],
            dtype=np.float32,
        )
        for camera_key, frame in frames.items():
            writer = self._writers.get(camera_key) or self._open_writer(camera_key, frame)
            expected_shape = self._video_shapes[camera_key]
            if tuple(frame.shape) != expected_shape:
                raise ValueError(f'Camera {camera_key} changed shape from {expected_shape} to {tuple(frame.shape)}.')
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._frame_counts[camera_key] += 1

        self._states.append(state)
        self._gripper_commands.append(gripper_commands)
        self._simulation_steps.append(int(task.step_counter))
        self._phase_indices.append(int(getattr(task, 'phase_index', -1)))
        self._phase_steps.append(int(getattr(task, 'phase_step_counter', -1)))
        self._phase_names.append(str(getattr(task, 'phase', '')))
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
        for writer in self._writers.values():
            writer.release()

        frame_counts = set(self._frame_counts.values())
        if not self._states or frame_counts != {len(self._states)}:
            raise RuntimeError(
                f'Unaligned raw episode {self.episode_idx}: states={len(self._states)}, '
                f'camera_frames={self._frame_counts}.'
            )
        states = np.asarray(self._states, dtype=np.float32)
        gripper_commands = np.asarray(self._gripper_commands, dtype=np.float32)
        actions = self._next_sample_actions(states, gripper_commands)
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
        )

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
            'trajectory_path': str(trajectory_path),
            'state_names': list(STATE_NAMES),
            'action_names': list(ACTION_NAMES),
            'action_semantics': ACTION_SEMANTICS,
            'videos': {key: str(path) for key, path in self._video_paths.items()},
            'video_shapes': {key: list(shape) for key, shape in self._video_shapes.items()},
            'source_video_shapes': {key: list(shape) for key, shape in self._source_video_shapes.items()},
            'image_preprocessing': {
                'method': 'aspect_ratio_preserving_letterbox',
                'output_width': self.output_resolution[0],
                'output_height': self.output_resolution[1],
                'padding_value': 0,
            },
            'video_frame_counts': dict(self._frame_counts),
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
            'success': bool(metrics.get('success', False)),
        }
