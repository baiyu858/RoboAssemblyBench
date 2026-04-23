from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import math
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from toolkits.factory_dual_franka_assembly.convert_dataset import load_episode_payloads
from toolkits.factory_dual_franka_assembly.planner_primitives import (
    euler_xyz_intrinsic_to_quat,
    quat_multiply,
    quat_rotate,
)


DEFAULT_VIDEO_MODE = 'topdown'
TOPDOWN_VIDEO_KEY = 'observation.images.topdown'
ISAAC_REPLAY_VIDEO_KEY = 'observation.images.isaac_3d'
FRONT_VIDEO_KEY = 'observation.images.front'
LEFT_WRIST_VIDEO_KEY = 'observation.images.left_wrist'
RIGHT_WRIST_VIDEO_KEY = 'observation.images.right_wrist'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly_lerobot'
ISAAC_REPLAY_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly_lerobot_isaac3d'
LIVE_ROLLOUT_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly_lerobot_live'

LEFT_COLOR = (219, 89, 89)
RIGHT_COLOR = (66, 127, 219)
OBJECT_COLORS = (
    (217, 127, 66),
    (120, 166, 85),
    (176, 122, 212),
    (86, 164, 182),
    (199, 92, 140),
    (205, 164, 84),
)


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _jsonl_dump(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def _default_video_key(video_mode: str) -> str:
    return FRONT_VIDEO_KEY if video_mode in {'isaac_replay', 'live_rollout'} else TOPDOWN_VIDEO_KEY


def _default_output_dir(video_mode: str) -> Path:
    if video_mode == 'isaac_replay':
        return ISAAC_REPLAY_OUTPUT_DIR
    if video_mode == 'live_rollout':
        return LIVE_ROLLOUT_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def _normalize_camera_spec(camera_spec: dict | None, *, default_width: int, default_height: int) -> dict:
    spec = dict(camera_spec or {})
    name = str(spec.get('name') or '').strip()
    view_type = str(spec.get('view_type') or '').strip().lower()

    owner_or_robot = str(spec.get('robot') or spec.get('owner') or '').lower()
    if not view_type:
        if 'wrist' in name or owner_or_robot in {'franka_left', 'franka_right'}:
            view_type = 'wrist'
        else:
            view_type = 'front'
    spec['view_type'] = view_type

    if view_type == 'wrist':
        if owner_or_robot not in {'franka_left', 'franka_right'}:
            owner_or_robot = 'franka_left'
        spec['robot'] = owner_or_robot
        spec['owner'] = owner_or_robot
        default_name = 'left_wrist' if owner_or_robot == 'franka_left' else 'right_wrist'
        spec.setdefault('name', default_name)
        default_video_key = LEFT_WRIST_VIDEO_KEY if owner_or_robot == 'franka_left' else RIGHT_WRIST_VIDEO_KEY
    else:
        spec.setdefault('name', 'front')
        default_video_key = FRONT_VIDEO_KEY

    video_key = spec.get('video_key')
    if video_key in {ISAAC_REPLAY_VIDEO_KEY, 'observation.images.third_person'}:
        video_key = FRONT_VIDEO_KEY
    spec['video_key'] = video_key or default_video_key

    resolution = spec.get('resolution') or [default_width, default_height]
    if isinstance(resolution, tuple):
        resolution = list(resolution)
    spec['resolution'] = [max(int(resolution[0]), 64), max(int(resolution[1]), 64)]
    return spec


def _instruction_for_episode(episode: dict) -> str:
    return (
        episode.get('task_description')
        or episode.get('annotation_description')
        or episode.get('prompt')
        or episode.get('recipe')
        or 'dual franka assembly'
    )


def _episode_camera_specs(
    episode: dict,
    *,
    video_mode: str,
    default_width: int,
    default_height: int,
    video_key: str | None = None,
) -> list[dict]:
    if video_mode not in {'isaac_replay', 'live_rollout'}:
        return [
            {
                'name': 'topdown',
                'view_type': 'topdown',
                'video_key': video_key or TOPDOWN_VIDEO_KEY,
                'resolution': [default_width, default_height],
            }
        ]

    raw_specs = [item for item in (episode.get('camera_metadata') or []) if isinstance(item, dict)]
    normalized = [
        _normalize_camera_spec(item, default_width=default_width, default_height=default_height)
        for item in raw_specs
    ]

    defaults = [
        _normalize_camera_spec(
            {'name': 'front', 'view_type': 'front', 'video_key': FRONT_VIDEO_KEY},
            default_width=default_width,
            default_height=default_height,
        ),
        _normalize_camera_spec(
            {'name': 'left_wrist', 'view_type': 'wrist', 'robot': 'franka_left', 'video_key': LEFT_WRIST_VIDEO_KEY},
            default_width=default_width,
            default_height=default_height,
        ),
        _normalize_camera_spec(
            {'name': 'right_wrist', 'view_type': 'wrist', 'robot': 'franka_right', 'video_key': RIGHT_WRIST_VIDEO_KEY},
            default_width=default_width,
            default_height=default_height,
        ),
    ]

    specs_by_key = {spec['video_key']: spec for spec in normalized}
    for default_spec in defaults:
        specs_by_key.setdefault(default_spec['video_key'], default_spec)

    ordered_keys = [FRONT_VIDEO_KEY, LEFT_WRIST_VIDEO_KEY, RIGHT_WRIST_VIDEO_KEY]
    return [specs_by_key[key] for key in ordered_keys]


def _extract_joint_positions(robot_obs: dict, expected_count: int = 9) -> list[float]:
    joint_values: dict[int, float] = {}
    max_index = expected_count - 1
    for action in robot_obs.get('joint_action') or []:
        positions = action.get('joint_positions') or []
        joint_indices = action.get('joint_indices')
        if joint_indices is not None:
            for index, value in zip(joint_indices, positions):
                if value is None:
                    continue
                joint_values[int(index)] = float(value)
                max_index = max(max_index, int(index))
            continue

        for index, value in enumerate(positions):
            if value is None:
                continue
            joint_values[index] = float(value)
            max_index = max(max_index, index)

    return [float(joint_values.get(index, 0.0)) for index in range(max_index + 1)]


def _gripper_command_to_float(gripper_command) -> float:
    if isinstance(gripper_command, (int, float)):
        return float(gripper_command)
    if isinstance(gripper_command, str):
        lowered = gripper_command.lower()
        if lowered == 'open':
            return 1.0
        if lowered == 'close':
            return 0.0
    return 0.0


def _flatten_robot_state(step: dict, robot_name: str) -> list[float]:
    robot_obs = step['observations'][robot_name]
    eef_position = [float(value) for value in robot_obs.get('eef_position') or robot_obs.get('position') or [0.0, 0.0, 0.0]]
    eef_orientation = [
        float(value) for value in robot_obs.get('eef_orientation') or robot_obs.get('orientation') or [1.0, 0.0, 0.0, 0.0]
    ]
    joint_positions = _extract_joint_positions(robot_obs)
    return eef_position + eef_orientation + joint_positions


def _flatten_robot_action(step: dict, robot_name: str) -> list[float]:
    robot_action = step['actions'].get(robot_name, {})
    arm_action = robot_action.get('arm_ik_controller') or [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    target_position = [float(value) for value in (arm_action[0] if len(arm_action) > 0 else [0.0, 0.0, 0.0])]
    target_orientation = [float(value) for value in (arm_action[1] if len(arm_action) > 1 else [1.0, 0.0, 0.0, 0.0])]
    gripper_raw = robot_action.get('gripper_controller') or [0.0]
    gripper_open = _gripper_command_to_float(gripper_raw[0] if isinstance(gripper_raw, list) and gripper_raw else gripper_raw)
    return target_position + target_orientation + [gripper_open]


def _object_union(episodes: list[dict]) -> list[str]:
    object_names = OrderedDict()
    for episode in episodes:
        for step in episode.get('steps', []):
            for object_name in step.get('objects', {}):
                object_names.setdefault(object_name, None)
    return list(object_names.keys())


def _flatten_environment_state(step: dict, object_names: list[str]) -> tuple[list[float], list[float]]:
    env_state = []
    env_presence = []
    objects = step.get('objects', {})
    for object_name in object_names:
        object_state = objects.get(object_name)
        if object_state is None:
            env_state.extend([0.0] * 7)
            env_presence.append(0.0)
            continue
        env_state.extend(float(value) for value in object_state.get('position', [0.0, 0.0, 0.0]))
        env_state.extend(float(value) for value in object_state.get('orientation', [1.0, 0.0, 0.0, 0.0]))
        env_presence.append(1.0)
    return env_state, env_presence


def _phase_index_lookup(episode: dict) -> dict[str, int]:
    phase_names = episode.get('phase_blueprint') or []
    return {phase_name: index for index, phase_name in enumerate(phase_names)}


def _task_index_lookup(episodes: list[dict]) -> tuple[dict[str, int], list[dict]]:
    task_to_index: OrderedDict[str, int] = OrderedDict()
    for episode in episodes:
        task = _instruction_for_episode(episode)
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
    tasks = [{'task_index': index, 'task': task} for task, index in task_to_index.items()]
    return dict(task_to_index), tasks


def _state_feature_names() -> list[str]:
    joint_names = [f'joint_{index}' for index in range(9)]
    names = []
    for robot_prefix in ('left', 'right'):
        names.extend(
            [
                f'{robot_prefix}_eef_pos_x',
                f'{robot_prefix}_eef_pos_y',
                f'{robot_prefix}_eef_pos_z',
                f'{robot_prefix}_eef_quat_w',
                f'{robot_prefix}_eef_quat_x',
                f'{robot_prefix}_eef_quat_y',
                f'{robot_prefix}_eef_quat_z',
            ]
        )
        names.extend(f'{robot_prefix}_{joint_name}' for joint_name in joint_names)
    return names


def _action_feature_names() -> list[str]:
    names = []
    for robot_prefix in ('left', 'right'):
        names.extend(
            [
                f'{robot_prefix}_target_pos_x',
                f'{robot_prefix}_target_pos_y',
                f'{robot_prefix}_target_pos_z',
                f'{robot_prefix}_target_quat_w',
                f'{robot_prefix}_target_quat_x',
                f'{robot_prefix}_target_quat_y',
                f'{robot_prefix}_target_quat_z',
                f'{robot_prefix}_gripper_open',
            ]
        )
    return names


def _environment_feature_names(object_names: list[str]) -> tuple[list[str], list[str]]:
    state_names = []
    for object_name in object_names:
        state_names.extend(
            [
                f'{object_name}_pos_x',
                f'{object_name}_pos_y',
                f'{object_name}_pos_z',
                f'{object_name}_quat_w',
                f'{object_name}_quat_x',
                f'{object_name}_quat_y',
                f'{object_name}_quat_z',
            ]
        )
    return state_names, list(object_names)


def _vector_stats(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return {
        'min': matrix.min(axis=0).astype(float).tolist(),
        'max': matrix.max(axis=0).astype(float).tolist(),
        'mean': matrix.mean(axis=0).astype(float).tolist(),
        'std': matrix.std(axis=0).astype(float).tolist(),
        'count': [int(matrix.shape[0])],
    }


def _scalar_stats(values: list[int | float]) -> dict:
    array = np.asarray(values, dtype=np.float32)
    return {
        'min': [float(array.min())],
        'max': [float(array.max())],
        'mean': [float(array.mean())],
        'std': [float(array.std())],
        'count': [int(array.shape[0])],
    }


class RunningImageStats:
    def __init__(self):
        self.channel_min = np.full(3, np.inf, dtype=np.float64)
        self.channel_max = np.full(3, -np.inf, dtype=np.float64)
        self.channel_sum = np.zeros(3, dtype=np.float64)
        self.channel_sumsq = np.zeros(3, dtype=np.float64)
        self.pixel_count = 0
        self.frame_count = 0

    def update(self, frame_rgb: np.ndarray):
        frame = frame_rgb.astype(np.float32) / 255.0
        flat = frame.reshape(-1, 3)
        self.channel_min = np.minimum(self.channel_min, flat.min(axis=0))
        self.channel_max = np.maximum(self.channel_max, flat.max(axis=0))
        self.channel_sum += flat.sum(axis=0)
        self.channel_sumsq += np.square(flat).sum(axis=0)
        self.pixel_count += int(flat.shape[0])
        self.frame_count += 1

    @staticmethod
    def _format_rgb_triplet(values: np.ndarray) -> list[list[list[float]]]:
        return [[[float(values[0])]], [[float(values[1])]], [[float(values[2])]]]

    def to_dict(self) -> dict:
        denominator = max(self.pixel_count, 1)
        mean = self.channel_sum / denominator
        variance = np.maximum(self.channel_sumsq / denominator - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            'min': self._format_rgb_triplet(self.channel_min),
            'max': self._format_rgb_triplet(self.channel_max),
            'mean': self._format_rgb_triplet(mean),
            'std': self._format_rgb_triplet(std),
            'count': [int(self.frame_count)],
        }


def _collect_xy_bounds(episode: dict) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for step in episode.get('steps', []):
        for robot_name in ('franka_left', 'franka_right'):
            robot_obs = step.get('observations', {}).get(robot_name, {})
            position = robot_obs.get('eef_position') or robot_obs.get('position')
            if position is not None:
                xs.append(float(position[0]))
                ys.append(float(position[1]))
        for object_state in step.get('objects', {}).values():
            position = object_state.get('position')
            if position is not None:
                xs.append(float(position[0]))
                ys.append(float(position[1]))

    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)

    margin = 0.18
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def _world_to_px(x: float, y: float, bounds: tuple[float, float, float, float], width: int, height: int, pad: int = 48):
    min_x, max_x, min_y, max_y = bounds
    draw_width = max(width - 2 * pad, 1)
    draw_height = max(height - 2 * pad, 1)
    px = pad + (x - min_x) / max(max_x - min_x, 1e-6) * draw_width
    py = height - pad - (y - min_y) / max(max_y - min_y, 1e-6) * draw_height
    return px, py


def _draw_path(draw: ImageDraw.ImageDraw, trace: list[tuple[float, float]], bounds, width: int, height: int, color):
    if len(trace) < 2:
        return
    points = [_world_to_px(x, y, bounds, width, height) for x, y in trace]
    draw.line(points, fill=color, width=4)


def _draw_robot(draw: ImageDraw.ImageDraw, position, bounds, width: int, height: int, color, label: str):
    px, py = _world_to_px(position[0], position[1], bounds, width, height)
    radius = 14
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline=(35, 35, 35), width=2)
    draw.text((px + 18, py - 10), label, fill=(40, 40, 40))


def _draw_object(draw: ImageDraw.ImageDraw, position, bounds, width: int, height: int, color, label: str):
    px, py = _world_to_px(position[0], position[1], bounds, width, height)
    extent = 14
    draw.rounded_rectangle((px - extent, py - extent, px + extent, py + extent), radius=5, fill=color, outline=(35, 35, 35), width=2)
    draw.text((px + 18, py - 10), label, fill=(40, 40, 40))


def render_episode_video(episode: dict, output_path: Path, fps: int, width: int, height: int) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = _collect_xy_bounds(episode)
    font = ImageFont.load_default()
    steps = episode.get('steps', [])
    if not steps:
        raise RuntimeError(f'Episode {episode.get("recipe")} has no steps to render.')

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open video writer for {output_path}')

    image_stats = RunningImageStats()
    left_trace = []
    right_trace = []
    object_traces = {object_name: [] for object_name in _object_union([episode])}

    try:
        for frame_index, step in enumerate(steps):
            left_position = step['observations']['franka_left'].get('eef_position') or step['observations']['franka_left'].get('position')
            right_position = step['observations']['franka_right'].get('eef_position') or step['observations']['franka_right'].get('position')
            left_xy = (float(left_position[0]), float(left_position[1]))
            right_xy = (float(right_position[0]), float(right_position[1]))
            left_trace.append(left_xy)
            right_trace.append(right_xy)

            for object_name, object_state in step.get('objects', {}).items():
                position = object_state.get('position') or [0.0, 0.0, 0.0]
                object_traces.setdefault(object_name, []).append((float(position[0]), float(position[1])))

            image = Image.new('RGB', (width, height), (246, 243, 236))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=26, outline=(208, 198, 186), width=2)
            draw.text((34, 28), f"{episode.get('recipe', 'assembly')}  frame={frame_index}/{len(steps) - 1}", fill=(32, 32, 32), font=font)
            draw.text((34, 52), f"phase={step.get('phase', 'unknown')}  seed={episode.get('seed', 0)}", fill=(58, 58, 58), font=font)
            draw.text((34, 76), _instruction_for_episode(episode), fill=(74, 74, 74), font=font)

            _draw_path(draw, left_trace, bounds, width, height, LEFT_COLOR)
            _draw_path(draw, right_trace, bounds, width, height, RIGHT_COLOR)

            for object_index, object_name in enumerate(sorted(step.get('objects', {}))):
                color = OBJECT_COLORS[object_index % len(OBJECT_COLORS)]
                _draw_path(draw, object_traces.get(object_name, []), bounds, width, height, color)

            _draw_robot(draw, left_xy, bounds, width, height, LEFT_COLOR, 'left eef')
            _draw_robot(draw, right_xy, bounds, width, height, RIGHT_COLOR, 'right eef')

            for object_index, object_name in enumerate(sorted(step.get('objects', {}))):
                color = OBJECT_COLORS[object_index % len(OBJECT_COLORS)]
                position = step['objects'][object_name].get('position') or [0.0, 0.0, 0.0]
                _draw_object(draw, position, bounds, width, height, color, object_name)

            frame_rgb = np.asarray(image)
            image_stats.update(frame_rgb)
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return {
        'image_stats': image_stats.to_dict(),
        'video_path': str(output_path),
        'width': width,
        'height': height,
        'fps': fps,
        'renderer': 'topdown',
    }


def _camera_focus_stats(episode: dict) -> tuple[float, float, float, float, float, float]:
    xs = []
    ys = []
    zs = []
    for robot in episode.get('robot_metadata', []):
        position = robot.get('position')
        if position is not None:
            xs.append(float(position[0]))
            ys.append(float(position[1]))
            zs.append(float(position[2]))
    for step in episode.get('steps', []):
        for robot_name in ('franka_left', 'franka_right'):
            robot_obs = step.get('observations', {}).get(robot_name, {})
            position = robot_obs.get('eef_position') or robot_obs.get('position')
            if position is not None:
                xs.append(float(position[0]))
                ys.append(float(position[1]))
                zs.append(float(position[2]))
        for object_state in step.get('objects', {}).values():
            position = object_state.get('position')
            if position is not None:
                xs.append(float(position[0]))
                ys.append(float(position[1]))
                zs.append(float(position[2]))

    if not xs or not ys or not zs:
        return (0.35, 0.0, 0.0, 1.0, 1.0, 0.5)

    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _isaac_camera_pose(episode: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    min_x, max_x, min_y, max_y, min_z, max_z = _camera_focus_stats(episode)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span_x = max(max_x - min_x, 0.6)
    span_y = max(max_y - min_y, 1.2)
    span_z = max(max_z - min_z, 0.4)

    camera_position = (
        center_x - max(0.7, span_x * 0.55),
        center_y - max(2.25, span_y * 1.15),
        max(1.45, 0.9 + span_z * 1.4),
    )
    look_at = (
        center_x + min(0.18, span_x * 0.1),
        center_y,
        max(0.28, min_z + span_z * 0.55),
    )
    return camera_position, look_at


def _front_camera_pose(episode: dict, camera_spec: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if camera_spec.get('position') is not None and camera_spec.get('look_at') is not None:
        camera_position = tuple(float(value) for value in camera_spec['position'])
        look_at = tuple(float(value) for value in camera_spec['look_at'])
        return camera_position, look_at

    if camera_spec.get('translation') is not None and (
        camera_spec.get('orientation') is not None or camera_spec.get('orientation_euler') is not None
    ):
        camera_position = np.asarray(camera_spec['translation'], dtype=float)
        if camera_spec.get('orientation') is not None:
            camera_orientation = np.asarray(camera_spec['orientation'], dtype=float)
        else:
            camera_orientation = euler_xyz_intrinsic_to_quat(
                camera_spec.get('orientation_euler', [0.0, 0.0, 0.0])
            )
        look_at = camera_position + quat_rotate(camera_orientation, [0.0, 0.0, -1.0])
        return (
            tuple(float(value) for value in camera_position),
            tuple(float(value) for value in look_at),
        )

    min_x, max_x, min_y, max_y, min_z, max_z = _camera_focus_stats(episode)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span_x = max(max_x - min_x, 0.7)
    span_y = max(max_y - min_y, 1.1)
    span_z = max(max_z - min_z, 0.35)

    if camera_spec.get('view_type') != 'front':
        return _isaac_camera_pose(episode)

    camera_position = (
        max_x + max(0.75, span_x * 0.95),
        center_y,
        max(1.05, min_z + span_z * 1.9),
    )
    look_at = (
        center_x,
        center_y,
        max(0.26, min_z + span_z * 0.55),
    )
    return camera_position, look_at


def _wrist_camera_pose(step: dict, camera_spec: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    robot_name = camera_spec['robot']
    robot_obs = step.get('observations', {}).get(robot_name, {})
    eef_position = np.asarray(robot_obs.get('eef_position') or robot_obs.get('position') or [0.0, 0.0, 0.0], dtype=float)
    eef_orientation = np.asarray(
        robot_obs.get('eef_orientation') or robot_obs.get('orientation') or [1.0, 0.0, 0.0, 0.0],
        dtype=float,
    )

    mount_offset = np.asarray(
        camera_spec.get('translation', camera_spec.get('mount_offset', [-0.06, 0.0, 0.045])),
        dtype=float,
    )
    eye = eef_position + quat_rotate(eef_orientation, mount_offset)

    camera_local_orientation = None
    if camera_spec.get('orientation') is not None:
        camera_local_orientation = np.asarray(camera_spec['orientation'], dtype=float)
    elif camera_spec.get('orientation_euler') is not None:
        camera_local_orientation = euler_xyz_intrinsic_to_quat(
            camera_spec.get('orientation_euler', [0.0, 0.0, 0.0])
        )

    target = None
    if camera_local_orientation is not None:
        camera_world_orientation = quat_multiply(eef_orientation, camera_local_orientation)
        target = eye + quat_rotate(camera_world_orientation, [0.0, 0.0, -0.18])
    elif camera_spec.get('target_source', 'task_target') == 'task_target':
        task_target = robot_obs.get('task_target')
        if task_target is not None:
            target = np.asarray(task_target, dtype=float)

    if target is None:
        look_offset = np.asarray(camera_spec.get('look_offset', [0.18, 0.0, -0.02]), dtype=float)
        target = eef_position + quat_rotate(eef_orientation, look_offset)

    if np.linalg.norm(target - eye) < 0.05:
        target = eye + np.asarray([0.15, 0.0, -0.01], dtype=float)

    return tuple(float(value) for value in eye), tuple(float(value) for value in target)


def _camera_pose_for_frame(episode: dict, step: dict, camera_spec: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    view_type = str(camera_spec.get('view_type', 'front')).lower()
    if view_type == 'wrist':
        return _wrist_camera_pose(step=step, camera_spec=camera_spec)
    return _front_camera_pose(episode=episode, camera_spec=camera_spec)


def _collect_png_paths(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.rglob('*.png') if path.is_file())


def _encode_mp4_from_pngs(frames_dir: Path, output_path: Path, fps: int) -> dict:
    png_paths = _collect_png_paths(frames_dir)
    if not png_paths:
        raise RuntimeError(f'No PNG frames were written to {frames_dir}')

    first_frame = cv2.imread(str(png_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f'Failed to read the first PNG frame at {png_paths[0]}')
    height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open MP4 writer for {output_path}')

    image_stats = RunningImageStats()
    try:
        for png_path in png_paths:
            frame_bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f'Failed to read frame {png_path}')
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_stats.update(frame_rgb)
            writer.write(frame_bgr)
    finally:
        writer.release()

    return {
        'png_count': len(png_paths),
        'image_stats': image_stats.to_dict(),
        'width': width,
        'height': height,
        'fps': fps,
    }


def _isaac_render_summary_path(output_path: Path) -> Path:
    return output_path.with_suffix('.render_summary.json')


def _isaac_render_log_path(output_path: Path) -> Path:
    return output_path.with_suffix('.render.log')


def _set_robot_joint_state(task, step: dict, robot_name: str):
    robot = task.robots[robot_name]
    joint_positions = _extract_joint_positions(step['observations'][robot_name])
    joint_array = np.asarray(joint_positions, dtype=float)
    robot.articulation.set_joint_positions(joint_array)
    robot.articulation.set_joint_velocities(np.zeros_like(joint_array))


def _set_object_state(task, object_name: str, object_state: dict):
    rigid_body = task._resolve_object(object_name)  # noqa: SLF001 - replay helper needs the tracked object handle
    rigid_body.set_linear_velocity(np.zeros(3))
    rigid_body.set_pose(
        np.asarray(object_state.get('position', [0.0, 0.0, 0.0]), dtype=float),
        np.asarray(object_state.get('orientation', [1.0, 0.0, 0.0, 0.0]), dtype=float),
    )


def _capture_viewport_rgba(viewport_api, app, *, max_updates: int = 120) -> np.ndarray:
    from omni.kit.viewport.utility import capture_viewport_to_buffer

    captured = {}

    def on_capture(buffer, buffer_size, width, height, fmt):
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_byte * buffer_size)
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
        frame = np.frombuffer(bytes(content.contents), dtype=np.uint8).reshape((height, width, 4)).copy()
        captured['rgba'] = frame
        captured['format'] = str(fmt)

    future = capture_viewport_to_buffer(viewport_api, on_capture)

    async def wait_capture():
        await future.wait_for_result()

    asyncio.ensure_future(wait_capture())
    for _ in range(max_updates):
        app.update()
        if 'rgba' in captured:
            return captured['rgba']

    raise RuntimeError('Viewport capture did not complete before timeout.')


def render_episode_video_isaac_replay(
    episode: dict,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    keep_frames: bool = False,
    summary_path: Path | None = None,
    camera_video_key: str | None = None,
) -> dict:
    from toolkits.factory_dual_franka_assembly.generate_demos import _build_env
    from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_path.with_suffix('')
    frames_dir = frames_dir.parent / f'{frames_dir.name}_frames'
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    if keep_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = build_dual_franka_assembly_episode(
        recipe=episode.get('recipe') or episode.get('spec_path') or 'factory_dual_franka_assembly',
        seed=int(episode.get('seed', 0)),
        episode_idx=int(episode.get('episode_idx', 0)),
        spec_path=episode.get('spec_path'),
        scene_profile=episode.get('scene_profile'),
    )
    env = _build_env([task_cfg], headless=False)
    env.runner.render_interval = 0

    summary = None
    writer = None
    try:
        env.reset()
        import omni.kit.app
        from omni.kit.viewport.utility import get_active_viewport
        from isaacsim.core.utils.viewports import set_camera_view

        app = omni.kit.app.get_app()
        camera_specs = _episode_camera_specs(
            episode,
            video_mode='isaac_replay',
            default_width=width,
            default_height=height,
            video_key=camera_video_key,
        )
        selected_camera_spec = None
        if camera_video_key is not None:
            for camera_spec in camera_specs:
                if camera_spec.get('video_key') == camera_video_key:
                    selected_camera_spec = camera_spec
                    break
            if selected_camera_spec is None:
                selected_camera_spec = {
                    'name': camera_video_key.split('.')[-1],
                    'view_type': 'front',
                    'video_key': camera_video_key,
                }
        if selected_camera_spec is None:
            selected_camera_spec = {
                'name': 'third_person_front',
                'view_type': 'front',
                'video_key': camera_video_key or ISAAC_REPLAY_VIDEO_KEY,
            }

        selected_camera_spec = _normalize_camera_spec(
            selected_camera_spec,
            default_width=width,
            default_height=height,
        )
        width, height = selected_camera_spec['resolution']
        camera_position, look_at = _front_camera_pose(episode=episode, camera_spec=selected_camera_spec)
        set_camera_view(
            eye=np.asarray(camera_position, dtype=float),
            target=np.asarray(look_at, dtype=float),
            camera_prim_path='/OmniverseKit_Persp',
        )
        viewport = get_active_viewport()
        viewport.set_texture_resolution((int(width), int(height)))

        # Let the freshly reset scene and the new camera pose settle before the
        # first capture; otherwise Isaac often returns the empty viewport grid.
        for _ in range(180):
            app.update()

        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]
        image_stats = RunningImageStats()
        written_png_count = 0
        camera_label = selected_camera_spec.get('video_key') or selected_camera_spec.get('name')

        for frame_index, step in enumerate(episode.get('steps', [])):
            _set_robot_joint_state(task, step, 'franka_left')
            _set_robot_joint_state(task, step, 'franka_right')
            for object_name, object_state in step.get('objects', {}).items():
                _set_object_state(task, object_name, object_state)

            camera_position, look_at = _camera_pose_for_frame(
                episode=episode,
                step=step,
                camera_spec=selected_camera_spec,
            )
            set_camera_view(
                eye=np.asarray(camera_position, dtype=float),
                target=np.asarray(look_at, dtype=float),
                camera_prim_path='/OmniverseKit_Persp',
            )

            for _ in range(2):
                app.update()

            frame_rgba = _capture_viewport_rgba(viewport, app, max_updates=90)
            frame_rgb = np.asarray(frame_rgba)[..., :3]
            image_stats.update(frame_rgb)

            if writer is None:
                frame_height, frame_width = frame_rgb.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f'Failed to open MP4 writer for {output_path}')

            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if keep_frames:
                frame_dir = frames_dir / camera_label
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f'frame_{frame_index:05d}.png'
                Image.fromarray(frame_rgb).save(frame_path)
                written_png_count += 1

        if writer is None:
            raise RuntimeError(f'Episode {episode.get("recipe")} has no steps to render.')

        summary = {
            'image_stats': image_stats.to_dict(),
            'video_path': str(output_path),
            'width': int(frame_width),
            'height': int(frame_height),
            'fps': int(fps),
            'renderer': 'isaac_replay_camera',
            'frames_dir': str(frames_dir) if keep_frames else None,
            'written_png_count': written_png_count,
            'camera_name': selected_camera_spec.get('name'),
            'camera_video_key': camera_label,
            'camera_view_type': selected_camera_spec.get('view_type', 'front'),
            'camera_position': list(camera_position),
            'camera_look_at': list(look_at),
        }
        if summary_path is not None:
            _json_dump(summary_path, summary)
    finally:
        if writer is not None:
            writer.release()
        env.close()
        if frames_dir.exists() and not keep_frames:
            shutil.rmtree(frames_dir)

    return summary


def _render_episode_video_isaac_subprocess(
    *,
    episode: dict,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    keep_video_frames: bool,
    camera_video_key: str | None = None,
) -> dict:
    episode_path = episode.get('source_path')
    if not episode_path:
        raise RuntimeError('Isaac replay rendering requires each episode to include a source_path.')

    summary_path = _isaac_render_summary_path(output_path)
    log_path = _isaac_render_log_path(output_path)
    if summary_path.exists():
        summary_path.unlink()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--render-single-episode-isaac',
        '--episode-path',
        str(Path(episode_path).resolve()),
        '--output',
        str(output_path.resolve()),
        '--summary-path',
        str(summary_path.resolve()),
        '--fps',
        str(int(fps)),
        '--video-width',
        str(int(width)),
        '--video-height',
        str(int(height)),
        '--/apps/extensions/fsWatcherEnabled=0',
    ]
    if camera_video_key is not None:
        command.extend(['--camera-video-key', camera_video_key])
    if keep_video_frames:
        command.append('--keep-video-frames')

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f'Isaac replay render failed for {episode_path}. See log at {log_path}.'
        )
    if not summary_path.exists():
        raise RuntimeError(
            f'Isaac replay render did not produce its summary file for {episode_path}. See log at {log_path}.'
        )
    return json.loads(summary_path.read_text(encoding='utf-8'))


def _copy_live_rollout_video(
    *,
    episode: dict,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    camera_video_key: str | None = None,
) -> dict:
    recorded_videos = episode.get('recorded_videos') or {}
    recorded_summaries = episode.get('recorded_video_summaries') or {}
    if camera_video_key is not None:
        desired_key = camera_video_key
    elif recorded_videos:
        desired_key = next(iter(recorded_videos.keys()))
    else:
        desired_key = FRONT_VIDEO_KEY
    if recorded_videos:
        source_path = recorded_videos.get(desired_key)
        if source_path is None and desired_key == ISAAC_REPLAY_VIDEO_KEY:
            source_path = recorded_videos.get(FRONT_VIDEO_KEY)
            desired_key = FRONT_VIDEO_KEY if source_path is not None else desired_key
        if source_path is None:
            raise RuntimeError(
                f"Episode {episode.get('source_path') or episode.get('recipe')} has no recorded video for {desired_key!r}."
            )

        source_video_path = Path(source_path).resolve()
        if not source_video_path.exists():
            raise RuntimeError(f'Live rollout video does not exist: {source_video_path}')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_video_path, output_path)

        summary = dict(recorded_summaries.get(desired_key) or {})
        if not summary:
            capture = cv2.VideoCapture(str(output_path))
            try:
                frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                frame_fps = int(capture.get(cv2.CAP_PROP_FPS) or fps)
            finally:
                capture.release()
            summary = {
                'video_path': str(output_path),
                'width': frame_width,
                'height': frame_height,
                'fps': frame_fps,
                'renderer': 'live_rollout_viewport',
                'image_stats': {},
            }
        else:
            summary['video_path'] = str(output_path)
            summary.setdefault('fps', int(fps))
            summary.setdefault('width', int(width))
            summary.setdefault('height', int(height))
            summary.setdefault('renderer', 'live_rollout_viewport')
            summary.setdefault('image_stats', {})
        return summary

    recorded_frames = episode.get('recorded_frames') or {}
    recorded_frame_summaries = episode.get('recorded_frame_summaries') or {}
    if not recorded_frames:
        raise RuntimeError(
            f"Episode {episode.get('source_path') or episode.get('recipe')} does not contain live rollout frames."
        )

    source_frames = recorded_frames.get(desired_key)
    if source_frames is None and desired_key == ISAAC_REPLAY_VIDEO_KEY:
        source_frames = recorded_frames.get(FRONT_VIDEO_KEY)
        desired_key = FRONT_VIDEO_KEY if source_frames is not None else desired_key
    if source_frames is None:
        raise RuntimeError(
            f"Episode {episode.get('source_path') or episode.get('recipe')} has no recorded frames for {desired_key!r}."
        )

    frames_dir = Path(source_frames).resolve()
    if not frames_dir.exists():
        raise RuntimeError(f'Live rollout frame directory does not exist: {frames_dir}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _encode_mp4_from_pngs(frames_dir, output_path, fps)
    merged_summary = dict(recorded_frame_summaries.get(desired_key) or {})
    merged_summary.update(summary)
    merged_summary['video_path'] = str(output_path)
    merged_summary.setdefault('renderer', 'live_rollout_frame_capture')
    return merged_summary


def _render_episode_video(
    *,
    episode: dict,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    video_mode: str,
    keep_video_frames: bool = False,
    camera_video_key: str | None = None,
) -> dict:
    if video_mode == 'isaac_replay':
        return _render_episode_video_isaac_subprocess(
            episode=episode,
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            keep_video_frames=keep_video_frames,
            camera_video_key=camera_video_key,
        )
    if video_mode == 'live_rollout':
        return _copy_live_rollout_video(
            episode=episode,
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            camera_video_key=camera_video_key,
        )
    return render_episode_video(
        episode=episode,
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
    )


def _episode_chunk_path(base_dir: Path, episode_index: int, chunk_size: int) -> Path:
    return base_dir / f'chunk-{episode_index // chunk_size:03d}'


def _episode_relative_data_path(episode_index: int, chunk_size: int) -> str:
    chunk_index = episode_index // chunk_size
    return f'data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet'


def _episode_relative_video_path(episode_index: int, chunk_size: int, video_key: str) -> str:
    chunk_index = episode_index // chunk_size
    return f'videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4'


def _build_episode_table(
    episode: dict,
    *,
    episode_index: int,
    global_index_start: int,
    task_index: int,
    fps: int,
    object_names: list[str],
) -> tuple[pa.Table, dict, int]:
    state_rows = []
    env_rows = []
    env_presence_rows = []
    action_rows = []
    reward_rows = []
    done_rows = []
    timestamps = []
    frame_indices = []
    episode_indices = []
    indices = []
    task_indices = []
    phase_indices = []
    phase_steps = []

    phase_lookup = _phase_index_lookup(episode)
    steps = episode.get('steps', [])
    for frame_index, step in enumerate(steps):
        state_rows.append(_flatten_robot_state(step, 'franka_left') + _flatten_robot_state(step, 'franka_right'))
        env_state, env_presence = _flatten_environment_state(step, object_names)
        env_rows.append(env_state)
        env_presence_rows.append(env_presence)
        action_rows.append(_flatten_robot_action(step, 'franka_left') + _flatten_robot_action(step, 'franka_right'))

        is_last_frame = frame_index == len(steps) - 1
        reward_rows.append(1.0 if is_last_frame and episode.get('metrics', {}).get('success', False) else 0.0)
        done_rows.append(bool(is_last_frame))
        timestamps.append(np.float32(frame_index / max(fps, 1)))
        frame_indices.append(frame_index)
        episode_indices.append(episode_index)
        indices.append(global_index_start + frame_index)
        task_indices.append(task_index)
        phase_indices.append(int(phase_lookup.get(step.get('phase'), -1)))
        robot_obs = step['observations'].get('franka_left', {})
        phase_steps.append(int(robot_obs.get('phase_step', frame_index)))

    state_names = _state_feature_names()
    action_names = _action_feature_names()
    environment_names, environment_presence_names = _environment_feature_names(object_names)

    table = pa.table(
        {
            'observation.state': pa.array(state_rows, type=pa.list_(pa.float32(), len(state_names))),
            'observation.environment_state': pa.array(env_rows, type=pa.list_(pa.float32(), len(environment_names))),
            'observation.environment_presence': pa.array(env_presence_rows, type=pa.list_(pa.float32(), len(environment_presence_names))),
            'action': pa.array(action_rows, type=pa.list_(pa.float32(), len(action_names))),
            'next.reward': pa.array(reward_rows, type=pa.float32()),
            'next.done': pa.array(done_rows, type=pa.bool_()),
            'timestamp': pa.array(timestamps, type=pa.float32()),
            'frame_index': pa.array(frame_indices, type=pa.int64()),
            'episode_index': pa.array(episode_indices, type=pa.int64()),
            'index': pa.array(indices, type=pa.int64()),
            'task_index': pa.array(task_indices, type=pa.int64()),
            'observation.phase_index': pa.array(phase_indices, type=pa.int64()),
            'observation.phase_step': pa.array(phase_steps, type=pa.int64()),
        }
    )

    stats = {
        'action': _vector_stats(np.asarray(action_rows, dtype=np.float32)),
        'observation.state': _vector_stats(np.asarray(state_rows, dtype=np.float32)),
        'observation.environment_state': _vector_stats(np.asarray(env_rows, dtype=np.float32)),
        'observation.environment_presence': _vector_stats(np.asarray(env_presence_rows, dtype=np.float32)),
        'next.reward': _scalar_stats(reward_rows),
        'timestamp': _scalar_stats(timestamps),
        'frame_index': _scalar_stats(frame_indices),
        'episode_index': _scalar_stats(episode_indices),
        'index': _scalar_stats(indices),
        'task_index': _scalar_stats(task_indices),
        'observation.phase_index': _scalar_stats(phase_indices),
        'observation.phase_step': _scalar_stats(phase_steps),
    }
    return table, stats, len(steps)


def export_lerobot_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    include_failures: bool,
    fps: int,
    video_width: int,
    video_height: int,
    chunk_size: int,
    video_key: str | None,
    video_mode: str = DEFAULT_VIDEO_MODE,
    keep_video_frames: bool = False,
) -> dict:
    all_episodes = load_episode_payloads(input_dir=input_dir)
    episodes = [
        episode
        for episode in all_episodes
        if include_failures or episode.get('metrics', {}).get('success', False)
    ]
    if not episodes:
        raise RuntimeError(f'No episodes found in {input_dir} after applying the success filter.')

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / 'meta'
    data_dir = output_dir / 'data'
    videos_dir = output_dir / 'videos'
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    task_to_index, tasks_rows = _task_index_lookup(episodes)
    object_names = _object_union(episodes)
    state_names = _state_feature_names()
    action_names = _action_feature_names()
    environment_names, environment_presence_names = _environment_feature_names(object_names)

    total_frames = 0
    total_video_streams = 0
    episodes_rows = []
    episodes_stats_rows = []
    video_feature_specs: dict[str, dict] = {}

    for episode_index, episode in enumerate(episodes):
        instruction = _instruction_for_episode(episode)
        task_index = task_to_index[instruction]
        table, table_stats, num_frames = _build_episode_table(
            episode,
            episode_index=episode_index,
            global_index_start=total_frames,
            task_index=task_index,
            fps=fps,
            object_names=object_names,
        )

        relative_data_path = _episode_relative_data_path(episode_index, chunk_size)
        data_path = output_dir / relative_data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)

        camera_specs = _episode_camera_specs(
            episode,
            video_mode=video_mode,
            default_width=video_width,
            default_height=video_height,
            video_key=video_key,
        )
        episode_video_paths: dict[str, str] = {}
        episode_video_summaries: dict[str, dict] = {}
        if video_mode in {'isaac_replay', 'live_rollout'} and camera_specs:
            if video_key is None:
                render_specs = camera_specs
            elif video_key in {camera_spec.get('video_key') for camera_spec in camera_specs}:
                render_specs = [camera_spec for camera_spec in camera_specs if camera_spec.get('video_key') == video_key]
            else:
                render_specs = camera_specs
        else:
            render_specs = [{'video_key': video_key}]

        for camera_spec in render_specs:
            camera_video_key = camera_spec.get('video_key', video_key)
            relative_video_path = _episode_relative_video_path(episode_index, chunk_size, camera_video_key)
            video_path = output_dir / relative_video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_summary = _render_episode_video(
                episode=episode,
                output_path=video_path,
                fps=fps,
                width=video_width,
                height=video_height,
                video_mode=video_mode,
                keep_video_frames=keep_video_frames,
                camera_video_key=camera_video_key if video_mode in {'isaac_replay', 'live_rollout'} and camera_specs else None,
            )

            table_stats[camera_video_key] = video_summary['image_stats']
            episode_video_paths[camera_video_key] = relative_video_path
            episode_video_summaries[camera_video_key] = video_summary
            video_feature_specs.setdefault(
                camera_video_key,
                {
                    'width': int(video_summary.get('width', video_width)),
                    'height': int(video_summary.get('height', video_height)),
                    'fps': int(video_summary.get('fps', fps)),
                },
            )
            total_video_streams += 1

        episodes_rows.append(
            {
                'episode_index': episode_index,
                'tasks': [instruction],
                'length': num_frames,
                'recipe': episode.get('recipe'),
                'scene_profile': episode.get('scene_profile'),
                'seed': episode.get('seed'),
                'success': bool(episode.get('metrics', {}).get('success', False)),
                'source_path': episode.get('source_path'),
                'data_path': relative_data_path,
                'video_paths': episode_video_paths,
                'camera_video_keys': list(episode_video_paths.keys()),
                'video_mode': video_mode,
            }
        )
        episodes_stats_rows.append(
            {
                'episode_index': episode_index,
                'stats': table_stats,
                'video_summaries': episode_video_summaries,
            }
        )
        total_frames += num_frames

    total_chunks = max(math.ceil(len(episodes) / max(chunk_size, 1)), 1)
    info = {
        'codebase_version': 'v2.1',
        'robot_type': 'dual_franka',
        'total_episodes': len(episodes),
        'total_frames': total_frames,
        'total_tasks': len(tasks_rows),
        'total_videos': total_video_streams,
        'total_chunks': total_chunks,
        'chunks_size': int(chunk_size),
        'fps': int(fps),
        'video_mode': video_mode,
        'splits': {'train': f'0:{len(episodes)}'},
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'source_benchmark': 'InternUtopia factory_dual_franka_assembly',
        'features': {
            'observation.state': {
                'dtype': 'float32',
                'shape': [len(state_names)],
                'names': state_names,
            },
            'observation.environment_state': {
                'dtype': 'float32',
                'shape': [len(environment_names)],
                'names': environment_names,
            },
            'observation.environment_presence': {
                'dtype': 'float32',
                'shape': [len(environment_presence_names)],
                'names': environment_presence_names,
            },
            'action': {
                'dtype': 'float32',
                'shape': [len(action_names)],
                'names': action_names,
            },
            'next.reward': {
                'dtype': 'float32',
                'shape': [1],
                'names': None,
            },
            'next.done': {
                'dtype': 'bool',
                'shape': [1],
                'names': None,
            },
            'timestamp': {
                'dtype': 'float32',
                'shape': [1],
                'names': None,
            },
            'frame_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'episode_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'task_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'observation.phase_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'observation.phase_step': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
        },
    }

    for camera_video_key, feature_spec in sorted(video_feature_specs.items()):
        info['features'][camera_video_key] = {
            'dtype': 'video',
            'shape': [feature_spec['height'], feature_spec['width'], 3],
            'names': ['height', 'width', 'channels'],
            'video_info': {
                'video.codec': 'mp4v',
                'video.pix_fmt': 'bgr24',
                'video.is_depth_map': False,
                'video.fps': int(feature_spec['fps']),
                'has_audio': False,
            },
        }

    _json_dump(meta_dir / 'info.json', info)
    _jsonl_dump(meta_dir / 'tasks.jsonl', tasks_rows)
    _jsonl_dump(meta_dir / 'episodes.jsonl', episodes_rows)
    _jsonl_dump(meta_dir / 'episodes_stats.jsonl', episodes_stats_rows)

    summary = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'num_source_episodes': len(all_episodes),
        'num_exported_episodes': len(episodes),
        'total_frames': total_frames,
        'video_key': video_key,
        'video_keys': sorted(video_feature_specs),
        'total_video_streams': total_video_streams,
        'video_mode': video_mode,
        'fps': fps,
        'include_failures': include_failures,
        'recipes': sorted({episode.get('recipe') for episode in episodes}),
        'scene_profiles': sorted({episode.get('scene_profile') or 'raw' for episode in episodes}),
    }
    _json_dump(meta_dir / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Export InternUtopia assembly episodes to a LeRobot-style dataset with meta/data/video.')
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly'),
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
    )
    parser.add_argument('--include-failures', action='store_true')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video-width', type=int, default=960)
    parser.add_argument('--video-height', type=int, default=540)
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--video-key', type=str, default=None)
    parser.add_argument('--video-mode', choices=['topdown', 'isaac_replay', 'live_rollout'], default=DEFAULT_VIDEO_MODE)
    parser.add_argument('--keep-video-frames', action='store_true')
    args, _unknown = parser.parse_known_args()

    video_mode = args.video_mode
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(video_mode).resolve()
    if args.video_key is not None:
        video_key = args.video_key
    elif video_mode in {'isaac_replay', 'live_rollout'}:
        video_key = None
    else:
        video_key = _default_video_key(video_mode)

    summary = export_lerobot_dataset(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=output_dir,
        include_failures=args.include_failures,
        fps=max(args.fps, 1),
        video_width=max(args.video_width, 64),
        video_height=max(args.video_height, 64),
        chunk_size=max(args.chunk_size, 1),
        video_key=video_key,
        video_mode=video_mode,
        keep_video_frames=args.keep_video_frames,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _render_single_episode_isaac_cli():
    parser = argparse.ArgumentParser(description='Render one assembly episode as a real Isaac replay MP4.')
    parser.add_argument('--render-single-episode-isaac', action='store_true')
    parser.add_argument('--episode-path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--summary-path', type=str, required=True)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video-width', type=int, default=960)
    parser.add_argument('--video-height', type=int, default=540)
    parser.add_argument('--camera-video-key', type=str, default=None)
    parser.add_argument('--keep-video-frames', action='store_true')
    args, _unknown = parser.parse_known_args()

    episode = json.loads(Path(args.episode_path).read_text(encoding='utf-8'))
    render_episode_video_isaac_replay(
        episode=episode,
        output_path=Path(args.output).resolve(),
        fps=max(args.fps, 1),
        width=max(args.video_width, 64),
        height=max(args.video_height, 64),
        keep_frames=args.keep_video_frames,
        summary_path=Path(args.summary_path).resolve(),
        camera_video_key=args.camera_video_key,
    )


if __name__ == '__main__':
    if '--render-single-episode-isaac' in sys.argv:
        _render_single_episode_isaac_cli()
    else:
        main()
