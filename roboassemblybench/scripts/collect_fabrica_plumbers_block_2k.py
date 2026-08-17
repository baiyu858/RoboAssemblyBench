from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roboassemblybench.core.domain_randomization import (
    RANDOMIZATION_PROFILE_CHOICES,
    apply_domain_randomization,
    normalize_randomization_profile,
)
from roboassemblybench.core.process_lock import exclusive_process_lock
from roboassemblybench.datasets.cartesian_episode import (
    ACTION_NAMES,
    ACTION_SEMANTICS,
    CAMERA_KEYS,
    STATE_NAMES,
    cartesian_trajectory_errors,
)
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_v3'
MANIFEST_NAME = 'collection_manifest.json'
QUALIFICATION_MANIFEST_NAME = 'qualification_manifest.json'
QUALIFICATION_STATUS_NAME = 'qualification_status.json'
FRONT_CAMERA_KEY = 'observation.images.front'
VISUAL_SAMPLE_FRACTIONS = tuple(index / 16.0 for index in range(17))
VISUAL_EARLY_FRAME_COUNT = 4
VISUAL_INITIAL_ROBOT_VISIBILITY_GRACE_FRAMES = 2
ROBOT_VISIBILITY_REGIONS = {
    'left': {
        'anchor': (0.025, 0.235, 0.48, 0.88),
        'structure': (0.0, 0.52, 0.04, 0.90),
    },
    'right': {
        'anchor': (0.535, 0.765, 0.48, 0.88),
        'structure': (0.48, 1.0, 0.04, 0.90),
    },
}
ROBOT_ANCHOR_EDGE_MEAN_THRESHOLD = 15.0
ROBOT_ANCHOR_EDGE_P90_THRESHOLD = 35.0
ROBOT_ANCHOR_LOCAL_CONTRAST_THRESHOLD = 9.5
ROBOT_STRUCTURE_BRIGHT_NEUTRAL_FRACTION_THRESHOLD = 0.02
# Backward-compatible alias for quality reports produced before the anchor detector.
ROBOT_EDGE_P90_THRESHOLD = ROBOT_ANCHOR_EDGE_P90_THRESHOLD
HORIZONTAL_LIGHT_STREAK_SPAN_THRESHOLD = 0.35
VERTICAL_LIGHT_STREAK_SPAN_THRESHOLD = 0.90
BACKGROUND_EDGE_MEAN_THRESHOLD = 4.0
RENDER_ARTIFACT_CHROMA_LAPLACIAN_THRESHOLD = 2.1
RENDER_ARTIFACT_HIGH_CHROMA_FRACTION_THRESHOLD = 0.008
RENDER_ARTIFACT_LAPLACIAN_RATIO_THRESHOLD = 1.35


@contextmanager
def _exclusive_collection_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(
        output_dir / '.collection.lock.d',
        description='collector',
    ):
        yield


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def _available_memory_gib() -> float:
    values = {}
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, raw_value = line.split(':', 1)
        values[key] = int(raw_value.strip().split()[0])
    return float(values.get('MemAvailable', 0)) / (1024.0**2)


def _visual_sample_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    uniform_count = min(len(VISUAL_SAMPLE_FRACTIONS), frame_count)
    uniform_indices = np.linspace(0, frame_count - 1, uniform_count).round().astype(int)
    early_indices = range(min(VISUAL_EARLY_FRAME_COUNT, frame_count))
    return sorted({int(value) for value in (*early_indices, *uniform_indices)})


def _normalized_roi(array: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    height, width = array.shape[:2]
    x_start, x_end, y_start, y_end = bounds
    return array[
        int(y_start * height) : int(y_end * height),
        int(x_start * width) : int(x_end * width),
    ]


def _robot_visibility_metrics(gray: np.ndarray, chroma: np.ndarray, *, side: str) -> dict[str, Any]:
    regions = ROBOT_VISIBILITY_REGIONS[side]
    anchor = _normalized_roi(gray, regions['anchor'])
    structure = _normalized_roi(gray, regions['structure'])
    structure_chroma = _normalized_roi(chroma, regions['structure'])

    blurred_anchor = cv2.GaussianBlur(anchor, (5, 5), 0)
    anchor_edges = cv2.magnitude(
        cv2.Sobel(blurred_anchor, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(blurred_anchor, cv2.CV_32F, 0, 1, ksize=3),
    )
    local_background = cv2.GaussianBlur(blurred_anchor, (0, 0), 12)
    local_contrast = cv2.absdiff(blurred_anchor, local_background)
    bright_neutral = (structure >= 210) & (structure_chroma <= 35)

    edge_mean = float(np.mean(anchor_edges))
    edge_p90 = float(np.percentile(anchor_edges, 90))
    local_contrast_mean = float(np.mean(local_contrast))
    structure_bright_neutral_fraction = float(np.mean(bright_neutral))
    anchor_visible = bool(
        edge_mean >= ROBOT_ANCHOR_EDGE_MEAN_THRESHOLD
        and edge_p90 >= ROBOT_ANCHOR_EDGE_P90_THRESHOLD
        and local_contrast_mean >= ROBOT_ANCHOR_LOCAL_CONTRAST_THRESHOLD
    )
    structure_visible = bool(
        structure_bright_neutral_fraction >= ROBOT_STRUCTURE_BRIGHT_NEUTRAL_FRACTION_THRESHOLD
    )
    return {
        'anchor_edge_mean': edge_mean,
        'anchor_edge_p90': edge_p90,
        'anchor_local_contrast_mean': local_contrast_mean,
        'structure_bright_neutral_fraction': structure_bright_neutral_fraction,
        'anchor_visible': anchor_visible,
        'structure_visible': structure_visible,
        'visible': bool(anchor_visible and structure_visible),
    }


def _front_visual_quality(video_path: Path) -> dict[str, Any]:
    """Check representative front frames for rendering and robot visibility failures."""

    result: dict[str, Any] = {
        'valid': False,
        'errors': [],
        'video_path': str(video_path),
        'sample_frame_indices': [],
        'luminance_mean': [],
        'luminance_std': [],
        'overexposed_fraction': [],
        'left_robot_roi_edge_mean': [],
        'left_robot_roi_edge_p90': [],
        'left_robot_roi_bright_neutral_fraction': [],
        'left_robot_anchor_local_contrast_mean': [],
        'left_robot_anchor_visible': [],
        'left_robot_structure_visible': [],
        'left_robot_visible': [],
        'right_robot_roi_edge_mean': [],
        'right_robot_roi_edge_p90': [],
        'right_robot_roi_bright_neutral_fraction': [],
        'right_robot_anchor_local_contrast_mean': [],
        'right_robot_anchor_visible': [],
        'right_robot_structure_visible': [],
        'right_robot_visible': [],
        'bright_row_span': [],
        'bright_column_span': [],
        'background_edge_mean': [],
        'chroma_laplacian_mean': [],
        'high_chroma_fraction': [],
    }
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or frame_count <= 0:
            result['errors'].append('decode')
            return result

        frame_indices = _visual_sample_frame_indices(frame_count)
        expected_sample_count = len(frame_indices)

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            decoded, frame = capture.read()
            if not decoded or frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 32:
                result['errors'].append('decode')
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            chroma = np.max(frame, axis=2) - np.min(frame, axis=2)

            result['sample_frame_indices'].append(frame_index)
            result['luminance_mean'].append(float(np.mean(gray)))
            result['luminance_std'].append(float(np.std(gray)))
            result['overexposed_fraction'].append(float(np.mean(gray >= 250)))
            chroma_laplacian = cv2.Laplacian(chroma.astype(np.float32), cv2.CV_32F)
            result['chroma_laplacian_mean'].append(float(np.mean(np.abs(chroma_laplacian))))
            result['high_chroma_fraction'].append(float(np.mean(chroma > 45)))

            for side in ('left', 'right'):
                robot_metrics = _robot_visibility_metrics(gray, chroma, side=side)
                result[f'{side}_robot_roi_edge_mean'].append(robot_metrics['anchor_edge_mean'])
                result[f'{side}_robot_roi_edge_p90'].append(robot_metrics['anchor_edge_p90'])
                result[f'{side}_robot_roi_bright_neutral_fraction'].append(
                    robot_metrics['structure_bright_neutral_fraction']
                )
                result[f'{side}_robot_anchor_local_contrast_mean'].append(
                    robot_metrics['anchor_local_contrast_mean']
                )
                result[f'{side}_robot_anchor_visible'].append(robot_metrics['anchor_visible'])
                result[f'{side}_robot_structure_visible'].append(robot_metrics['structure_visible'])
                result[f'{side}_robot_visible'].append(robot_metrics['visible'])

            bright = gray >= 245
            result['bright_row_span'].append(
                float(np.max(np.mean(bright[:, int(0.05 * width) : int(0.95 * width)], axis=1)))
            )
            result['bright_column_span'].append(
                float(np.max(np.mean(bright[int(0.05 * height) : int(0.95 * height), :], axis=0)))
            )
            background_roi = gray[
                int(0.03 * height) : int(0.38 * height),
                int(0.25 * width) : int(0.75 * width),
            ]
            background_edge = cv2.magnitude(
                cv2.Sobel(background_roi, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(background_roi, cv2.CV_32F, 0, 1, ksize=3),
            )
            result['background_edge_mean'].append(float(np.mean(background_edge)))
    finally:
        capture.release()

    if len(result['sample_frame_indices']) != expected_sample_count:
        result['errors'].append('decode')
    if any(value < 8.0 or value > 247.0 for value in result['luminance_mean']):
        result['errors'].append('luminance')
    if any(value < 10.0 for value in result['luminance_std']):
        result['errors'].append('low-variance')
    if any(value > 0.25 for value in result['overexposed_fraction']):
        result['errors'].append('overexposed')

    missing_robot_sides = []
    for side in ('left', 'right'):
        edge_values = result[f'{side}_robot_roi_edge_mean']
        result[f'{side}_robot_roi_edge_median'] = float(np.median(edge_values)) if edge_values else 0.0
        edge_p90_values = result[f'{side}_robot_roi_edge_p90']
        result[f'{side}_robot_roi_edge_p90_median'] = (
            float(np.median(edge_p90_values)) if edge_p90_values else 0.0
        )
        neutral_values = result[f'{side}_robot_roi_bright_neutral_fraction']
        result[f'{side}_robot_roi_bright_neutral_fraction_median'] = (
            float(np.median(neutral_values)) if neutral_values else 0.0
        )
        local_contrast_values = result[f'{side}_robot_anchor_local_contrast_mean']
        result[f'{side}_robot_anchor_local_contrast_mean_median'] = (
            float(np.median(local_contrast_values)) if local_contrast_values else 0.0
        )
        visibility_values = [
            visible
            for frame_index, visible in zip(
                result['sample_frame_indices'],
                result[f'{side}_robot_visible'],
            )
            if frame_index >= VISUAL_INITIAL_ROBOT_VISIBILITY_GRACE_FRAMES
        ]
        result[f'{side}_robot_visible_fraction'] = (
            float(np.mean(visibility_values)) if visibility_values else 0.0
        )
        if not visibility_values or not all(visibility_values):
            missing_robot_sides.append(side)
            result['errors'].append(f'{side}-robot-not-visible')
    if missing_robot_sides:
        result['errors'].append('robot-not-visible')
    if (
        max(result['bright_row_span'], default=0.0) >= HORIZONTAL_LIGHT_STREAK_SPAN_THRESHOLD
        or max(result['bright_column_span'], default=0.0) >= VERTICAL_LIGHT_STREAK_SPAN_THRESHOLD
    ):
        result['errors'].append('light-streak')
    result['chroma_laplacian_median'] = (
        float(np.median(result['chroma_laplacian_mean'])) if result['chroma_laplacian_mean'] else 0.0
    )
    result['high_chroma_fraction_median'] = (
        float(np.median(result['high_chroma_fraction'])) if result['high_chroma_fraction'] else 0.0
    )
    if any(
        laplacian >= RENDER_ARTIFACT_CHROMA_LAPLACIAN_THRESHOLD
        and laplacian
        >= result['chroma_laplacian_median'] * RENDER_ARTIFACT_LAPLACIAN_RATIO_THRESHOLD
        and high_chroma >= RENDER_ARTIFACT_HIGH_CHROMA_FRACTION_THRESHOLD
        for laplacian, high_chroma in zip(
            result['chroma_laplacian_mean'],
            result['high_chroma_fraction'],
        )
    ):
        result['errors'].append('temporal-render-artifact')
    result['background_edge_mean_median'] = (
        float(np.median(result['background_edge_mean'])) if result['background_edge_mean'] else 0.0
    )
    result['background_edge_mean_min'] = min(result['background_edge_mean'], default=0.0)
    if result['background_edge_mean_min'] < BACKGROUND_EDGE_MEAN_THRESHOLD:
        result['errors'].append('factory-background-not-visible')

    result['errors'] = sorted(set(result['errors']))
    result['valid'] = not result['errors']
    return result


def _quality_check_episode(
    metadata_path: Path,
    *,
    expected_recipe_fingerprint: str | None = None,
    allowed_layout_seeds: set[int] | None = None,
    require_extended_observations: bool = False,
    require_visual_quality: bool = False,
    expected_randomization_profile: str | None = None,
) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    errors = []
    if metadata.get('schema_version') != 'roboassemblybench_raw_cartesian_v1':
        errors.append('schema_version')
    if list(metadata.get('state_names') or []) != list(STATE_NAMES):
        errors.append('state_schema')
    if list(metadata.get('action_names') or []) != list(ACTION_NAMES):
        errors.append('action_schema')
    if metadata.get('action_semantics') != ACTION_SEMANTICS:
        errors.append('action_semantics')
    recipe_fingerprint = str(metadata.get('recipe_fingerprint') or '')
    if expected_recipe_fingerprint is not None and recipe_fingerprint != expected_recipe_fingerprint:
        errors.append('recipe_fingerprint')
    if not bool((metadata.get('metrics') or {}).get('success', False)):
        errors.append('task_success')
    if not bool((metadata.get('domain_randomization') or {}).get('enabled', False)):
        errors.append('domain_randomization')
    domain_randomization = metadata.get('domain_randomization') or {}
    actual_randomization_profile = normalize_randomization_profile(domain_randomization.get('profile'))
    if (
        expected_randomization_profile is not None
        and actual_randomization_profile != normalize_randomization_profile(expected_randomization_profile)
    ):
        errors.append('randomization_profile')
    randomized_groups = domain_randomization.get('groups') or {}
    appearance_groups = domain_randomization.get('appearance_groups') or {}
    if require_extended_observations and not {'start_parts', 'assembly_base'}.issubset(randomized_groups):
        errors.append('position_randomization_groups')
    if require_extended_observations:
        if actual_randomization_profile == 'mixed':
            if not {'table_surface', 'background'}.issubset(appearance_groups):
                errors.append('appearance_randomization_groups')
            if not domain_randomization.get('visual_distractors'):
                errors.append('visual_distractors')
        elif actual_randomization_profile == 'object_distractors':
            if 'visual_distractors' not in domain_randomization:
                errors.append('visual_distractors')
        elif actual_randomization_profile == 'texture':
            if not (domain_randomization.get('table_texture') or {}).get('path'):
                errors.append('table_texture')
        elif actual_randomization_profile == 'lighting':
            light_count = len(domain_randomization.get('lighting') or [])
            if not 1 <= light_count <= 4:
                errors.append('lighting_randomization')
        elif actual_randomization_profile == 'table_color':
            if 'table_surface' not in appearance_groups:
                errors.append('table_color_randomization')
        elif actual_randomization_profile == 'scene':
            if not domain_randomization.get('scene') or 'background' not in appearance_groups:
                errors.append('scene_randomization')
    layout_seed = int(metadata.get('layout_seed', domain_randomization.get('seed', metadata.get('seed', -1))))
    if int(domain_randomization.get('seed', layout_seed)) != layout_seed:
        errors.append('layout_seed_metadata')
    if allowed_layout_seeds is not None and layout_seed not in allowed_layout_seeds:
        errors.append('layout_seed_contract')
    fps = int(metadata.get('fps', 0))
    simulation_fps = int(metadata.get('simulation_fps', 0))
    frame_stride = int(metadata.get('frame_stride', 0))
    timing = metadata.get('timing') or {}
    expected_timing = {
        'physics_fps': simulation_fps,
        'control_fps': simulation_fps,
        'dataset_fps': fps,
        'dataset_frame_stride': frame_stride,
        'rendering_interval': frame_stride - 1,
        'camera_render_period_steps': frame_stride,
    }
    try:
        timing_matches = all(int(timing.get(key, -1)) == value for key, value in expected_timing.items())
    except (TypeError, ValueError):
        timing_matches = False
    if (
        fps <= 0
        or simulation_fps != fps * frame_stride
        or not timing_matches
        or not bool(timing.get('camera_state_action_aligned', False))
    ):
        errors.append('timing_contract')
    frame_count = int(metadata.get('frame_count', 0))
    if frame_count < 100:
        errors.append('frame_count')
    if set(metadata.get('videos') or {}) != set(CAMERA_KEYS):
        errors.append('camera_keys')
    if any(int(value) != frame_count for value in (metadata.get('video_frame_counts') or {}).values()):
        errors.append('video_frame_alignment')

    if require_visual_quality:
        scene_asset_path = str(metadata.get('scene_asset_path') or '')
        scene_profile = str(metadata.get('scene_profile') or '')
        if not metadata.get('scene_profile'):
            errors.append('visual:scene-profile')
        if not scene_asset_path or Path(scene_asset_path).name == 'empty.usd':
            errors.append('visual:scene-asset')
        if not metadata.get('scene_family'):
            errors.append('visual:scene-family')
        if scene_profile == 'taoyuan_grscenes_tabletop':
            if metadata.get('scene_asset_source') != 'primary':
                errors.append('visual:scene-source')
            if Path(scene_asset_path).name != 'warehouse_with_forklifts.usd':
                errors.append('visual:scene-asset')
            if metadata.get('scene_family') != 'isaac_simple_warehouse_tabletop':
                errors.append('visual:scene-family')
        runtime_integrity = metadata.get('runtime_scene_integrity') or {}
        if (
            not isinstance(runtime_integrity, dict)
            or not bool((runtime_integrity.get('start') or {}).get('valid', False))
            or not bool((runtime_integrity.get('end') or {}).get('valid', False))
        ):
            errors.append('visual:runtime-scene-integrity')

    if require_extended_observations:
        depth_streams = metadata.get('depth') or {}
        if set(depth_streams) != set(CAMERA_KEYS):
            errors.append('depth_camera_keys')
        for camera_key, depth_metadata in depth_streams.items():
            if camera_key not in CAMERA_KEYS or not isinstance(depth_metadata, dict):
                errors.append(f'depth:{camera_key}')
                continue
            depth_path = Path(depth_metadata.get('path') or '')
            shape = depth_metadata.get('shape') or []
            count = int(depth_metadata.get('count', -1))
            if (
                count != frame_count
                or len(shape) != 2
                or depth_metadata.get('dtype') != 'float16'
                or depth_metadata.get('compression') != 'zstd'
                or not depth_path.is_file()
                or depth_path.stat().st_size == 0
            ):
                errors.append(f'depth:{camera_key}')
        if not bool((metadata.get('capabilities') or {}).get('depth', False)):
            errors.append('depth_capability')

        annotation_path = Path(metadata.get('annotation_path') or '')
        if not annotation_path.is_file():
            errors.append('annotation_missing')
        else:
            annotation = _load_json(annotation_path)
            if (
                annotation.get('schema_version') != 'roboassemblybench_long_horizon_annotation_v1'
                or not annotation.get('assembly_steps')
                or not annotation.get('phase_annotations')
                or not annotation.get('robot_roles')
                or not annotation.get('execution_order')
                or 'task_result' not in annotation
            ):
                errors.append('annotation_schema')

    trajectory_path = Path(metadata.get('trajectory_path') or '')
    if not trajectory_path.is_file():
        errors.append('trajectory_missing')
    else:
        with np.load(trajectory_path) as trajectory:
            states = np.asarray(trajectory['observation_state'])
            actions = np.asarray(trajectory['action'])
            if states.shape != (frame_count, len(STATE_NAMES)):
                errors.append('state')
            if actions.shape != (frame_count, len(ACTION_NAMES)):
                errors.append('action')
            if states.shape == (frame_count, len(STATE_NAMES)) and actions.shape == (
                frame_count,
                len(ACTION_NAMES),
            ):
                trajectory_errors = cartesian_trajectory_errors(
                    states,
                    actions,
                    simulation_steps=trajectory.get('simulation_step'),
                    frame_stride=frame_stride,
                )
                errors.extend(f'trajectory:{error}' for error in trajectory_errors)
            expected_features = {
                'joint_state': (frame_count, 14),
                'joint_velocity': (frame_count, 14),
                'joint_effort': (frame_count, 14),
                'wrist_wrench': (frame_count, 12),
                'collision_signal': (frame_count, 4),
                'subtask_index': (frame_count,),
                'substage_index': (frame_count,),
                'waiting_state': (frame_count,),
                'handoff_state': (frame_count,),
            }
            if require_extended_observations:
                for feature_name, expected_shape in expected_features.items():
                    if feature_name not in trajectory or np.asarray(trajectory[feature_name]).shape != expected_shape:
                        errors.append(f'trajectory:{feature_name}')

    for camera_key, video_path in (metadata.get('videos') or {}).items():
        if camera_key not in CAMERA_KEYS or not Path(video_path).is_file() or Path(video_path).stat().st_size == 0:
            errors.append(f'video:{camera_key}')
    visual_quality = None
    if require_visual_quality:
        front_video_path = Path((metadata.get('videos') or {}).get(FRONT_CAMERA_KEY) or '')
        if front_video_path.is_file() and front_video_path.stat().st_size > 0:
            visual_quality = _front_visual_quality(front_video_path)
            errors.extend(f'visual:{error}' for error in visual_quality['errors'])
        else:
            errors.append('visual:front-video')
    return {
        'valid': not errors,
        'errors': sorted(set(errors)),
        'seed': int(metadata.get('seed', -1)),
        'layout_seed': layout_seed,
        'frame_count': frame_count,
        'metadata_path': str(metadata_path.resolve()),
        'recipe_fingerprint': recipe_fingerprint,
        'domain_randomization': domain_randomization,
        'randomization_profile': actual_randomization_profile,
        'visual_quality': visual_quality,
    }


def _scan_existing(
    output_dir: Path,
    *,
    expected_recipe_fingerprint: str | None = None,
    allowed_layout_seeds: set[int] | None = None,
    require_extended_observations: bool = False,
    require_visual_quality: bool = False,
    expected_randomization_profile: str | None = None,
) -> dict[int, dict[str, Any]]:
    episodes = {}
    for metadata_path in output_dir.rglob('episode_*_cartesian_raw/metadata.json'):
        quality = _quality_check_episode(
            metadata_path,
            expected_recipe_fingerprint=expected_recipe_fingerprint,
            allowed_layout_seeds=allowed_layout_seeds,
            require_extended_observations=require_extended_observations,
            require_visual_quality=require_visual_quality,
            expected_randomization_profile=expected_randomization_profile,
        )
        if quality['valid']:
            episodes[quality['seed']] = quality
    return episodes


def _initial_manifest(args, existing: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        'schema_version': 'roboassemblybench_position_2k_collection_v2',
        'recipe': args.recipe,
        'scene_profile': args.scene_profile,
        'recipe_fingerprint': str(getattr(args, 'recipe_fingerprint', '')),
        'target_successful_episodes': int(args.num_episodes),
        'max_attempts': int(args.max_attempts),
        'start_seed': int(args.start_seed),
        'next_seed': max([int(args.start_seed), *[seed + 1 for seed in existing]], default=int(args.start_seed)),
        'batch_size': int(args.batch_size),
        'dataset_fps': int(args.dataset_fps),
        'dataset_frame_stride': int(args.dataset_frame_stride),
        'rendering_fps': int(args.rendering_fps),
        'worker_timeout_seconds': float(getattr(args, 'worker_timeout_seconds', 1800.0)),
        'worker_stall_timeout_seconds': float(getattr(args, 'worker_stall_timeout_seconds', 0.0)),
        'timing_contract': {
            'physics_fps': int(args.rendering_fps),
            'control_fps': int(args.rendering_fps),
            'dataset_fps': int(args.dataset_fps),
            'dataset_frame_stride': int(args.dataset_frame_stride),
            'rendering_interval': int(args.dataset_frame_stride) - 1,
            'camera_render_period_steps': int(args.dataset_frame_stride),
            'camera_fps': int(args.rendering_fps) / int(args.dataset_frame_stride),
            'camera_state_action_aligned': True,
        },
        'domain_randomization': True,
        'randomization_profile': normalize_randomization_profile(getattr(args, 'randomization_profile', None)),
        'collection_layout_seeds': [int(seed) for seed in getattr(args, 'layout_seeds', [])],
        'layout_assignment': (
            'episode_seed' if bool(getattr(args, 'unique_layout_seeds', False)) else 'episode_seed_modulo_round_robin'
        ),
        'single_worker': True,
        'qualification_manifest': str(getattr(args, 'qualification_manifest', '')),
        'successful_episodes': {str(seed): value for seed, value in sorted(existing.items())},
        'failed_attempts': [],
        'batches': [],
    }


def _reconcile_manifest_on_resume(
    manifest: dict[str, Any],
    existing: dict[int, dict[str, Any]],
) -> None:
    manifest['successful_episodes'] = {str(seed): value for seed, value in sorted(existing.items())}
    existing_next_seed = max((seed + 1 for seed in existing), default=0)
    manifest['next_seed'] = max(int(manifest.get('next_seed', 0)), existing_next_seed)
    for batch in manifest.get('batches') or []:
        if batch.get('status') != 'running':
            continue
        batch_dir = Path(batch.get('batch_dir') or '').resolve()
        qualities = [
            quality
            for quality in existing.values()
            if Path(quality['metadata_path']).resolve().is_relative_to(batch_dir)
        ]
        batch['status'] = 'recovered_completed' if qualities else 'recovered_interrupted'
        batch['quality'] = qualities
        batch['reconciled_at_unix'] = time.time()
    manifest['num_successful'] = len(existing)
    manifest['num_failed_attempts'] = len(manifest.get('failed_attempts') or [])


def _batch_command(
    args,
    *,
    seeds: list[int],
    batch_dir: Path,
    results_path: Path,
    layout_seeds: list[int] | None = None,
    record_raw: bool = True,
) -> list[str]:
    environment = [
        'PYTHONNOUSERSITE=1',
        'PYTHONUNBUFFERED=1',
        'OMP_NUM_THREADS=4',
        'MKL_NUM_THREADS=4',
        'OPENBLAS_NUM_THREADS=4',
        f'UR5E_DEBUG_GRASP={os.environ.get("UR5E_DEBUG_GRASP", "0")}',
        f'UR5E_DEBUG_TRANSPORT_EVERY={os.environ.get("UR5E_DEBUG_TRANSPORT_EVERY", "0")}',
    ]
    configured_isaac_python = getattr(args, 'isaac_python', None)
    if configured_isaac_python:
        isaac_python = Path(configured_isaac_python).expanduser().resolve()
        if not isaac_python.is_file():
            raise RuntimeError(f'Isaac Python executable was not found: {isaac_python}')
        command = ['env', *environment, str(isaac_python)]
    else:
        conda = shutil.which('conda')
        if conda is None:
            raise RuntimeError('conda executable was not found; pass --isaac-python for a direct runtime.')
        command = [
            conda,
            'run',
            '--no-capture-output',
            '-n',
            args.conda_env,
            'env',
            *environment,
            'python',
        ]
    command.extend(
        [
        str(REPO_ROOT / 'roboassemblybench' / 'scripts' / 'generate_demos.py'),
        '--worker-mode',
        'collect',
        '--worker-recipe',
        args.recipe,
        '--worker-scene-profile',
        args.scene_profile,
        '--worker-results-path',
        str(results_path),
        '--worker-seeds',
        *[str(seed) for seed in seeds],
        '--max-trials',
        str(len(seeds)),
        '--start-seed',
        str(seeds[0]),
        '--output-dir',
        str(batch_dir),
        '--dataset-fps',
        str(args.dataset_fps),
        '--dataset-frame-stride',
        str(args.dataset_frame_stride),
        '--rendering-fps',
        str(args.rendering_fps),
        '--domain-randomization',
        '--randomization-profile',
        normalize_randomization_profile(getattr(args, 'randomization_profile', None)),
        '--skip-episode-steps',
        '--headless',
        ]
    )
    if layout_seeds is not None:
        if len(layout_seeds) != len(seeds):
            raise ValueError('layout_seeds must contain exactly one entry per episode seed.')
        worker_seed_end = command.index('--max-trials')
        command[worker_seed_end:worker_seed_end] = [
            '--worker-layout-seeds',
            *[str(seed) for seed in layout_seeds],
        ]
    if record_raw:
        command.insert(command.index('--dataset-fps'), '--record-lerobot-raw')
    return command


def _randomization_feature(
    recipe_spec: dict[str, Any],
    seed: int,
    *,
    randomization_profile: str | None = None,
) -> np.ndarray:
    _, result = apply_domain_randomization(
        recipe_spec,
        seed=seed,
        enabled_override=True,
        profile=randomization_profile,
    )
    values = []
    for group_name in sorted(result.get('groups') or {}):
        values.extend(float(value) for value in result['groups'][group_name]['translation'])
    return np.asarray(values, dtype=float)


def _select_qualification_seeds(
    recipe_spec: dict[str, Any],
    *,
    count: int,
    candidate_pool: int,
    start_seed: int,
    required_seeds: list[int],
) -> list[int]:
    """Select deterministic qualification seeds near the nominal layout."""

    count = max(int(count), 0)
    if count == 0:
        return []
    candidate_pool = max(int(candidate_pool), count)
    candidates = list(range(int(start_seed), int(start_seed) + candidate_pool))
    for seed in required_seeds:
        seed = int(seed)
        if seed not in candidates:
            candidates.append(seed)

    feature_by_seed = {seed: _randomization_feature(recipe_spec, seed) for seed in candidates}
    matrix = np.stack([feature_by_seed[seed] for seed in candidates])
    span = np.ptp(matrix, axis=0)
    span[span < 1e-12] = 1.0
    normalized = {seed: (feature_by_seed[seed] - np.min(matrix, axis=0)) / span for seed in candidates}

    selected = []
    for seed in required_seeds:
        seed = int(seed)
        if seed not in selected:
            selected.append(seed)
        if len(selected) == count:
            return selected

    center = np.full(matrix.shape[1], 0.5, dtype=float)
    remaining = sorted(
        (seed for seed in candidates if seed not in selected),
        key=lambda seed: (float(np.linalg.norm(normalized[seed] - center)), seed),
    )
    selected.extend(remaining[: count - len(selected)])
    return selected


def _qualification_result(results_path: Path, *, seed: int) -> dict[str, Any]:
    results = _load_json(results_path) if results_path.is_file() else []
    result = next((item for item in results if int(item.get('seed', -1)) == int(seed)), {})
    history = list(result.get('phase_transition_history') or [])
    terminal = history[-1] if history else {}
    return {
        'seed': int(seed),
        'passed': bool(result.get('success', False)) and not bool(result.get('failed', False)),
        'phase_status': result.get('phase_status'),
        'steps': int(result.get('steps', 0)),
        'terminal_reason': terminal.get('reason', 'missing-qualification-result'),
        'terminal_phase': terminal.get('from_phase'),
        'success_diagnostics': result.get('success_diagnostics') or [],
        'results_path': str(results_path.resolve()),
    }


def _write_qualification_status(output_dir: Path, manifest: dict[str, Any]) -> None:
    failed_result = next(
        (item for item in manifest.get('results', []) if not bool(item.get('passed', False))),
        None,
    )
    _write_json_atomic(
        output_dir / QUALIFICATION_STATUS_NAME,
        {
            'schema_version': 'roboassemblybench_recipe_qualification_status_v1',
            'recipe': manifest.get('recipe'),
            'scene_profile': manifest.get('scene_profile'),
            'recipe_fingerprint': manifest.get('recipe_fingerprint'),
            'passed': bool(manifest.get('passed', False)),
            'failed': bool(manifest.get('failed', False)),
            'selected_seeds': manifest.get('selected_seeds') or [],
            'num_passed': sum(bool(item.get('passed', False)) for item in manifest.get('results', [])),
            'num_resource_aborts': len(manifest.get('resource_aborts') or []),
            'failed_result': failed_result,
            'qualification_manifest': str(
                output_dir
                / 'qualification'
                / str(manifest.get('recipe_fingerprint') or '')
                / QUALIFICATION_MANIFEST_NAME
            ),
            'updated_at_unix': time.time(),
        },
    )


def _reconcile_qualification_seed_contract(
    manifest: dict[str, Any],
    selected_seeds: list[int],
) -> bool:
    previous_seeds = [int(seed) for seed in manifest.get('selected_seeds') or []]
    selected_seeds = [int(seed) for seed in selected_seeds]
    if previous_seeds == selected_seeds:
        return False

    selected_set = set(selected_seeds)
    retained_results = [
        item
        for item in manifest.get('results') or []
        if int(item.get('seed', -1)) in selected_set and bool(item.get('passed', False))
    ]
    result_by_seed = {int(item['seed']): item for item in retained_results}
    manifest.setdefault('seed_contract_history', []).append(
        {
            'previous_seeds': previous_seeds,
            'selected_seeds': selected_seeds,
            'retained_passed_seeds': [seed for seed in selected_seeds if seed in result_by_seed],
            'changed_at_unix': time.time(),
        }
    )
    manifest['selected_seeds'] = selected_seeds
    manifest['required_passes'] = len(selected_seeds)
    manifest['results'] = [result_by_seed[seed] for seed in selected_seeds if seed in result_by_seed]
    manifest['passed'] = False
    manifest['failed'] = False
    manifest.pop('failed_at_unix', None)
    manifest.pop('finished_at_unix', None)
    return True


def _ensure_recipe_qualified(args, output_dir: Path) -> dict[str, Any]:
    if bool(args.skip_qualification):
        return {
            'schema_version': 'roboassemblybench_recipe_qualification_v1',
            'recipe_fingerprint': str(args.recipe_fingerprint),
            'passed': True,
            'skipped': True,
        }

    fingerprint = str(args.recipe_fingerprint)
    qualification_root = output_dir / 'qualification' / fingerprint
    manifest_path = qualification_root / QUALIFICATION_MANIFEST_NAME
    qualification_root.mkdir(parents=True, exist_ok=True)
    selected_seeds = _select_qualification_seeds(
        args.recipe_spec,
        count=int(args.qualification_seed_count),
        candidate_pool=int(args.qualification_candidate_pool),
        start_seed=int(args.qualification_start_seed),
        required_seeds=list(args.qualification_required_seeds),
    )
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if str(manifest.get('recipe_fingerprint') or '') != fingerprint:
            raise RuntimeError(f'Qualification manifest fingerprint mismatch: {manifest_path}.')
        if _reconcile_qualification_seed_contract(manifest, selected_seeds):
            _write_json_atomic(manifest_path, manifest)
            _write_qualification_status(output_dir, manifest)
        if bool(manifest.get('passed', False)):
            _write_qualification_status(output_dir, manifest)
            return manifest
        failed_results = [item for item in manifest.get('results', []) if not bool(item.get('passed', False))]
        resource_failures = [item for item in failed_results if item.get('resource_abort') is not None]
        if resource_failures:
            manifest.setdefault('resource_aborts', []).extend(resource_failures)
            manifest['results'] = [
                item
                for item in manifest.get('results', [])
                if bool(item.get('passed', False)) or item.get('resource_abort') is None
            ]
            manifest['failed'] = False
            manifest.pop('failed_at_unix', None)
            _write_json_atomic(manifest_path, manifest)
            _write_qualification_status(output_dir, manifest)
        if bool(manifest.get('failed', False)) and not bool(args.retry_failed_qualification):
            _write_qualification_status(output_dir, manifest)
            failure = next(
                (item for item in manifest.get('results', []) if not bool(item.get('passed', False))),
                {},
            )
            raise RuntimeError(
                'Recipe qualification already failed for the current fingerprint: '
                f"seed={failure.get('seed')} reason={failure.get('terminal_reason')}; "
                f'inspect {manifest_path}.'
            )
    else:
        manifest = {
            'schema_version': 'roboassemblybench_recipe_qualification_v1',
            'recipe': args.recipe,
            'scene_profile': args.scene_profile,
            'recipe_fingerprint': fingerprint,
            'selected_seeds': selected_seeds,
            'required_passes': len(selected_seeds),
            'results': [],
            'resource_aborts': [],
            'passed': False,
            'failed': False,
            'started_at_unix': time.time(),
        }
        _write_json_atomic(manifest_path, manifest)
        _write_qualification_status(output_dir, manifest)

    result_by_seed = {int(item['seed']): item for item in manifest.get('results') or []}
    for seed in selected_seeds:
        existing_result = result_by_seed.get(seed)
        if existing_result is not None and bool(existing_result.get('passed', False)):
            continue
        if existing_result is not None and not bool(args.retry_failed_qualification):
            break
        while True:
            _wait_for_resources(args, output_dir, int(args.num_episodes))
            attempt_dir = qualification_root / f'seed_{seed:06d}'
            if attempt_dir.exists():
                attempt_dir = qualification_root / f'seed_{seed:06d}_retry_{int(time.time())}'
            attempt_dir.mkdir(parents=True, exist_ok=False)
            results_path = attempt_dir / 'collect_results.json'
            log_path = attempt_dir / 'worker.log'
            command = _batch_command(
                args,
                seeds=[seed],
                batch_dir=attempt_dir,
                results_path=results_path,
                record_raw=False,
            )
            print(
                f'Qualifying recipe {fingerprint[:12]} with seed={seed} '
                f'({len(result_by_seed) + 1}/{len(selected_seeds)}).',
                flush=True,
            )
            with log_path.open('w', encoding='utf-8') as log_file:
                returncode, resource_abort = _run_worker_with_resource_monitor(
                    command,
                    log_file=log_file,
                    args=args,
                )
            result = _qualification_result(results_path, seed=seed)
            result.update(
                {
                    'returncode': int(returncode),
                    'resource_abort': resource_abort,
                    'log_path': str(log_path.resolve()),
                    'finished_at_unix': time.time(),
                }
            )
            result['passed'] = bool(result['passed'] and returncode == 0 and resource_abort is None)
            if resource_abort is not None:
                manifest.setdefault('resource_aborts', []).append(result)
                manifest['failed'] = False
                manifest.pop('failed_at_unix', None)
                _write_json_atomic(manifest_path, manifest)
                _write_qualification_status(output_dir, manifest)
                print(
                    f'Qualification seed={seed} was stopped by resource protection; retrying after '
                    f'{float(args.resource_wait_seconds):.0f}s.',
                    flush=True,
                )
                time.sleep(float(args.resource_wait_seconds))
                continue

            result_by_seed[seed] = result
            manifest['results'] = [result_by_seed[key] for key in selected_seeds if key in result_by_seed]
            if not result['passed']:
                manifest['failed'] = True
                manifest['failed_at_unix'] = time.time()
                _write_json_atomic(manifest_path, manifest)
                _write_qualification_status(output_dir, manifest)
                raise RuntimeError(
                    f'Recipe qualification failed at seed={seed}: {result["terminal_reason"]}; '
                    f'inspect {log_path}. Formal collection was not started.'
                )
            _write_json_atomic(manifest_path, manifest)
            _write_qualification_status(output_dir, manifest)
            break

    manifest['passed'] = len(result_by_seed) == len(selected_seeds) and all(
        bool(result_by_seed[seed].get('passed', False)) for seed in selected_seeds
    )
    manifest['failed'] = not manifest['passed']
    manifest['finished_at_unix'] = time.time()
    _write_json_atomic(manifest_path, manifest)
    _write_qualification_status(output_dir, manifest)
    if not manifest['passed']:
        raise RuntimeError(f'Recipe qualification is incomplete; inspect {manifest_path}.')
    return manifest


def _wait_for_resources(args, output_dir: Path, remaining_episodes: int) -> None:
    minimum_memory = float(args.min_available_memory_gib)
    wait_seconds = max(float(getattr(args, 'resource_wait_seconds', 30.0)), 1.0)
    while True:
        available_memory = _available_memory_gib()
        if available_memory >= minimum_memory:
            break
        print(
            f'Waiting for memory before the next Isaac batch: {available_memory:.2f} GiB available, '
            f'{minimum_memory:.2f} GiB required.',
            flush=True,
        )
        time.sleep(wait_seconds)
    disk = shutil.disk_usage(output_dir)
    available_disk_gib = disk.free / (1024.0**3)
    estimated_gib = remaining_episodes * float(args.estimated_episode_mib) / 1024.0
    required_gib = estimated_gib + float(args.disk_reserve_gib)
    if available_disk_gib < required_gib:
        raise RuntimeError(
            f'Only {available_disk_gib:.1f} GiB disk is available; estimated remaining data plus reserve '
            f'requires {required_gib:.1f} GiB.'
        )


def _terminate_worker_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_worker_with_resource_monitor(command: list[str], *, log_file, args) -> tuple[int, dict[str, Any] | None]:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stop_signal: int | None = None

    def request_stop(signum, _frame) -> None:
        nonlocal stop_signal
        stop_signal = int(signum)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    low_memory_polls = 0
    resource_abort = None
    poll_seconds = max(float(args.resource_poll_seconds), 1.0)
    abort_threshold = max(float(args.abort_available_memory_gib), 0.0)
    grace_polls = max(int(args.low_memory_grace_polls), 1)
    timeout_seconds = max(float(getattr(args, 'worker_timeout_seconds', 1800.0)), 0.0)
    stall_timeout_seconds = max(float(getattr(args, 'worker_stall_timeout_seconds', 0.0)), 0.0)
    started_at = time.monotonic()
    last_log_activity_at = started_at
    last_log_size = os.fstat(log_file.fileno()).st_size
    try:
        while process.poll() is None:
            if stop_signal is not None:
                _terminate_worker_process_group(process)
                raise SystemExit(128 + stop_signal)
            available_memory = _available_memory_gib()
            if abort_threshold > 0.0 and available_memory < abort_threshold:
                low_memory_polls += 1
            else:
                low_memory_polls = 0
            if low_memory_polls >= grace_polls:
                resource_abort = {
                    'reason': 'low-available-memory',
                    'available_memory_gib': available_memory,
                    'threshold_gib': abort_threshold,
                    'consecutive_polls': low_memory_polls,
                }
            else:
                now = time.monotonic()
                log_size = os.fstat(log_file.fileno()).st_size
                if log_size != last_log_size:
                    last_log_size = log_size
                    last_log_activity_at = now
                stalled_seconds = max(now - last_log_activity_at, 0.0)
                elapsed_seconds = max(now - started_at, 0.0)
                if stall_timeout_seconds > 0.0 and stalled_seconds >= stall_timeout_seconds:
                    resource_abort = {
                        'reason': 'worker-log-stalled',
                        'stalled_seconds': stalled_seconds,
                        'threshold_seconds': stall_timeout_seconds,
                    }
                elif timeout_seconds > 0.0 and elapsed_seconds >= timeout_seconds:
                    resource_abort = {
                        'reason': 'worker-wall-timeout',
                        'elapsed_seconds': elapsed_seconds,
                        'threshold_seconds': timeout_seconds,
                    }
            if resource_abort is not None:
                _terminate_worker_process_group(process)
                break
            time.sleep(poll_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return int(process.wait()), resource_abort


def collect(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qualification = _ensure_recipe_qualified(args, output_dir)
    args.qualification_manifest = str(
        output_dir / 'qualification' / str(args.recipe_fingerprint) / QUALIFICATION_MANIFEST_NAME
    )
    if bool(qualification.get('skipped', False)):
        args.qualification_manifest = ''
    manifest_path = output_dir / MANIFEST_NAME
    expected_recipe_fingerprint = str(args.recipe_fingerprint)
    expected_randomization_profile = normalize_randomization_profile(
        getattr(args, 'randomization_profile', None)
    )
    unique_layout_seeds = bool(getattr(args, 'unique_layout_seeds', False))
    allowed_layout_seeds = None if unique_layout_seeds else {int(seed) for seed in args.layout_seeds}
    existing = _scan_existing(
        output_dir,
        expected_recipe_fingerprint=expected_recipe_fingerprint,
        allowed_layout_seeds=allowed_layout_seeds,
        require_extended_observations=bool(getattr(args, 'require_extended_observations', False)),
        require_visual_quality=bool(getattr(args, 'require_visual_quality', False)),
        expected_randomization_profile=expected_randomization_profile,
    )
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        recorded_fingerprint = str(manifest.get('recipe_fingerprint') or '')
        if recorded_fingerprint and recorded_fingerprint != expected_recipe_fingerprint:
            raise RuntimeError('Collection manifest recipe fingerprint does not match the current resolved recipe.')
        recorded_layout_seeds = [int(seed) for seed in manifest.get('collection_layout_seeds') or []]
        recorded_randomization_profile = normalize_randomization_profile(manifest.get('randomization_profile'))
        if recorded_randomization_profile != expected_randomization_profile:
            raise RuntimeError(
                'Collection manifest randomization profile does not match this run: '
                f'{recorded_randomization_profile!r} != {expected_randomization_profile!r}.'
            )
        expected_layout_assignment = 'episode_seed' if unique_layout_seeds else 'episode_seed_modulo_round_robin'
        if (
            recorded_layout_seeds != list(args.layout_seeds)
            or manifest.get('layout_assignment') != expected_layout_assignment
        ):
            raise RuntimeError(
                'Collection manifest layout seed contract does not match this run: '
                f'{recorded_layout_seeds}/{manifest.get("layout_assignment")} != '
                f'{list(args.layout_seeds)}/{expected_layout_assignment}.'
            )
        manifest['recipe_fingerprint'] = expected_recipe_fingerprint
        _reconcile_manifest_on_resume(manifest, existing)
        manifest['batch_size'] = int(args.batch_size)
        manifest['max_attempts'] = int(args.max_attempts)
        manifest['worker_timeout_seconds'] = float(args.worker_timeout_seconds)
        manifest['worker_stall_timeout_seconds'] = float(args.worker_stall_timeout_seconds)
    else:
        manifest = _initial_manifest(args, existing)
    _write_json_atomic(manifest_path, manifest)

    max_attempt_seed = int(args.start_seed) + int(args.max_attempts)
    while len(manifest['successful_episodes']) < int(args.num_episodes):
        remaining = int(args.num_episodes) - len(manifest['successful_episodes'])
        _wait_for_resources(args, output_dir, remaining)
        next_seed = int(manifest.get('next_seed', args.start_seed))
        if next_seed >= max_attempt_seed:
            raise RuntimeError(
                f'Reached max_attempts={args.max_attempts} with '
                f"{len(manifest['successful_episodes'])}/{args.num_episodes} successful episodes."
            )
        count = min(int(args.batch_size), remaining, max_attempt_seed - next_seed)
        seeds = list(range(next_seed, next_seed + count))
        layout_seeds = (
            list(seeds)
            if unique_layout_seeds
            else [args.layout_seeds[seed % len(args.layout_seeds)] for seed in seeds]
        )
        batch_name = f'batch_{seeds[0]:06d}_{seeds[-1]:06d}'
        batch_dir = output_dir / 'batches' / batch_name
        if batch_dir.exists():
            batch_dir = output_dir / 'batches' / f'{batch_name}_retry_{int(time.time())}'
        batch_dir.mkdir(parents=True, exist_ok=False)
        results_path = batch_dir / 'collect_results.json'
        log_path = batch_dir / 'worker.log'
        command = _batch_command(
            args,
            seeds=seeds,
            layout_seeds=layout_seeds,
            batch_dir=batch_dir,
            results_path=results_path,
        )
        batch_record = {
            'seeds': seeds,
            'layout_seeds': layout_seeds,
            'batch_dir': str(batch_dir),
            'log_path': str(log_path),
            'started_at_unix': time.time(),
            'status': 'running',
        }
        manifest['batches'].append(batch_record)
        manifest['next_seed'] = seeds[-1] + 1
        _write_json_atomic(manifest_path, manifest)
        print(
            f"Starting {batch_name}: seeds={seeds}, collected={len(manifest['successful_episodes'])}/"
            f'{args.num_episodes}',
            flush=True,
        )
        if args.dry_run:
            print(' '.join(command), flush=True)
            batch_record['status'] = 'dry_run'
            _write_json_atomic(manifest_path, manifest)
            break

        with log_path.open('w', encoding='utf-8') as log_file:
            returncode, resource_abort = _run_worker_with_resource_monitor(
                command,
                log_file=log_file,
                args=args,
            )
        batch_record['returncode'] = returncode
        batch_record['finished_at_unix'] = time.time()
        batch_record['status'] = (
            'resource_aborted' if resource_abort is not None else ('completed' if returncode == 0 else 'worker_failed')
        )
        if resource_abort is not None:
            batch_record['resource_abort'] = resource_abort

        batch_quality = []
        for metadata_path in batch_dir.rglob('episode_*_cartesian_raw/metadata.json'):
            quality = _quality_check_episode(
                metadata_path,
                expected_recipe_fingerprint=expected_recipe_fingerprint,
                allowed_layout_seeds=allowed_layout_seeds,
                require_extended_observations=bool(getattr(args, 'require_extended_observations', False)),
                require_visual_quality=bool(getattr(args, 'require_visual_quality', False)),
                expected_randomization_profile=expected_randomization_profile,
            )
            batch_quality.append(quality)
            if quality['valid']:
                manifest['successful_episodes'][str(quality['seed'])] = quality
        batch_record['quality'] = batch_quality

        if bool(getattr(args, 'prune_failed_raw', False)):
            valid_episode_dirs = {
                str(Path(item['metadata_path']).resolve().parent)
                for item in batch_quality
                if item['valid']
            }
            for episode_dir in batch_dir.rglob('episode_*_cartesian_raw'):
                if str(episode_dir.resolve()) not in valid_episode_dirs:
                    shutil.rmtree(episode_dir, ignore_errors=True)

        result_by_seed = {}
        if results_path.is_file():
            result_by_seed = {int(item['seed']): item for item in _load_json(results_path)}
        valid_seeds = {int(item['seed']) for item in batch_quality if item['valid']}
        for seed in seeds:
            if seed in valid_seeds:
                continue
            result = result_by_seed.get(seed, {})
            manifest['failed_attempts'].append(
                {
                    'seed': seed,
                    'layout_seed': layout_seeds[seeds.index(seed)],
                    'batch_dir': str(batch_dir),
                    'returncode': returncode,
                    'terminal_reason': result.get('terminal_reason', 'missing-valid-raw-episode'),
                    'success_diagnostics': result.get('success_diagnostics', []),
                    'resource_abort': resource_abort,
                }
            )
        manifest['num_successful'] = len(manifest['successful_episodes'])
        manifest['num_failed_attempts'] = len(manifest['failed_attempts'])
        _write_json_atomic(manifest_path, manifest)
        print(
            f'Finished {batch_name}: valid={len(valid_seeds)}/{len(seeds)}, total='
            f"{len(manifest['successful_episodes'])}/{args.num_episodes}",
            flush=True,
        )
        if resource_abort is not None and not batch_quality:
            print(
                f'Resource protection stopped {batch_name}; waiting for resources before continuing.',
                flush=True,
            )
            continue
        if returncode != 0 and not batch_quality:
            raise RuntimeError(f'Isaac worker failed for {batch_name}; inspect {log_path}.')

    manifest['complete'] = len(manifest['successful_episodes']) >= int(args.num_episodes)
    manifest['num_successful'] = len(manifest['successful_episodes'])
    manifest['finished_at_unix'] = time.time() if manifest['complete'] else None
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect 2k randomized plumbers-block Cartesian episodes safely.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument(
        '--layout-seeds',
        type=int,
        nargs='+',
        default=None,
        help='Fixed layout seeds assigned round-robin while episode seeds remain unique.',
    )
    parser.add_argument(
        '--unique-layout-seeds',
        action='store_true',
        help='Use each unique episode seed as its layout/domain-randomization seed.',
    )
    parser.add_argument('--max-attempts', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--conda-env', default='internutopia311')
    parser.add_argument(
        '--isaac-python',
        default=None,
        help='Run Isaac workers with this Python executable instead of conda run.',
    )
    parser.add_argument('--recipe', default='fabrica_plumbers_block_ur5e_right_base_prepare')
    parser.add_argument('--scene-profile', default='taoyuan_grscenes_tabletop')
    parser.add_argument(
        '--randomization-profile',
        choices=RANDOMIZATION_PROFILE_CHOICES,
        default='mixed',
        help='Select the extra visual domain while retaining position randomization.',
    )
    parser.add_argument('--dataset-fps', type=int, default=30)
    parser.add_argument('--dataset-frame-stride', type=int, default=8)
    parser.add_argument(
        '--rendering-fps',
        type=int,
        default=240,
        help='Isaac render/control cadence; keep this independent from the downsampled dataset FPS.',
    )
    parser.add_argument('--min-available-memory-gib', type=float, default=5.5)
    parser.add_argument('--abort-available-memory-gib', type=float, default=1.5)
    parser.add_argument('--resource-poll-seconds', type=float, default=5.0)
    parser.add_argument('--resource-wait-seconds', type=float, default=30.0)
    parser.add_argument('--low-memory-grace-polls', type=int, default=3)
    parser.add_argument(
        '--worker-timeout-seconds',
        type=float,
        default=1800.0,
        help='Terminate and reject an Isaac worker that does not exit within this wall time; 0 disables.',
    )
    parser.add_argument(
        '--worker-stall-timeout-seconds',
        type=float,
        default=0.0,
        help='Terminate an Isaac worker whose log has not grown for this duration; 0 disables.',
    )
    parser.add_argument('--estimated-episode-mib', type=float, default=64.0)
    parser.add_argument('--disk-reserve-gib', type=float, default=80.0)
    parser.add_argument('--qualification-seed-count', type=int, default=None)
    parser.add_argument('--qualification-candidate-pool', type=int, default=128)
    parser.add_argument('--qualification-start-seed', type=int, default=0)
    parser.add_argument('--qualification-required-seeds', type=int, nargs='*', default=None)
    parser.add_argument(
        '--retry-failed-qualification',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--skip-qualification', action='store_true')
    parser.add_argument(
        '--require-extended-observations',
        action='store_true',
        help='Require synchronized RGB-D, rich robot state, and long-horizon annotations.',
    )
    parser.add_argument(
        '--require-visual-quality',
        action='store_true',
        help='Reject episodes with a missing scene, invalid front render, or invisible robot.',
    )
    parser.add_argument(
        '--prune-failed-raw',
        action='store_true',
        help='Delete RGB-D payloads for episodes rejected by quality checks while preserving logs.',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.recipe_spec = load_task_recipe(args.recipe, scene_profile=args.scene_profile)
    args.recipe_fingerprint = str(args.recipe_spec['recipe_fingerprint'])
    qualification_spec = args.recipe_spec.get('qualification') or {}
    collection_spec = args.recipe_spec.get('collection') or {}
    if args.qualification_seed_count is None:
        args.qualification_seed_count = int(qualification_spec.get('seed_count', 4))
    if args.qualification_required_seeds is None:
        args.qualification_required_seeds = [int(seed) for seed in qualification_spec.get('required_seeds', [17])]
    if args.unique_layout_seeds:
        args.layout_seeds = []
    elif args.layout_seeds is None:
        args.layout_seeds = [int(seed) for seed in collection_spec.get('layout_seeds', [])]
    if not args.unique_layout_seeds and not args.layout_seeds:
        parser.error('--layout-seeds requires at least one seed or recipe collection.layout_seeds.')
    if len(args.layout_seeds) != len(set(args.layout_seeds)):
        parser.error('--layout-seeds values must be unique.')

    if (
        args.num_episodes <= 0
        or args.batch_size <= 0
        or args.max_attempts < args.num_episodes
        or args.dataset_fps <= 0
        or args.dataset_frame_stride <= 0
        or args.rendering_fps <= 0
        or args.resource_wait_seconds <= 0
        or args.worker_timeout_seconds < 0
        or args.worker_stall_timeout_seconds < 0
        or args.qualification_seed_count < 0
        or args.qualification_candidate_pool <= 0
    ):
        parser.error(
            'num-episodes, batch-size, dataset-fps, dataset-frame-stride, and rendering-fps must be '
            'positive; qualification-seed-count must be non-negative; qualification-candidate-pool '
            'and resource-wait-seconds must be positive; worker timeouts must be non-negative; '
            'max-attempts must cover num-episodes.'
        )
    with _exclusive_collection_lock(Path(args.output_dir).resolve()):
        manifest = collect(args)
    print(
        json.dumps(
            {
                'complete': manifest.get('complete', False),
                'num_successful': manifest.get('num_successful', 0),
                'num_failed_attempts': manifest.get('num_failed_attempts', 0),
                'manifest': str(Path(args.output_dir).resolve() / MANIFEST_NAME),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
