from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.util import has_display
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions

from toolkits.factory_dual_franka_assembly.demo_policy import DualFrankaAssemblyDemoPolicy
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_batch
from toolkits.factory_dual_franka_assembly.scene_profiles import DEFAULT_SCENE_PROFILE, list_scene_profiles
from toolkits.factory_dual_franka_assembly.task_specs import list_task_recipes, load_task_recipe
from roboassemblybench.robobrain.runtime_monitor import RuntimeRoboChecker


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _sanitize_episode_value(value: Any, *, array_size_limit: int = 64):
    if isinstance(value, dict):
        return {key: _sanitize_episode_value(item, array_size_limit=array_size_limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > array_size_limit and all(isinstance(item, (int, float, bool, np.number)) for item in value):
            return {
                '__type__': 'sequence',
                'length': len(value),
                'preview': _to_jsonable(list(value[: min(8, len(value))])),
            }
        return [_sanitize_episode_value(item, array_size_limit=array_size_limit) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > array_size_limit:
            return {
                '__type__': 'ndarray',
                'shape': list(value.shape),
                'dtype': str(value.dtype),
            }
        return value.tolist()
    if hasattr(value, 'shape') and hasattr(value, 'dtype') and getattr(value, 'size', 0) > array_size_limit:
        return {
            '__type__': type(value).__name__,
            'shape': list(getattr(value, 'shape', ())),
            'dtype': str(getattr(value, 'dtype', 'unknown')),
        }
    return _to_jsonable(value)


def _task_export_metadata(task) -> dict:
    config = task.config
    return {
        'prompt': config.prompt,
        'task_description': getattr(config, 'task_description', '') or config.prompt,
        'annotation_name': getattr(config, 'annotation_name', ''),
        'annotation_path': getattr(config, 'annotation_path', None),
        'annotation_title': getattr(config, 'annotation_title', ''),
        'annotation_summary': getattr(config, 'annotation_summary', ''),
        'annotation_description': getattr(config, 'annotation_description', ''),
        'annotation_tags': _to_jsonable(getattr(config, 'annotation_tags', [])),
        'annotation_metadata': _to_jsonable(getattr(config, 'annotation_metadata', {})),
        'annotation_object_roles': _to_jsonable(getattr(config, 'annotation_object_roles', {})),
        'annotation_target_roles': _to_jsonable(getattr(config, 'annotation_target_roles', {})),
        'annotation_phase_notes': _to_jsonable(getattr(config, 'annotation_phase_notes', [])),
        'target_annotations': _to_jsonable(getattr(config, 'target_annotations', {})),
        'phase_annotations': _to_jsonable(getattr(config, 'phase_annotations', [])),
        'metadata': _to_jsonable(getattr(config, 'benchmark_metadata', {})),
        'task_metadata': _to_jsonable(getattr(config, 'task_metadata', {})),
        'scene_profile_metadata': _to_jsonable(getattr(config, 'scene_profile_metadata', {})),
        'source_benchmark': getattr(config, 'source_benchmark', 'factory_dual_franka_assembly'),
        'source_config_path': getattr(config, 'source_config_path', None),
        'camera_metadata': _to_jsonable(getattr(config, 'camera_metadata', [])),
        'robot_metadata': _to_jsonable(getattr(config, 'robot_metadata', [])),
        'object_metadata': _to_jsonable(getattr(config, 'object_metadata', [])),
    }


class EpisodeRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []

    def record(self, task, obs: dict, actions: dict):
        self.steps.append(
            {
                'phase': task.phase,
                'actions': _sanitize_episode_value(actions),
                'observations': _sanitize_episode_value(obs),
                'objects': _sanitize_episode_value(task.get_tracked_object_states()),
            }
        )

    def save(
        self,
        task,
        episode_idx: int,
        metrics: dict,
        *,
        recorded_videos: dict[str, str] | None = None,
        recorded_video_summaries: dict[str, dict] | None = None,
        recorded_video_mode: str | None = None,
        recorded_frames: dict[str, str] | None = None,
        recorded_frame_summaries: dict[str, dict] | None = None,
        recorded_frame_mode: str | None = None,
    ):
        payload = {
            'episode_idx': episode_idx,
            'seed': task.config.seed,
            'recipe': task.config.recipe,
            'prompt': task.config.prompt,
            'task_description': getattr(task.config, 'task_description', '') or task.config.prompt,
            'scene_profile': getattr(task.config, 'scene_profile', '') or None,
            'spec_path': getattr(task.config, 'spec_path', None),
            'scene_profile_path': getattr(task.config, 'scene_profile_path', None),
            'scene_asset_path': task.config.scene_asset_path,
            'workspace_offset': _to_jsonable(getattr(task.config, 'workspace_offset', [])),
            'asset_references': _to_jsonable(getattr(task.config, 'asset_references', [])),
            **_task_export_metadata(task),
            'phase_blueprint': [phase_spec.get('name') for phase_spec in task.config.phase_specs],
            'metrics': _to_jsonable(metrics),
            'recorded_videos': _to_jsonable(recorded_videos or {}),
            'recorded_video_summaries': _to_jsonable(recorded_video_summaries or {}),
            'recorded_video_mode': recorded_video_mode,
            'recorded_frames': _to_jsonable(recorded_frames or {}),
            'recorded_frame_summaries': _to_jsonable(recorded_frame_summaries or {}),
            'recorded_frame_mode': recorded_frame_mode,
            'steps': self.steps,
        }
        path = self.output_dir / f'episode_{episode_idx:04d}.json'
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


class LiveRolloutVideoRecorder:
    def __init__(
        self,
        *,
        output_dir: Path,
        episode_idx: int,
        task,
        fps: int,
        frame_stride: int,
        keep_frames: bool = False,
    ):
        self.output_dir = output_dir.resolve()
        self.episode_idx = int(episode_idx)
        self.keep_frames = keep_frames
        self.frame_stride = max(int(frame_stride), 1)
        self.fps = max(int(fps), 1)
        self.frames_root_dir = self.output_dir / f'episode_{self.episode_idx:04d}_live_frames'
        self.frames_root_dir.mkdir(parents=True, exist_ok=True)

        from toolkits.factory_dual_franka_assembly.export_lerobot import RunningImageStats, _episode_camera_specs

        self._running_image_stats_cls = RunningImageStats
        self.camera_specs = _episode_camera_specs(
            {
                'camera_metadata': _to_jsonable(getattr(task.config, 'camera_metadata', [])),
                'robot_metadata': _to_jsonable(getattr(task.config, 'robot_metadata', [])),
                'steps': [],
            },
            video_mode='isaac_replay',
            default_width=960,
            default_height=540,
            video_key=None,
        )
        self._image_stats = {}
        self._frame_dirs: dict[str, Path] = {}
        self._written_png_counts: dict[str, int] = {}
        self._frame_shapes: dict[str, tuple[int, int]] = {}
        self._sampled_steps: list[dict] = []
        self._camera_bindings = {
            camera_spec['video_key']: {
                'robot_name': str(camera_spec.get('owner') or camera_spec.get('robot') or 'franka_left'),
                'sensor_name': str(camera_spec.get('name') or ''),
            }
            for camera_spec in self.camera_specs
        }

    @staticmethod
    def _to_uint8_rgb(frame) -> np.ndarray | None:
        if frame is None:
            return None
        frame_array = np.asarray(frame)
        if frame_array.size == 0 or frame_array.ndim < 3:
            return None
        if frame_array.shape[-1] >= 4:
            frame_array = frame_array[..., :3]
        elif frame_array.shape[-1] == 1:
            frame_array = np.repeat(frame_array, 3, axis=-1)
        if np.issubdtype(frame_array.dtype, np.floating):
            if float(np.nanmax(frame_array)) <= 1.0 + 1e-6:
                frame_array = frame_array * 255.0
        frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(frame_array, 0.0, 255.0).astype(np.uint8)

    def _sensor_frame_from_obs(self, obs: dict, *, camera_video_key: str) -> np.ndarray | None:
        binding = self._camera_bindings.get(camera_video_key, {})
        robot_name = binding.get('robot_name')
        sensor_name = binding.get('sensor_name')
        if not robot_name or not sensor_name:
            return None

        robot_obs = obs.get(robot_name)
        if not isinstance(robot_obs, dict):
            return None
        sensor_obs = (robot_obs.get('sensors') or {}).get(sensor_name)
        if not isinstance(sensor_obs, dict):
            return None
        return self._to_uint8_rgb(sensor_obs.get('rgba'))

    def record(self, task, obs: dict, *, step_index: int):
        if step_index % self.frame_stride != 0:
            return

        step_payload = {
            'observations': _sanitize_episode_value(obs),
            'objects': _sanitize_episode_value(task.get_tracked_object_states()),
        }
        captured_any = False
        for camera_spec in self.camera_specs:
            camera_video_key = camera_spec['video_key']
            frame_rgb = self._sensor_frame_from_obs(obs, camera_video_key=camera_video_key)
            if frame_rgb is None:
                continue

            frame_dir = self._frame_dirs.get(camera_video_key)
            if frame_dir is None:
                frame_dir = self.frames_root_dir / camera_video_key.replace('.', '_')
                frame_dir.mkdir(parents=True, exist_ok=True)
                self._frame_dirs[camera_video_key] = frame_dir
                self._image_stats[camera_video_key] = self._running_image_stats_cls()
                self._written_png_counts[camera_video_key] = 0
                self._frame_shapes[camera_video_key] = (int(frame_rgb.shape[1]), int(frame_rgb.shape[0]))

            self._image_stats[camera_video_key].update(frame_rgb)
            frame_index = self._written_png_counts[camera_video_key]
            frame_path = frame_dir / f'frame_{frame_index:05d}.png'
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            self._written_png_counts[camera_video_key] += 1
            captured_any = True

        if captured_any:
            self._sampled_steps.append(step_payload)

    def finalize(self):
        summaries = {}
        for camera_video_key, frame_dir in list(self._frame_dirs.items()):
            width, height = self._frame_shapes.get(camera_video_key, (0, 0))
            summaries[camera_video_key] = {
                'frames_dir': str(frame_dir),
                'fps': int(self.fps),
                'width': int(width),
                'height': int(height),
                'renderer': 'live_rollout_frame_capture',
                'frame_count': int(self._written_png_counts.get(camera_video_key, 0)),
                'image_stats': self._image_stats[camera_video_key].to_dict(),
            }
        return {
            'frames': {key: str(path) for key, path in self._frame_dirs.items()},
            'frame_summaries': summaries,
            'frame_mode': 'live_rollout_frames',
        }


def _write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding='utf-8')


def _runtime_failed_results(results: list[dict]) -> list[dict]:
    failed_results = []
    for result in results:
        runtime_report = result.get('runtime_robochecker') or {}
        has_runtime_feedback = bool(runtime_report.get('feedback_count') or runtime_report.get('feedback'))
        if not result.get('success') and has_runtime_feedback:
            failed_results.append(result)
    return failed_results


def _build_env(task_configs, headless: bool):
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            native=False,
            webrtc=False,
        ),
        env_num=1,
        metrics_save_path='none',
        task_configs=task_configs,
    )
    import_extensions()
    return Env(config)


def _run_task_sequence(
    task_configs,
    headless: bool,
    output_dir: Path | None = None,
    results_output_path: Path | None = None,
    *,
    record_live_video: bool = False,
    live_video_fps: int = 30,
    live_video_frame_stride: int = 8,
    keep_video_frames: bool = False,
    runtime_robochecker: bool = False,
    runtime_feedback_path: Path | None = None,
    runtime_observation_dir: Path | None = None,
    runtime_checker_stride: int = 8,
    runtime_stop_on_violation: bool = True,
    runtime_capture_rgb: bool = True,
    runtime_rgb_frame_stride: int = 24,
):
    env = _build_env(task_configs=task_configs, headless=headless)
    policy = DualFrankaAssemblyDemoPolicy()
    obs_list, task_cfgs = env.reset()
    results = []
    episode_idx = 0
    recorder = None
    video_recorder = None
    runtime_checker = None
    if runtime_robochecker:
        runtime_checker = RuntimeRoboChecker(
            output_dir=runtime_observation_dir or ((output_dir / 'runtime_observations') if output_dir is not None else None),
            feedback_path=runtime_feedback_path,
            check_stride=runtime_checker_stride,
            capture_rgb=runtime_capture_rgb,
            rgb_frame_stride=runtime_rgb_frame_stride,
            stop_on_violation=runtime_stop_on_violation,
        )

    try:
        # In headless Isaac workers, `SimulationApp.is_running()` may flip to
        # false immediately after reset even though stepping the world still
        # works. Drive rollouts from active tasks instead of viewport state.
        while task_cfgs and task_cfgs[0] is not None and not env.finished():
            task_name = next(iter(env.runner.current_tasks.keys()))
            task = env.runner.current_tasks[task_name]
            if output_dir is not None and recorder is None:
                recorder = EpisodeRecorder(output_dir=output_dir)
            if record_live_video and video_recorder is None:
                video_recorder = LiveRolloutVideoRecorder(
                    output_dir=output_dir or Path.cwd(),
                    episode_idx=episode_idx,
                    task=task,
                    fps=live_video_fps,
                    frame_stride=live_video_frame_stride,
                    keep_frames=keep_video_frames,
                )

            env_actions = policy.act(task)
            obs_list, _, terminated, _, _ = env.step([env_actions])

            if recorder is not None:
                recorder.record(task, obs_list[0], env_actions)
            if video_recorder is not None:
                video_recorder.record(task, obs_list[0], step_index=int(task.step_counter))
            if runtime_checker is not None:
                runtime_result = runtime_checker.observe(
                    task=task,
                    obs=obs_list[0],
                    actions=env_actions,
                    episode_idx=episode_idx,
                )
                if runtime_result.get('blocking'):
                    detail = {
                        'runtime_robochecker': runtime_result.get('feedback', []),
                        'snapshot': runtime_result.get('snapshot', {}),
                    }
                    if hasattr(task, '_set_terminal_state'):
                        task._set_terminal_state(
                            'failed',
                            reason='runtime-robochecker-violation',
                            status='failed',
                            detail=detail,
                        )
                    else:
                        task.failed = True
                        task.success = False
                        task.terminal_reason = 'runtime-robochecker-violation'
                    terminated = list(terminated)
                    terminated[0] = True

            if terminated[0]:
                metrics = task.calculate_metrics()
                if runtime_checker is not None:
                    metrics['runtime_robochecker'] = runtime_checker.finalize()
                results.append(metrics)
                recorded_videos = {}
                recorded_video_summaries = {}
                recorded_video_mode = None
                recorded_frames = {}
                recorded_frame_summaries = {}
                recorded_frame_mode = None
                if video_recorder is not None:
                    live_video_output = video_recorder.finalize()
                    recorded_videos = live_video_output.get('videos', {})
                    recorded_video_summaries = live_video_output.get('summaries', {})
                    recorded_video_mode = live_video_output.get('video_mode')
                    recorded_frames = live_video_output.get('frames', {})
                    recorded_frame_summaries = live_video_output.get('frame_summaries', {})
                    recorded_frame_mode = live_video_output.get('frame_mode')
                if recorder is not None:
                    recorder.save(
                        task=task,
                        episode_idx=episode_idx,
                        metrics=metrics,
                        recorded_videos=recorded_videos,
                        recorded_video_summaries=recorded_video_summaries,
                        recorded_video_mode=recorded_video_mode,
                        recorded_frames=recorded_frames,
                        recorded_frame_summaries=recorded_frame_summaries,
                        recorded_frame_mode=recorded_frame_mode,
                    )
                episode_idx += 1
                recorder = None
                video_recorder = None
                obs_list, task_cfgs = env.reset([0])
                if not task_cfgs or task_cfgs[0] is None:
                    break
        if results_output_path is not None:
            _write_json(results_output_path, results)
    finally:
        if runtime_checker is not None:
            try:
                runtime_checker.finalize()
            except Exception:
                pass
        if video_recorder is not None:
            try:
                video_recorder.finalize()
            except Exception:
                pass
        env.close()

    return results


def search_successful_seeds(
    recipe: str,
    scene_profile: str | None,
    num_demos: int,
    start_seed: int,
    max_trials: int,
    headless: bool,
):
    candidate_seeds = list(range(start_seed, start_seed + max_trials))
    results = _run_task_sequence(
        task_configs=build_dual_franka_assembly_batch(
            recipe=recipe,
            seeds=candidate_seeds,
            scene_profile=scene_profile,
            attach_runtime_cameras=False,
        ),
        headless=headless,
    )
    success_seeds = [result['seed'] for result in results if result['success']]
    return success_seeds[:num_demos]


def collect_demos(recipe: str, scene_profile: str | None, seeds, output_dir: Path, headless: bool):
    task_configs = build_dual_franka_assembly_batch(
        recipe=recipe,
        seeds=seeds,
        scene_profile=scene_profile,
        attach_runtime_cameras=False,
    )
    return _run_task_sequence(task_configs=task_configs, headless=headless, output_dir=output_dir)


def _worker_mode(args, *, headless: bool):
    scene_profile = None if args.worker_scene_profile in {None, '', 'raw', 'none'} else args.worker_scene_profile
    if args.worker_mode == 'search':
        _run_task_sequence(
            task_configs=build_dual_franka_assembly_batch(
                recipe=args.worker_recipe,
                seeds=list(range(args.start_seed, args.start_seed + args.max_trials)),
                scene_profile=scene_profile,
                attach_runtime_cameras=bool(args.runtime_capture_rgb),
            ),
            headless=headless,
            results_output_path=Path(args.worker_results_path).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            runtime_robochecker=bool(args.runtime_robochecker),
            runtime_feedback_path=None
            if args.runtime_feedback_path is None
            else Path(args.runtime_feedback_path).resolve(),
            runtime_observation_dir=None
            if args.runtime_observation_dir is None
            else Path(args.runtime_observation_dir).resolve(),
            runtime_checker_stride=max(int(args.runtime_checker_stride), 1),
            runtime_stop_on_violation=bool(args.runtime_stop_on_violation),
            runtime_capture_rgb=bool(args.runtime_capture_rgb),
            runtime_rgb_frame_stride=max(int(args.runtime_rgb_frame_stride), 1),
        )
        return

    if args.worker_mode != 'collect':
        raise ValueError(f'Unsupported worker mode: {args.worker_mode!r}')

    if not args.worker_seeds:
        raise ValueError('Collect worker requires at least one seed.')

    _run_task_sequence(
        task_configs=build_dual_franka_assembly_batch(
            recipe=args.worker_recipe,
            seeds=[int(seed) for seed in args.worker_seeds],
            scene_profile=scene_profile,
            attach_runtime_cameras=bool(args.record_live_video or args.runtime_capture_rgb),
        ),
        headless=headless,
        output_dir=Path(args.output_dir).resolve(),
        results_output_path=Path(args.worker_results_path).resolve(),
        record_live_video=bool(args.record_live_video),
        live_video_fps=max(int(args.live_video_fps), 1),
        live_video_frame_stride=max(int(args.live_video_frame_stride), 1),
        keep_video_frames=bool(args.keep_video_frames),
        runtime_robochecker=bool(args.runtime_robochecker),
        runtime_feedback_path=None if args.runtime_feedback_path is None else Path(args.runtime_feedback_path).resolve(),
        runtime_observation_dir=None
        if args.runtime_observation_dir is None
        else Path(args.runtime_observation_dir).resolve(),
        runtime_checker_stride=max(int(args.runtime_checker_stride), 1),
        runtime_stop_on_violation=bool(args.runtime_stop_on_violation),
        runtime_capture_rgb=bool(args.runtime_capture_rgb),
        runtime_rgb_frame_stride=max(int(args.runtime_rgb_frame_stride), 1),
    )


def _invoke_worker(
    *,
    mode: str,
    recipe: str,
    scene_profile: str | None,
    headless: bool,
    output_dir: Path,
    results_path: Path,
    start_seed: int,
    max_trials: int,
    seeds: list[int] | None = None,
    record_live_video: bool = False,
    live_video_fps: int = 30,
    live_video_frame_stride: int = 8,
    keep_video_frames: bool = False,
    runtime_robochecker: bool = False,
    runtime_feedback_path: Path | None = None,
    runtime_observation_dir: Path | None = None,
    runtime_checker_stride: int = 8,
    runtime_stop_on_violation: bool = True,
    runtime_capture_rgb: bool = True,
    runtime_rgb_frame_stride: int = 24,
):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--worker-mode',
        mode,
        '--worker-recipe',
        recipe,
        '--worker-results-path',
        str(results_path),
        '--output-dir',
        str(output_dir),
        '--start-seed',
        str(start_seed),
        '--max-trials',
        str(max_trials),
    ]
    if headless:
        command.append('--headless')
    if record_live_video:
        command.append('--record-live-video')
        command.extend(['--live-video-fps', str(int(live_video_fps))])
        command.extend(['--live-video-frame-stride', str(int(live_video_frame_stride))])
    if keep_video_frames:
        command.append('--keep-video-frames')
    if runtime_robochecker:
        command.append('--runtime-robochecker')
        command.extend(['--runtime-checker-stride', str(int(runtime_checker_stride))])
        command.extend(['--runtime-rgb-frame-stride', str(int(runtime_rgb_frame_stride))])
        if runtime_feedback_path is not None:
            command.extend(['--runtime-feedback-path', str(runtime_feedback_path)])
        if runtime_observation_dir is not None:
            command.extend(['--runtime-observation-dir', str(runtime_observation_dir)])
        if runtime_stop_on_violation:
            command.append('--runtime-stop-on-violation')
        if runtime_capture_rgb:
            command.append('--runtime-capture-rgb')
    if scene_profile is not None:
        command.extend(['--worker-scene-profile', scene_profile])
    if seeds:
        command.extend(['--worker-seeds', *[str(seed) for seed in seeds]])
    subprocess.run(command, check=True)


def _results_from_worker(
    *,
    mode: str,
    recipe: str,
    scene_profile: str | None,
    output_dir: Path,
    worker_dir: Path,
    headless: bool,
    start_seed: int,
    max_trials: int,
    seeds: list[int] | None = None,
    record_live_video: bool = False,
    live_video_fps: int = 30,
    live_video_frame_stride: int = 8,
    keep_video_frames: bool = False,
    runtime_robochecker: bool = False,
    runtime_feedback_path: Path | None = None,
    runtime_observation_dir: Path | None = None,
    runtime_checker_stride: int = 8,
    runtime_stop_on_violation: bool = True,
    runtime_capture_rgb: bool = True,
    runtime_rgb_frame_stride: int = 24,
) -> list[dict]:
    worker_dir.mkdir(parents=True, exist_ok=True)
    results_path = worker_dir / (
        f'{_safe_filename_component(_output_profile_name(scene_profile))}'
        f'__{_safe_filename_component(recipe)}__{mode}_results.json'
    )
    if results_path.exists():
        results_path.unlink()
    _invoke_worker(
        mode=mode,
        recipe=recipe,
        scene_profile=scene_profile,
        headless=headless,
        output_dir=output_dir,
        results_path=results_path,
        start_seed=start_seed,
        max_trials=max_trials,
        seeds=seeds,
        record_live_video=record_live_video,
        live_video_fps=live_video_fps,
        live_video_frame_stride=live_video_frame_stride,
        keep_video_frames=keep_video_frames,
        runtime_robochecker=runtime_robochecker,
        runtime_feedback_path=runtime_feedback_path,
        runtime_observation_dir=runtime_observation_dir,
        runtime_checker_stride=runtime_checker_stride,
        runtime_stop_on_violation=runtime_stop_on_violation,
        runtime_capture_rgb=runtime_capture_rgb,
        runtime_rgb_frame_stride=runtime_rgb_frame_stride,
    )
    if not results_path.exists():
        raise RuntimeError(
            f'Worker {mode!r} for recipe {recipe!r} and scene profile {_output_profile_name(scene_profile)!r} '
            f'did not write expected results file: {results_path}'
        )
    return json.loads(results_path.read_text(encoding='utf-8'))


def _normalize_requested_items(requested_items, available_items, default_items):
    if not requested_items:
        return list(default_items)

    normalized = []
    for item in requested_items:
        if item in {'all', '*'}:
            return list(available_items)
        if item in {'none', 'raw'}:
            normalized.append(None)
            continue
        normalized.append(item)
    return normalized


def _output_profile_name(scene_profile: str | None) -> str:
    return scene_profile or 'raw'


def _safe_filename_component(value: str | None) -> str:
    value = str(value or 'raw').replace('\\', '/').rstrip('/')
    if '/' in value:
        path = Path(value)
        parent_name = path.parent.name
        value = f'{parent_name}_{path.stem}' if parent_name else path.stem
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')
    return value or 'raw'


def main():
    parser = argparse.ArgumentParser(description='Generate dual-Franka assembly demos in InternUtopia.')
    parser.add_argument('--recipes', nargs='+', default=None)
    parser.add_argument('--scene-profiles', nargs='+', default=None)
    parser.add_argument('--num-demos', type=int, default=2)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--max-trials', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker-mode', choices=['search', 'collect'], default=None)
    parser.add_argument('--worker-recipe', type=str, default=None)
    parser.add_argument('--worker-scene-profile', type=str, default=None)
    parser.add_argument('--worker-results-path', type=str, default=None)
    parser.add_argument('--worker-seeds', nargs='*', default=None)
    parser.add_argument('--record-live-video', action='store_true')
    parser.add_argument('--live-video-fps', type=int, default=30)
    parser.add_argument('--live-video-frame-stride', type=int, default=8)
    parser.add_argument('--keep-video-frames', action='store_true')
    parser.add_argument('--runtime-robochecker', action='store_true')
    parser.add_argument('--runtime-feedback-path', type=str, default=None)
    parser.add_argument('--runtime-observation-dir', type=str, default=None)
    parser.add_argument('--runtime-checker-stride', type=int, default=8)
    parser.add_argument('--runtime-stop-on-violation', action='store_true')
    parser.add_argument('--runtime-capture-rgb', action='store_true')
    parser.add_argument('--runtime-rgb-frame-stride', type=int, default=24)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly'),
    )
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    headless = args.headless or not has_display()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.worker_mode is not None:
        _worker_mode(args, headless=headless)
        return

    recipes = _normalize_requested_items(
        requested_items=args.recipes,
        available_items=list_task_recipes(),
        default_items=list_task_recipes(),
    )
    scene_profiles = _normalize_requested_items(
        requested_items=args.scene_profiles,
        available_items=list_scene_profiles(),
        default_items=[DEFAULT_SCENE_PROFILE],
    )

    overall_manifest = {}
    for scene_profile in scene_profiles:
        profile_key = _output_profile_name(scene_profile)
        overall_manifest[profile_key] = {}
        for recipe in recipes:
            recipe_spec = load_task_recipe(recipe, scene_profile=scene_profile)
            recipe_name = recipe_spec['task_name']
            recipe_dir = output_root / profile_key / recipe_name
            recipe_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = recipe_dir / 'manifest.json'
            if args.resume and manifest_path.exists():
                existing_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if existing_manifest.get('num_collected', 0) >= args.num_demos:
                    overall_manifest[profile_key][recipe_name] = existing_manifest
                    continue

            worker_dir = output_root / '_worker'
            search_results = _results_from_worker(
                mode='search',
                recipe=recipe,
                scene_profile=scene_profile,
                output_dir=recipe_dir,
                worker_dir=worker_dir,
                headless=headless,
                start_seed=args.start_seed,
                max_trials=args.max_trials,
                runtime_robochecker=bool(args.runtime_robochecker),
                runtime_feedback_path=None
                if args.runtime_feedback_path is None
                else Path(args.runtime_feedback_path).resolve(),
                runtime_observation_dir=None
                if args.runtime_observation_dir is None
                else Path(args.runtime_observation_dir).resolve(),
                runtime_checker_stride=max(int(args.runtime_checker_stride), 1),
                runtime_stop_on_violation=bool(args.runtime_stop_on_violation),
                runtime_capture_rgb=bool(args.runtime_capture_rgb),
                runtime_rgb_frame_stride=max(int(args.runtime_rgb_frame_stride), 1),
            )
            successful_seeds = [result['seed'] for result in search_results if result.get('success')][: args.num_demos]
            if not successful_seeds:
                raise RuntimeError(
                    f'No successful seeds found for recipe {recipe_name!r} with scene profile {profile_key!r}. '
                    'Increase max_trials or inspect the task specification.'
                )

            (recipe_dir / 'seed.txt').write_text(' '.join(str(seed) for seed in successful_seeds), encoding='utf-8')
            results = _results_from_worker(
                mode='collect',
                recipe=recipe,
                scene_profile=scene_profile,
                output_dir=recipe_dir,
                worker_dir=worker_dir,
                headless=headless,
                start_seed=args.start_seed,
                max_trials=args.max_trials,
                seeds=successful_seeds,
                record_live_video=bool(args.record_live_video),
                live_video_fps=max(int(args.live_video_fps), 1),
                live_video_frame_stride=max(int(args.live_video_frame_stride), 1),
                keep_video_frames=bool(args.keep_video_frames),
                runtime_robochecker=bool(args.runtime_robochecker),
                runtime_feedback_path=None
                if args.runtime_feedback_path is None
                else Path(args.runtime_feedback_path).resolve(),
                runtime_observation_dir=None
                if args.runtime_observation_dir is None
                else Path(args.runtime_observation_dir).resolve(),
                runtime_checker_stride=max(int(args.runtime_checker_stride), 1),
                runtime_stop_on_violation=bool(args.runtime_stop_on_violation),
                runtime_capture_rgb=bool(args.runtime_capture_rgb),
                runtime_rgb_frame_stride=max(int(args.runtime_rgb_frame_stride), 1),
            )
            runtime_failed_results = _runtime_failed_results(results) if args.runtime_robochecker else []
            if runtime_failed_results:
                _write_json(recipe_dir / 'runtime_failed_results.json', runtime_failed_results)
                failed_seeds = [result.get('seed') for result in runtime_failed_results]
                raise RuntimeError(
                    f'Runtime RoboChecker rejected collected demos for recipe {recipe_name!r}; '
                    f'failed seeds: {failed_seeds}. Inspect runtime_feedback.json and runtime_observations.'
                )
            manifest = {
                'recipe': recipe_name,
                'recipe_request': recipe,
                'scene_profile': scene_profile,
                'scene_profile_key': profile_key,
                'spec_path': recipe_spec.get('spec_path'),
                'scene_profile_path': recipe_spec.get('scene_profile_path'),
                'prompt': recipe_spec.get('prompt'),
                'task_description': recipe_spec.get('task_description') or recipe_spec.get('prompt'),
                'annotation_name': recipe_spec.get('annotation_name', ''),
                'annotation_path': recipe_spec.get('annotation_path'),
                'annotation_title': recipe_spec.get('annotation_title', ''),
                'annotation_summary': recipe_spec.get('annotation_summary', ''),
                'annotation_description': recipe_spec.get('annotation_description', ''),
                'annotation_tags': _to_jsonable(recipe_spec.get('annotation_tags', [])),
                'annotation_metadata': _to_jsonable(recipe_spec.get('annotation_metadata', {})),
                'annotation_object_roles': _to_jsonable(recipe_spec.get('annotation_object_roles', {})),
                'annotation_target_roles': _to_jsonable(recipe_spec.get('annotation_target_roles', {})),
                'annotation_phase_notes': _to_jsonable(recipe_spec.get('annotation_phase_notes', [])),
                'target_annotations': _to_jsonable(recipe_spec.get('target_annotations', {})),
                'phase_annotations': _to_jsonable(recipe_spec.get('phase_annotations', [])),
                'workspace_offset': recipe_spec.get('workspace_offset', [0.0, 0.0, 0.0]),
                'num_requested': args.num_demos,
                'num_collected': len(results),
                'headless': headless,
                'record_live_video': bool(args.record_live_video),
                'live_video_fps': max(int(args.live_video_fps), 1),
                'live_video_frame_stride': max(int(args.live_video_frame_stride), 1),
                'runtime_robochecker': bool(args.runtime_robochecker),
                'runtime_feedback_path': args.runtime_feedback_path,
                'runtime_observation_dir': args.runtime_observation_dir,
                'runtime_checker_stride': max(int(args.runtime_checker_stride), 1),
                'runtime_stop_on_violation': bool(args.runtime_stop_on_violation),
                'runtime_capture_rgb': bool(args.runtime_capture_rgb),
                'runtime_rgb_frame_stride': max(int(args.runtime_rgb_frame_stride), 1),
                'successful_seeds': successful_seeds,
                'asset_references': _to_jsonable(recipe_spec.get('asset_references', [])),
                'metadata': _to_jsonable(recipe_spec.get('metadata', {})),
                'results': _to_jsonable(results),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            overall_manifest[profile_key][recipe_name] = manifest

    (output_root / 'manifest.json').write_text(json.dumps(overall_manifest, indent=2), encoding='utf-8')
    print(json.dumps(overall_manifest, indent=2))


if __name__ == '__main__':
    main()
