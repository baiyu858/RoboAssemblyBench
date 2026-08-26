import json
import socket
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from internutopia.core.config.object import ObjectCfg
from internutopia.core.config.robot import RobotCfg
from internutopia.core.config.sensor import SensorCfg
from internutopia.core.task.task import BaseTask
from internutopia.core.task_config_manager.base import (
    runtime_root_path,
    setup_offset_for_assets,
)
from roboassemblybench.core import task_registry
from roboassemblybench.core.paths import BENCHMARK_ROOT
from roboassemblybench.core.process_lock import (
    exclusive_process_lock,
    process_lock_is_held,
)
from roboassemblybench.core.task_registry import task_recipe_fingerprint
from roboassemblybench.datasets.cartesian_episode import (
    ACTION_NAMES,
    ACTION_SEMANTICS,
    CAMERA_KEYS,
    STATE_NAMES,
)
from roboassemblybench.scripts import collect_fabrica_plumbers_block_2k as collector
from toolkits.factory_dual_franka_assembly import generate_demos


def test_batch_command_keeps_simulation_cadence_independent_from_dataset_fps(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(collector.shutil, 'which', lambda executable: f'/usr/bin/{executable}')
    args = SimpleNamespace(
        conda_env='internutopia311',
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
        dataset_fps=30,
        dataset_frame_stride=8,
        rendering_fps=240,
        randomization_profile='lighting',
    )

    command = collector._batch_command(
        args,
        seeds=[34],
        batch_dir=tmp_path,
        results_path=tmp_path / 'results.json',
    )

    assert command[command.index('--dataset-fps') + 1] == '30'
    assert command[command.index('--dataset-frame-stride') + 1] == '8'
    assert command[command.index('--rendering-fps') + 1] == '240'
    assert command[command.index('--randomization-profile') + 1] == 'lighting'


def test_batch_command_propagates_grasp_debug_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(collector.shutil, 'which', lambda executable: f'/usr/bin/{executable}')
    monkeypatch.setenv('UR5E_DEBUG_GRASP', '1')
    monkeypatch.setenv('UR5E_DEBUG_TRANSPORT_EVERY', '24')
    args = SimpleNamespace(
        conda_env='internutopia311',
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
        dataset_fps=30,
        dataset_frame_stride=8,
        rendering_fps=240,
    )

    command = collector._batch_command(
        args,
        seeds=[34],
        batch_dir=tmp_path,
        results_path=tmp_path / 'results.json',
    )

    assert 'UR5E_DEBUG_GRASP=1' in command
    assert 'UR5E_DEBUG_TRANSPORT_EVERY=24' in command


def test_batch_command_keeps_episode_and_layout_seeds_independent(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(collector.shutil, 'which', lambda executable: f'/usr/bin/{executable}')
    args = SimpleNamespace(
        conda_env='internutopia311',
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
        dataset_fps=30,
        dataset_frame_stride=8,
        rendering_fps=240,
    )

    command = collector._batch_command(
        args,
        seeds=[100, 101, 102, 103],
        layout_seeds=[4906, 485, 34, 12],
        batch_dir=tmp_path,
        results_path=tmp_path / 'results.json',
    )

    layout_start = command.index('--worker-layout-seeds') + 1
    assert command[layout_start : layout_start + 4] == ['4906', '485', '34', '12']


def test_qualification_worker_does_not_record_formal_raw_data(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(collector.shutil, 'which', lambda executable: f'/usr/bin/{executable}')
    args = SimpleNamespace(
        conda_env='internutopia311',
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
        dataset_fps=30,
        dataset_frame_stride=8,
        rendering_fps=240,
    )

    command = collector._batch_command(
        args,
        seeds=[17],
        batch_dir=tmp_path,
        results_path=tmp_path / 'results.json',
        record_raw=False,
    )

    assert '--record-lerobot-raw' not in command


def test_qualification_seed_selection_is_deterministic_and_keeps_hard_seed():
    recipe = collector.load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    kwargs = {
        'count': 8,
        'candidate_pool': 64,
        'start_seed': 0,
        'required_seeds': [17],
    }

    selected = collector._select_qualification_seeds(recipe, **kwargs)

    assert selected == collector._select_qualification_seeds(recipe, **kwargs)
    assert selected[0] == 17
    assert len(selected) == len(set(selected)) == 8


def test_collection_manifest_records_aligned_camera_timing_contract():
    args = SimpleNamespace(
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
        num_episodes=2000,
        max_attempts=4000,
        start_seed=0,
        batch_size=1,
        dataset_fps=30,
        dataset_frame_stride=8,
        rendering_fps=240,
    )

    manifest = collector._initial_manifest(args, {})

    assert manifest['timing_contract'] == {
        'physics_fps': 240,
        'control_fps': 240,
        'dataset_fps': 30,
        'dataset_frame_stride': 8,
        'rendering_interval': 7,
        'camera_render_period_steps': 8,
        'camera_fps': 30.0,
        'camera_state_action_aligned': True,
    }
    assert manifest['worker_timeout_seconds'] == 1800.0


def test_worker_monitor_terminates_a_hung_isaac_process(monkeypatch, tmp_path: Path):
    class FakeProcess:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -15

    fake_process = FakeProcess()
    monotonic_values = iter((100.0, 1900.0))
    signals = []
    monkeypatch.setattr(collector.subprocess, 'Popen', lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(collector.time, 'monotonic', lambda: next(monotonic_values))
    monkeypatch.setattr(collector, '_available_memory_gib', lambda: 8.0)
    monkeypatch.setattr(collector.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    args = SimpleNamespace(
        resource_poll_seconds=5.0,
        abort_available_memory_gib=0.5,
        low_memory_grace_polls=3,
        worker_timeout_seconds=1800.0,
    )

    with (tmp_path / 'worker.log').open('w', encoding='utf-8') as log_file:
        returncode, resource_abort = collector._run_worker_with_resource_monitor(
            ['worker'],
            log_file=log_file,
            args=args,
        )

    assert returncode == -15
    assert resource_abort == {
        'reason': 'worker-wall-timeout',
        'elapsed_seconds': 1800.0,
        'threshold_seconds': 1800.0,
    }
    assert signals == [(1234, collector.signal.SIGTERM)]


def _write_quality_fixture(tmp_path: Path, *, rendering_interval: int) -> Path:
    tmp_path.mkdir(parents=True)
    trajectory_path = tmp_path / 'trajectory.npz'
    states = np.zeros((100, len(STATE_NAMES)), dtype=np.float32)
    states[:, 3] = 1.0
    states[:, 11] = 1.0
    np.savez_compressed(
        trajectory_path,
        observation_state=states,
        action=states.copy(),
        simulation_step=np.arange(100, dtype=np.int64) * 8,
    )
    videos = {}
    for camera_key in CAMERA_KEYS:
        video_path = tmp_path / f'{camera_key}.mp4'
        video_path.write_bytes(b'video')
        videos[camera_key] = str(video_path)
    metadata = {
        'schema_version': 'roboassemblybench_raw_cartesian_v1',
        'recipe_fingerprint': 'current-recipe',
        'seed': 1,
        'scene_profile': 'taoyuan_grscenes_tabletop',
        'scene_asset_path': '/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd',
        'scene_asset_fallback_path': '/benchmark/scenes/usd/factory_cell.usda',
        'scene_asset_source': 'primary',
        'scene_family': 'isaac_simple_warehouse_tabletop',
        'runtime_scene_integrity': {
            'start': {'valid': True},
            'end': {'valid': True},
        },
        'fps': 30,
        'simulation_fps': 240,
        'frame_stride': 8,
        'frame_count': 100,
        'state_names': list(STATE_NAMES),
        'action_names': list(ACTION_NAMES),
        'action_semantics': ACTION_SEMANTICS,
        'trajectory_path': str(trajectory_path),
        'videos': videos,
        'video_frame_counts': {key: 100 for key in CAMERA_KEYS},
        'domain_randomization': {'enabled': True},
        'metrics': {'success': True},
        'timing': {
            'physics_fps': 240,
            'control_fps': 240,
            'dataset_fps': 30,
            'dataset_frame_stride': 8,
            'rendering_interval': rendering_interval,
            'camera_render_period_steps': rendering_interval + 1,
            'camera_state_action_aligned': rendering_interval == 7,
        },
    }
    metadata_path = tmp_path / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    return metadata_path


def _write_visual_test_video(
    path: Path,
    *,
    left_robot_visible: bool = True,
    right_robot_visible: bool = True,
    missing_robot_frames: dict[str, set[int]] | None = None,
    move_arms_to_center: bool = False,
    light_streak: bool = False,
    broad_bright_surface: bool = False,
    temporal_render_artifact: bool = False,
    flat_background_frames: set[int] | None = None,
) -> None:
    width, height = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
    assert writer.isOpened()
    missing_robot_frames = missing_robot_frames or {}
    flat_background_frames = flat_background_frames or set()

    def draw_robot(frame, *, base_center: tuple[int, int], reaches_right: bool) -> None:
        base_x, base_y = base_center
        cv2.ellipse(frame, (base_x, base_y), (22, 12), 0, 0, 360, (238, 238, 238), -1)
        cv2.rectangle(frame, (base_x - 17, base_y - 34), (base_x + 17, base_y), (225, 225, 225), -1)
        cv2.line(frame, (base_x - 17, base_y - 13), (base_x + 17, base_y - 13), (25, 25, 25), 4)
        shoulder = (base_x, base_y - 51)
        cv2.circle(frame, shoulder, 16, (240, 240, 240), -1)
        cv2.circle(frame, shoulder, 16, (35, 35, 35), 3)
        direction = 1 if reaches_right else -1
        elbow = (base_x + direction * 30, base_y - 78)
        wrist = (base_x + direction * 54, base_y - 55)
        cv2.line(frame, shoulder, elbow, (235, 235, 235), 15)
        cv2.line(frame, elbow, wrist, (235, 235, 235), 13)
        cv2.circle(frame, elbow, 10, (35, 35, 35), 3)

    try:
        for frame_index in range(5):
            frame = np.full((height, width, 3), (75, 95, 115), dtype=np.uint8)
            checker = (np.indices((height, width))[0] // 16 + np.indices((height, width))[1] // 16) % 2
            frame[:, int(0.22 * width) : int(0.78 * width)] = np.where(
                checker[:, int(0.22 * width) : int(0.78 * width), None] == 0,
                np.asarray((55, 80, 110), dtype=np.uint8),
                np.asarray((115, 145, 175), dtype=np.uint8),
            )
            if frame_index in flat_background_frames:
                frame[
                    int(0.03 * height) : int(0.38 * height),
                    int(0.25 * width) : int(0.75 * width),
                ] = (75, 95, 115)
            left_visible_in_frame = left_robot_visible and frame_index not in missing_robot_frames.get('left', set())
            right_visible_in_frame = right_robot_visible and frame_index not in missing_robot_frames.get('right', set())
            if left_visible_in_frame:
                draw_robot(
                    frame,
                    base_center=(42, 202),
                    reaches_right=move_arms_to_center or frame_index % 2 == 0,
                )
            if right_visible_in_frame:
                draw_robot(
                    frame,
                    base_center=(211, 198),
                    reaches_right=not (move_arms_to_center or frame_index % 2 == 0),
                )
            if light_streak:
                cv2.line(frame, (0, height // 3), (width - 1, height // 3), (255, 255, 255), 6)
            if broad_bright_surface:
                cv2.rectangle(frame, (0, 24), (width - 1, 84), (255, 255, 255), -1)
            if temporal_render_artifact and frame_index == 1:
                noise = np.random.default_rng(7).integers(-45, 46, size=frame.shape, dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_visual_quality_gate_accepts_visible_robot_and_rejects_missing_robot(tmp_path: Path):
    visible_metadata_path = _write_quality_fixture(tmp_path / 'visible', rendering_interval=7)
    visible_metadata = json.loads(visible_metadata_path.read_text(encoding='utf-8'))
    visible_front_path = Path(visible_metadata['videos'][collector.FRONT_CAMERA_KEY])
    _write_visual_test_video(visible_front_path)

    visible_quality = collector._quality_check_episode(
        visible_metadata_path,
        require_visual_quality=True,
    )

    assert visible_quality['valid']
    assert (
        visible_quality['visual_quality']['left_robot_roi_edge_p90_median']
        >= collector.ROBOT_ANCHOR_EDGE_P90_THRESHOLD
    )

    missing_metadata_path = _write_quality_fixture(tmp_path / 'missing', rendering_interval=7)
    missing_metadata = json.loads(missing_metadata_path.read_text(encoding='utf-8'))
    missing_front_path = Path(missing_metadata['videos'][collector.FRONT_CAMERA_KEY])
    _write_visual_test_video(
        missing_front_path,
        left_robot_visible=False,
        right_robot_visible=False,
    )

    missing_quality = collector._quality_check_episode(
        missing_metadata_path,
        require_visual_quality=True,
    )

    assert not missing_quality['valid']
    assert 'visual:robot-not-visible' in missing_quality['errors']
    assert 'visual:left-robot-not-visible' in missing_quality['errors']
    assert 'visual:right-robot-not-visible' in missing_quality['errors']


def test_visual_quality_gate_rejects_one_missing_arm_and_light_streak(tmp_path: Path):
    missing_right_metadata_path = _write_quality_fixture(tmp_path / 'missing-right', rendering_interval=7)
    missing_right_metadata = json.loads(missing_right_metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(missing_right_metadata['videos'][collector.FRONT_CAMERA_KEY]),
        right_robot_visible=False,
    )

    missing_right_quality = collector._quality_check_episode(
        missing_right_metadata_path,
        require_visual_quality=True,
    )

    assert not missing_right_quality['valid']
    assert 'visual:right-robot-not-visible' in missing_right_quality['errors']

    streak_metadata_path = _write_quality_fixture(tmp_path / 'light-streak', rendering_interval=7)
    streak_metadata = json.loads(streak_metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(streak_metadata['videos'][collector.FRONT_CAMERA_KEY]),
        light_streak=True,
    )

    streak_quality = collector._quality_check_episode(
        streak_metadata_path,
        require_visual_quality=True,
    )

    assert not streak_quality['valid']
    assert 'visual:light-streak' in streak_quality['errors']


def test_visual_quality_gate_does_not_treat_a_broad_bright_scene_surface_as_a_streak(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'broad-bright-surface', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(metadata['videos'][collector.FRONT_CAMERA_KEY]),
        broad_bright_surface=True,
    )

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert 'visual:light-streak' not in quality['errors']


def test_visual_quality_gate_uses_typical_background_frame_instead_of_single_flat_frame(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'one-flat-background-frame', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(metadata['videos'][collector.FRONT_CAMERA_KEY]),
        flat_background_frames={0},
    )

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert quality['visual_quality']['background_edge_mean_min'] < collector.BACKGROUND_EDGE_MEAN_THRESHOLD
    assert quality['visual_quality']['background_edge_mean_median'] >= collector.BACKGROUND_EDGE_MEAN_THRESHOLD
    assert 'visual:factory-background-not-visible' not in quality['errors']


def test_visual_quality_gate_allows_shared_workspace_motion_and_only_graces_first_two_frames(tmp_path: Path):
    moving_metadata_path = _write_quality_fixture(tmp_path / 'moving-arms', rendering_interval=7)
    moving_metadata = json.loads(moving_metadata_path.read_text(encoding='utf-8'))
    moving_front_path = Path(moving_metadata['videos'][collector.FRONT_CAMERA_KEY])
    _write_visual_test_video(moving_front_path, move_arms_to_center=True)

    moving_quality = collector._quality_check_episode(
        moving_metadata_path,
        require_visual_quality=True,
    )

    assert moving_quality['valid']
    assert moving_quality['visual_quality']['left_robot_visible_fraction'] == 1.0
    assert moving_quality['visual_quality']['right_robot_visible_fraction'] == 1.0

    warmup_metadata_path = _write_quality_fixture(tmp_path / 'warmup-grace', rendering_interval=7)
    warmup_metadata = json.loads(warmup_metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(warmup_metadata['videos'][collector.FRONT_CAMERA_KEY]),
        missing_robot_frames={'left': {0, 1}, 'right': {0, 1}},
        move_arms_to_center=True,
    )
    warmup_quality = collector._quality_check_episode(warmup_metadata_path, require_visual_quality=True)
    assert warmup_quality['valid']

    missing_metadata_path = _write_quality_fixture(tmp_path / 'missing-after-warmup', rendering_interval=7)
    missing_metadata = json.loads(missing_metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(missing_metadata['videos'][collector.FRONT_CAMERA_KEY]),
        missing_robot_frames={'right': {2}},
        move_arms_to_center=True,
    )
    missing_quality = collector._quality_check_episode(missing_metadata_path, require_visual_quality=True)
    assert not missing_quality['valid']
    assert 'visual:right-robot-not-visible' in missing_quality['errors']


def test_visual_quality_gate_checks_early_frames_for_temporal_render_artifacts(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'temporal-artifact', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(
        Path(metadata['videos'][collector.FRONT_CAMERA_KEY]),
        temporal_render_artifact=True,
    )

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert not quality['valid']
    assert quality['visual_quality']['sample_frame_indices'][:4] == [0, 1, 2, 3]
    assert 'visual:temporal-render-artifact' in quality['errors']


def test_runtime_camera_warmup_renders_without_advancing_physics_and_refreshes_observations():
    class FakeEnv:
        def __init__(self):
            self.warmup_calls = []

        def warm_up(self, *, steps, render, physics):
            self.warmup_calls.append((steps, render, physics))

        def get_observations(self):
            return [{'front': 'converged'}]

    env = FakeEnv()

    observations = generate_demos._warm_up_runtime_cameras(env)

    assert env.warmup_calls == [(generate_demos.RUNTIME_CAMERA_WARMUP_RENDER_STEPS, True, False)]
    assert observations == [{'front': 'converged'}]


def test_kinematic_replay_frame_advances_physics_before_reading_camera(monkeypatch):
    events = []

    class FakeEnv:
        def warm_up(self, *, steps, render, physics):
            events.append(('warm_up', steps, render, physics))

        def get_observations(self):
            events.append(('get_observations',))
            return [{'front': 'updated'}]

    monkeypatch.setattr(
        generate_demos,
        '_apply_kinematic_replay_frame',
        lambda task, trajectory, frame_index: events.append(('apply', task, trajectory, frame_index)),
    )

    observations = generate_demos._render_kinematic_replay_frame(FakeEnv(), 'task', 'trajectory', 7)

    assert events == [
        ('apply', 'task', 'trajectory', 7),
        ('warm_up', 1, True, True),
        ('get_observations',),
    ]
    assert observations == [{'front': 'updated'}]


def test_visual_quality_gate_rejects_static_rendered_replay(tmp_path: Path):
    static_metadata_path = _write_quality_fixture(tmp_path / 'static-replay', rendering_interval=7)
    static_metadata = json.loads(static_metadata_path.read_text(encoding='utf-8'))
    static_metadata['recording_mode'] = 'rendered_replay'
    static_metadata_path.write_text(json.dumps(static_metadata), encoding='utf-8')
    _write_visual_test_video(
        Path(static_metadata['videos'][collector.FRONT_CAMERA_KEY]),
        move_arms_to_center=True,
    )

    static_quality = collector._quality_check_episode(static_metadata_path, require_visual_quality=True)

    assert not static_quality['valid']
    assert 'visual:static-replay' in static_quality['errors']

    moving_metadata_path = _write_quality_fixture(tmp_path / 'moving-replay', rendering_interval=7)
    moving_metadata = json.loads(moving_metadata_path.read_text(encoding='utf-8'))
    moving_metadata['recording_mode'] = 'rendered_replay'
    moving_metadata_path.write_text(json.dumps(moving_metadata), encoding='utf-8')
    _write_visual_test_video(Path(moving_metadata['videos'][collector.FRONT_CAMERA_KEY]))

    moving_quality = collector._quality_check_episode(moving_metadata_path, require_visual_quality=True)

    assert 'visual:static-replay' not in moving_quality['errors']


def test_two_stage_pipeline_keeps_position_rgbd_and_exports_six_groups():
    script_path = (
        Path(__file__).resolve().parents[2]
        / 'roboassemblybench'
        / 'scripts'
        / 'collect_fabrica_7tasks_50k_twostage_multigpu.sh'
    )
    script = script_path.read_text(encoding='utf-8')

    assert 'TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"' in script
    assert 'DATA_PROFILES=(position "${REPLAY_PROFILES[@]}")' in script
    stage1 = script[script.index('run_stage1_shard()') : script.index('wait_for_stage1_barrier()')]
    assert '--trajectory-only' not in stage1
    assert '--require-extended-observations' in stage1
    assert '--require-visual-quality' in stage1
    assert 'raw="$OUTPUT_ROOT/stage1/$task"' in script


def test_two_stage_pipeline_replays_successful_demos_before_stage1_shard_completes():
    script_path = (
        Path(__file__).resolve().parents[2]
        / 'roboassemblybench'
        / 'scripts'
        / 'collect_fabrica_7tasks_50k_twostage_multigpu.sh'
    )
    script = script_path.read_text(encoding='utf-8')
    replay_loop = script[script.index('pipeline_replay_loop()') : script.index('lerobot_complete()')]

    assert 'PIPELINED_REPLAY="${PIPELINED_REPLAY:-1}"' in script
    assert 'PIPELINE_REPLAY_WORKERS_PER_GPU="${PIPELINE_REPLAY_WORKERS_PER_GPU:-1}"' in script
    assert 'stage1_shard_has_unreplayed_success()' in replay_loop
    assert 'source_seeds - replayed_seeds' in replay_loop
    assert 'if ! stage1_shard_has_unreplayed_success "$task" "$profile" "$shard_id"' in replay_loop
    assert 'run_replay_job "$gpu" "$worker_id" "$task" "$profile" "$shard_id"' in replay_loop
    assert replay_loop.index('stage1_shard_has_unreplayed_success') < replay_loop.index('run_replay_job')
    pipeline_loop = replay_loop[replay_loop.index('pipeline_replay_loop()') : replay_loop.index('replay_shard_complete()')]
    assert 'stage1_shard_complete' not in pipeline_loop
    assert 'wait_for_stage1_barrier' not in replay_loop


def test_two_stage_pipeline_can_pause_stage1_and_assign_all_slots_to_replay():
    root = Path(__file__).resolve().parents[2]
    collector_script = (
        root / 'roboassemblybench' / 'scripts' / 'collect_fabrica_7tasks_50k_twostage_multigpu.sh'
    ).read_text(encoding='utf-8')
    launcher_script = (
        root / 'roboassemblybench' / 'scripts' / 'launch_fabrica_ur5e_60060_autodl.sh'
    ).read_text(encoding='utf-8')

    assert 'STAGE1_COLLECTION_ENABLED="${STAGE1_COLLECTION_ENABLED:-1}"' in collector_script
    launch_block = collector_script[collector_script.index('cd "$REPO_ROOT"') :]
    assert 'if (( STAGE1_COLLECTION_ENABLED == 1 )); then' in launch_block
    assert 'STAGE1_COLLECTION_ENABLED="${STAGE1_COLLECTION_ENABLED:-0}"' in launcher_script
    assert 'PIPELINE_REPLAY_WORKERS_PER_GPU="${PIPELINE_REPLAY_WORKERS_PER_GPU:-3}"' in launcher_script
    assert 'PIPELINE_REPLAY_RAMP_SECONDS="${PIPELINE_REPLAY_RAMP_SECONDS:-30}"' in launcher_script
    assert 'ISAACSIM_PORTABLE_BASE="${ISAACSIM_PORTABLE_BASE:-$PROJECT_ROOT/runtime/isaacsim_portable}"' in launcher_script


def test_visual_quality_gate_rejects_empty_scene_metadata(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'empty-scene', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    front_path = Path(metadata['videos'][collector.FRONT_CAMERA_KEY])
    _write_visual_test_video(front_path)
    metadata['scene_asset_path'] = '/assets/scenes/empty.usd'
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert not quality['valid']
    assert 'visual:scene-asset' in quality['errors']


def test_visual_quality_gate_rejects_factory_fallback_and_invalid_runtime_prims(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'fallback-scene', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(Path(metadata['videos'][collector.FRONT_CAMERA_KEY]))
    metadata['scene_asset_source'] = 'fallback'
    metadata['scene_asset_path'] = metadata['scene_asset_fallback_path']
    metadata['scene_family'] = 'roboassemblybench_factory_cell_tabletop'
    metadata['runtime_scene_integrity']['end']['valid'] = False
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert not quality['valid']
    assert 'visual:scene-source' in quality['errors']
    assert 'visual:runtime-scene-integrity' in quality['errors']


def test_visual_quality_gate_accepts_declared_randomized_factory_scene(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'randomized-scene', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(Path(metadata['videos'][collector.FRONT_CAMERA_KEY]))
    scene_path = '/Isaac/Environments/Grid/default_environment.usd'
    metadata['scene_asset_path'] = scene_path
    metadata['scene_family'] = 'isaac_grid_factory_tabletop'
    metadata['domain_randomization'] = {
        'enabled': True,
        'profile': 'scene',
        'scene': {'asset_path': scene_path, 'variant': 'isaac_grid'},
    }
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert 'visual:scene-asset' not in quality['errors']
    assert 'visual:scene-family' not in quality['errors']


def test_visual_quality_gate_expands_scene_asset_environment_placeholders(tmp_path: Path, monkeypatch):
    metadata_path = _write_quality_fixture(tmp_path / 'expanded-randomized-scene', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _write_visual_test_video(Path(metadata['videos'][collector.FRONT_CAMERA_KEY]))
    assets_root = tmp_path / 'isaac-assets'
    scene_suffix = Path('Isaac/Environments/Simple_Warehouse/warehouse.usd')
    metadata['scene_asset_path'] = str(assets_root / scene_suffix)
    metadata['scene_family'] = 'isaac_simple_warehouse_tabletop'
    metadata['domain_randomization'] = {
        'enabled': True,
        'profile': 'scene',
        'scene': {'asset_path': f'${{ISAAC_ASSETS_ROOT}}/{scene_suffix.as_posix()}'},
    }
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setenv('ISAAC_ASSETS_ROOT', str(assets_root))

    quality = collector._quality_check_episode(metadata_path, require_visual_quality=True)

    assert 'visual:scene-asset' not in quality['errors']


def test_extended_quality_allows_dome_plus_four_randomized_lights(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'five-lights', rendering_interval=7)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['domain_randomization'] = {
        'enabled': True,
        'profile': 'lighting',
        'groups': {'start_parts': {}, 'assembly_base': {}},
        'lighting': [
            {
                'name': f'light_{index}',
                'domain_randomization_intensity_multiplier': 0.8 + index * 0.1,
            }
            for index in range(5)
        ],
    }
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

    quality = collector._quality_check_episode(metadata_path, require_extended_observations=True)

    assert 'lighting_randomization' not in quality['errors']


def test_task_cleanup_removes_robot_articulation_prim():
    remove_calls = []
    remove_prim_path_calls = []
    robot_cleanup_calls = []
    task = object.__new__(BaseTask)
    task.objects = {}
    task.robots = {
        'left': SimpleNamespace(
            articulation=SimpleNamespace(name='ur5e_left'),
            config=SimpleNamespace(prim_path='/World/env_0/robots/ur5e_left'),
            cleanup=lambda: robot_cleanup_calls.append('ur5e_left'),
        )
    }
    task._scene = SimpleNamespace(
        remove=lambda *args, **kwargs: remove_calls.append((args, kwargs)),
        remove_prim_path=remove_prim_path_calls.append,
    )

    task.cleanup()

    assert robot_cleanup_calls == ['ur5e_left']
    assert remove_calls == [(('ur5e_left',), {'registry_only': True})]
    assert remove_prim_path_calls == ['/World/env_0/robots/ur5e_left']


def test_serial_episodes_use_distinct_runtime_usd_namespaces():
    def task_config(episode_idx):
        return SimpleNamespace(
            episode_idx=episode_idx,
            robots_root_path='/robots',
            objects_root_path='/objects',
            robots=[
                RobotCfg(
                    name='left',
                    type='robot',
                    prim_path='/ur5e_left',
                    position=[0.0, 0.0, 0.0],
                    sensors=[
                        SensorCfg(
                            name='front',
                            type='camera',
                            prim_path='/World/env_0/cameras/front',
                        ),
                        SensorCfg(
                            name='wrist',
                            type='camera',
                            prim_path='wrist/camera',
                        ),
                    ],
                )
            ],
            objects=[
                ObjectCfg(
                    name='part',
                    type='object',
                    prim_path='/part',
                    position=[0.0, 0.0, 0.0],
                )
            ],
        )

    first = task_config(0)
    second = task_config(1)
    setup_offset_for_assets(first, env_id=0, offset=[0.0, 0.0, 0.0])
    setup_offset_for_assets(second, env_id=0, offset=[0.0, 0.0, 0.0])

    assert runtime_root_path(first, 0) == '/World/env_0/episode_000000'
    assert runtime_root_path(second, 0) == '/World/env_0/episode_000001'
    assert first.robots[0].prim_path == '/World/env_0/episode_000000/robots/ur5e_left'
    assert second.robots[0].prim_path == '/World/env_0/episode_000001/robots/ur5e_left'
    assert first.objects[0].prim_path == '/World/env_0/episode_000000/objects/part'
    assert second.objects[0].prim_path == '/World/env_0/episode_000001/objects/part'
    assert first.robots[0].sensors[0].prim_path == '/World/env_0/episode_000000/cameras/front'
    assert second.robots[0].sensors[0].prim_path == '/World/env_0/episode_000001/cameras/front'
    assert first.robots[0].sensors[1].prim_path == 'wrist/camera'


def test_quality_check_rejects_stale_camera_timing(tmp_path: Path):
    assert collector._quality_check_episode(_write_quality_fixture(tmp_path / 'aligned', rendering_interval=7))['valid']
    quality = collector._quality_check_episode(_write_quality_fixture(tmp_path / 'stale', rendering_interval=5))
    assert not quality['valid']
    assert quality['errors'] == ['timing_contract']


def test_quality_check_rejects_a_stale_recipe_fingerprint(tmp_path: Path):
    metadata_path = _write_quality_fixture(tmp_path / 'fingerprint', rendering_interval=7)
    quality = collector._quality_check_episode(
        metadata_path,
        expected_recipe_fingerprint='new-recipe',
    )
    assert not quality['valid']
    assert quality['errors'] == ['recipe_fingerprint']


def test_resource_gate_waits_for_memory_instead_of_exiting(tmp_path: Path, monkeypatch):
    available_memory = iter((1.25, 4.0))
    sleeps = []
    monkeypatch.setattr(collector, '_available_memory_gib', lambda: next(available_memory))
    monkeypatch.setattr(collector.time, 'sleep', sleeps.append)
    monkeypatch.setattr(
        collector.shutil,
        'disk_usage',
        lambda _path: SimpleNamespace(free=500 * 1024**3),
    )
    args = SimpleNamespace(
        min_available_memory_gib=3.5,
        resource_wait_seconds=7.0,
        estimated_episode_mib=64.0,
        disk_reserve_gib=80.0,
    )

    collector._wait_for_resources(args, tmp_path, remaining_episodes=2000)

    assert sleeps == [7.0]


def test_cgroup_memory_limit_takes_precedence_over_host_memory(tmp_path: Path):
    limit = tmp_path / 'memory.max'
    current = tmp_path / 'memory.current'
    limit.write_text(str(440 * 1024**3), encoding='utf-8')
    current.write_text(str(401 * 1024**3), encoding='utf-8')

    available = collector._cgroup_available_memory_bytes(((limit, current),))

    assert available == 39 * 1024**3


def test_cgroup_memory_available_reclaims_inactive_file_cache(tmp_path: Path):
    limit = tmp_path / 'memory.max'
    current = tmp_path / 'memory.current'
    stat = tmp_path / 'memory.stat'
    limit.write_text(str(440 * 1024**3), encoding='utf-8')
    current.write_text(str(401 * 1024**3), encoding='utf-8')
    stat.write_text(f'inactive_file {120 * 1024**3}\n', encoding='utf-8')

    available = collector._cgroup_available_memory_bytes(
        ((limit, current),),
        (stat,),
    )

    assert available == 159 * 1024**3


def test_collection_lock_rejects_a_second_writer(tmp_path: Path):
    with collector._exclusive_collection_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another collector'):
            with collector._exclusive_collection_lock(tmp_path):
                pass


def test_process_lock_recovers_a_dead_same_host_owner(tmp_path: Path):
    lock_dir = tmp_path / '.collection.lock.d'
    lock_dir.mkdir()
    (lock_dir / 'owner.json').write_text(
        json.dumps({'pid': 2**30, 'hostname': socket.gethostname(), 'token': 'stale'}),
        encoding='utf-8',
    )

    assert process_lock_is_held(lock_dir) is False
    with exclusive_process_lock(lock_dir, description='test collector'):
        assert process_lock_is_held(lock_dir) is True
    assert not lock_dir.exists()


def test_qualification_selects_required_seed_and_simple_near_center_layouts():
    recipe = collector.load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    selected = collector._select_qualification_seeds(
        recipe,
        count=4,
        candidate_pool=128,
        start_seed=0,
        required_seeds=[17],
    )

    assert selected[0] == 17
    assert len(selected) == 4
    assert len(set(selected)) == 4

    candidates = list(range(128))
    features = np.stack([collector._randomization_feature(recipe, seed) for seed in candidates])
    span = np.ptp(features, axis=0)
    span[span < 1e-12] = 1.0
    normalized = (features - np.min(features, axis=0)) / span
    center = np.full(features.shape[1], 0.5, dtype=float)
    expected_simple = sorted(
        (seed for seed in candidates if seed != 17),
        key=lambda seed: (float(np.linalg.norm(normalized[seed] - center)), seed),
    )[:3]
    assert selected[1:] == expected_simple


def test_recipe_pins_four_prevalidated_qualification_layouts():
    recipe = collector.load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )

    assert recipe['qualification'] == {
        'seed_count': 4,
        'required_seeds': [4906, 485, 34, 12],
    }
    assert recipe['collection'] == {'layout_seeds': [4906, 485, 34, 12]}
    assert collector._select_qualification_seeds(
        recipe,
        count=recipe['qualification']['seed_count'],
        candidate_pool=128,
        start_seed=0,
        required_seeds=recipe['qualification']['required_seeds'],
    ) == [4906, 485, 34, 12]


def test_qualification_metadata_does_not_change_physical_recipe_fingerprint():
    recipe = {'task_name': 'task', 'phases': [{'name': 'move'}]}
    with_qualification = {
        **recipe,
        'qualification': {'seed_count': 4, 'required_seeds': [17, 34, 4906, 485]},
    }

    assert task_recipe_fingerprint(recipe) == task_recipe_fingerprint(with_qualification)


def test_collection_seed_contract_does_not_change_physical_recipe_fingerprint():
    recipe = {'task_name': 'task', 'phases': [{'name': 'move'}]}
    with_collection_contract = {
        **recipe,
        'collection': {'layout_seeds': [4906, 485, 34, 12]},
    }

    assert task_recipe_fingerprint(recipe) == task_recipe_fingerprint(with_collection_contract)


def test_recipe_fingerprint_can_use_a_canonical_cross_machine_root(monkeypatch):
    local_payload = {
        'spec_path': str(BENCHMARK_ROOT / 'tasks' / 'example' / 'recipe.yaml'),
        'objects': [{'usd_path': str(BENCHMARK_ROOT / 'assets' / 'part.usd')}],
    }
    local_repo_root = BENCHMARK_ROOT.parent
    remote_repo_root = Path('/data/remote/InternUtopia')
    remote_root = remote_repo_root / 'roboassemblybench'
    remote_payload = {
        'spec_path': str(remote_root / 'tasks' / 'example' / 'recipe.yaml'),
        'objects': [{'usd_path': str(remote_root / 'assets' / 'part.usd')}],
        'empty_scene': str(remote_repo_root / 'internutopia' / 'assets' / 'scenes' / 'empty.usd'),
    }
    local_payload['empty_scene'] = str(local_repo_root / 'internutopia' / 'assets' / 'scenes' / 'empty.usd')
    local_fingerprint = task_recipe_fingerprint(local_payload)

    monkeypatch.setattr(task_registry, 'BENCHMARK_ROOT', remote_root)
    monkeypatch.setenv('ROBOASSEMBLYBENCH_FINGERPRINT_REPO_ROOT', str(local_repo_root))

    assert task_recipe_fingerprint(remote_payload) == local_fingerprint


def test_qualification_contract_change_retains_only_selected_passes():
    manifest = {
        'selected_seeds': [17, 34, 103, 67],
        'required_passes': 4,
        'results': [
            {'seed': 17, 'passed': True},
            {'seed': 34, 'passed': True},
            {'seed': 103, 'passed': False},
        ],
        'passed': False,
        'failed': True,
        'failed_at_unix': 1.0,
    }

    changed = collector._reconcile_qualification_seed_contract(manifest, [17, 34, 3, 5])

    assert changed is True
    assert manifest['selected_seeds'] == [17, 34, 3, 5]
    assert [item['seed'] for item in manifest['results']] == [17, 34]
    assert manifest['failed'] is False
    assert manifest['passed'] is False
    assert 'failed_at_unix' not in manifest
    assert manifest['seed_contract_history'][-1]['retained_passed_seeds'] == [17, 34]


def test_resume_reconciles_completed_running_batch_and_next_seed(tmp_path: Path):
    batch_dir = tmp_path / 'batch_000003_000003'
    metadata_path = batch_dir / 'episode_000000_cartesian_raw' / 'metadata.json'
    quality = {'valid': True, 'seed': 3, 'metadata_path': str(metadata_path)}
    manifest = {
        'next_seed': 3,
        'successful_episodes': {},
        'failed_attempts': [{'seed': 2}],
        'batches': [{'seeds': [3], 'batch_dir': str(batch_dir), 'status': 'running'}],
    }

    collector._reconcile_manifest_on_resume(manifest, {3: quality})

    assert manifest['next_seed'] == 4
    assert manifest['num_successful'] == 1
    assert manifest['num_failed_attempts'] == 1
    assert manifest['batches'][0]['status'] == 'recovered_completed'
    assert manifest['batches'][0]['quality'] == [quality]


def test_qualification_retries_resource_abort_without_freezing_recipe(tmp_path: Path, monkeypatch):
    attempts = []
    sleeps = []
    monkeypatch.setattr(collector, '_select_qualification_seeds', lambda *_args, **_kwargs: [26])
    monkeypatch.setattr(collector, '_wait_for_resources', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collector, '_batch_command', lambda *_args, **_kwargs: ['worker'])
    monkeypatch.setattr(collector.time, 'sleep', sleeps.append)
    monkeypatch.setattr(
        collector,
        '_qualification_result',
        lambda results_path, *, seed: {
            'seed': seed,
            'passed': True,
            'phase_status': 'success',
            'steps': 100,
            'terminal_reason': 'success-criteria-met',
            'terminal_phase': 'complete',
            'success_diagnostics': [],
            'results_path': str(results_path),
        },
    )

    def run_worker(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            return -15, {
                'reason': 'low-available-memory',
                'available_memory_gib': 0.4,
                'threshold_gib': 0.5,
            }
        return 0, None

    monkeypatch.setattr(collector, '_run_worker_with_resource_monitor', run_worker)
    args = SimpleNamespace(
        skip_qualification=False,
        recipe_fingerprint='recipe-fingerprint',
        recipe='recipe',
        scene_profile='scene',
        recipe_spec={},
        qualification_seed_count=1,
        qualification_candidate_pool=1,
        qualification_start_seed=0,
        qualification_required_seeds=[26],
        retry_failed_qualification=False,
        num_episodes=2000,
        resource_wait_seconds=7.0,
    )

    manifest = collector._ensure_recipe_qualified(args, tmp_path)

    assert manifest['passed'] is True
    assert manifest['failed'] is False
    assert len(manifest['resource_aborts']) == 1
    assert manifest['resource_aborts'][0]['resource_abort']['reason'] == 'low-available-memory'
    assert manifest['results'][0]['passed'] is True
    assert len(attempts) == 2
    assert sleeps == [7.0]
