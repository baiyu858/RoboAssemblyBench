import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from roboassemblybench.core import task_registry
from roboassemblybench.core.paths import BENCHMARK_ROOT
from roboassemblybench.core.process_lock import exclusive_process_lock, process_lock_is_held
from roboassemblybench.core.task_registry import task_recipe_fingerprint
from roboassemblybench.datasets.cartesian_episode import (
    ACTION_NAMES,
    ACTION_SEMANTICS,
    CAMERA_KEYS,
    STATE_NAMES,
)
from roboassemblybench.scripts import collect_fabrica_plumbers_block_2k as collector


def test_batch_command_keeps_simulation_cadence_independent_from_dataset_fps(monkeypatch, tmp_path: Path):
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
        seeds=[34],
        batch_dir=tmp_path,
        results_path=tmp_path / 'results.json',
    )

    assert command[command.index('--dataset-fps') + 1] == '30'
    assert command[command.index('--dataset-frame-stride') + 1] == '8'
    assert command[command.index('--rendering-fps') + 1] == '240'


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


def test_quality_check_rejects_stale_camera_timing(tmp_path: Path):
    assert collector._quality_check_episode(
        _write_quality_fixture(tmp_path / 'aligned', rendering_interval=7)
    )['valid']
    quality = collector._quality_check_episode(
        _write_quality_fixture(tmp_path / 'stale', rendering_interval=5)
    )
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
    local_payload['empty_scene'] = str(
        local_repo_root / 'internutopia' / 'assets' / 'scenes' / 'empty.usd'
    )
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
