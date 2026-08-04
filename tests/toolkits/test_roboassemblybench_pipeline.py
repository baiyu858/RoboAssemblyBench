import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboassemblybench.core.process_lock import exclusive_process_lock
from roboassemblybench.scripts import run_fabrica_plumbers_block_act_pipeline as pipeline_module
from roboassemblybench.scripts.run_fabrica_plumbers_block_act_pipeline import (
    Pipeline,
    _lock_is_held,
    _resolve_checkpoint,
    _validate_collection_manifest,
)
from roboassemblybench.scripts.export_fabrica_plumbers_block_lerobot_v3 import (
    _reconcile_conversion_manifest,
)


def _manifest(count: int = 2):
    return {
        'complete': True,
        'recipe_fingerprint': 'test-recipe-fingerprint',
        'target_successful_episodes': count,
        'num_successful': count,
        'successful_episodes': {
            str(seed): {
                'metadata_path': f'/tmp/{seed}.json',
                'recipe_fingerprint': 'test-recipe-fingerprint',
            }
            for seed in range(count)
        },
        'timing_contract': {
            'physics_fps': 240,
            'control_fps': 240,
            'dataset_fps': 30,
            'dataset_frame_stride': 8,
            'rendering_interval': 7,
            'camera_render_period_steps': 8,
            'camera_state_action_aligned': True,
        },
    }


def test_pipeline_requires_exact_complete_collection():
    _validate_collection_manifest(_manifest(), expected_episodes=2)
    incomplete = _manifest()
    incomplete['complete'] = False
    with pytest.raises(ValueError, match='not complete'):
        _validate_collection_manifest(incomplete, expected_episodes=2)


def test_pipeline_lock_detection_uses_nfs_safe_directory(tmp_path: Path):
    legacy_lock = tmp_path / '.collection.lock'
    legacy_lock.touch()
    assert _lock_is_held(legacy_lock) is False

    with exclusive_process_lock(
        tmp_path / '.collection.lock.d',
        description='test collector',
    ):
        assert _lock_is_held(legacy_lock) is True

    assert _lock_is_held(legacy_lock) is False


def test_pipeline_rejects_stale_camera_contract():
    manifest = _manifest()
    manifest['timing_contract']['rendering_interval'] = 5
    with pytest.raises(ValueError, match='240/30/8'):
        _validate_collection_manifest(manifest, expected_episodes=2)


def test_pipeline_rejects_mixed_recipe_fingerprints():
    manifest = _manifest()
    manifest['successful_episodes']['1']['recipe_fingerprint'] = 'stale-recipe'
    with pytest.raises(ValueError, match='different recipe fingerprint'):
        _validate_collection_manifest(manifest, expected_episodes=2)


def test_pipeline_resolves_latest_numeric_checkpoint(tmp_path: Path):
    for step in (1, 100):
        checkpoint = tmp_path / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text(json.dumps({'type': 'act'}), encoding='utf-8')
    assert _resolve_checkpoint(tmp_path).parent.name == '000100'


def test_pipeline_resumes_training_when_checkpoint_exists(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / 'train' / 'checkpoints' / '010000' / 'pretrained_model'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
    args = SimpleNamespace(
        raw_dir=tmp_path / 'raw',
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        act_env='act',
        dataset_repo_id='test/repo',
        train_steps=100000,
        batch_size=4,
        num_workers=2,
    )
    pipeline = Pipeline(args)
    captured = {}

    def fake_run(name, command, *, env=None):
        captured.update(env or {})

    monkeypatch.setattr(pipeline, '_run', fake_run)
    assert pipeline.train() == checkpoint.resolve()
    assert captured['RESUME'] == 'true'


def test_pipeline_restarts_missing_collector_and_resumes_manifest(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    manifest_path = raw_dir / 'collection_manifest.json'
    incomplete = _manifest(count=1)
    incomplete.update({'complete': False, 'num_successful': 0, 'successful_episodes': {}})
    manifest_path.write_text(json.dumps(incomplete), encoding='utf-8')
    args = SimpleNamespace(
        raw_dir=raw_dir,
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        expected_episodes=1,
        poll_seconds=0.01,
        supervise_collection=True,
        recipe_fingerprint='test-recipe-fingerprint',
    )
    pipeline = Pipeline(args)
    starts = []

    def fake_start_collector():
        starts.append(True)
        manifest_path.write_text(json.dumps(_manifest(count=1)), encoding='utf-8')

    monkeypatch.setattr(pipeline_module, '_lock_is_held', lambda _path: False)
    monkeypatch.setattr(pipeline, '_start_collector', fake_start_collector)
    monkeypatch.setattr(pipeline_module.time, 'sleep', lambda _seconds: None)

    manifest = pipeline.wait_for_collection()

    assert starts == [True]
    assert manifest['complete'] is True


def test_pipeline_waits_for_external_parallel_collection_without_starting_collector(
    tmp_path: Path,
    monkeypatch,
):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    manifest_path = raw_dir / 'collection_manifest.json'
    incomplete = _manifest(count=1)
    incomplete.update({'complete': False, 'num_successful': 0, 'successful_episodes': {}})
    manifest_path.write_text(json.dumps(incomplete), encoding='utf-8')
    args = SimpleNamespace(
        raw_dir=raw_dir,
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        expected_episodes=1,
        poll_seconds=0.01,
        supervise_collection=True,
        external_collection=True,
        recipe_fingerprint='test-recipe-fingerprint',
    )
    pipeline = Pipeline(args)
    starts = []

    def finish_external_collection(_seconds):
        manifest_path.write_text(json.dumps(_manifest(count=1)), encoding='utf-8')

    monkeypatch.setattr(pipeline_module, '_lock_is_held', lambda _path: False)
    monkeypatch.setattr(pipeline, '_start_collector', lambda: starts.append(True))
    monkeypatch.setattr(pipeline_module.time, 'sleep', finish_external_collection)

    manifest = pipeline.wait_for_collection()

    assert starts == []
    assert manifest['complete'] is True


def test_pipeline_passes_worker_timeout_to_collector(tmp_path: Path, monkeypatch):
    args = SimpleNamespace(
        raw_dir=tmp_path / 'raw',
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        recipe_fingerprint='test-recipe-fingerprint',
        qualification_seeds=[17],
        isaac_env='isaac',
        expected_episodes=2000,
        collection_max_attempts=10000,
        recipe='recipe',
        scene_profile='scene',
        collection_min_available_memory_gib=5.5,
        collection_abort_available_memory_gib=0.5,
        collection_worker_timeout_seconds=1800.0,
    )
    monkeypatch.setattr(pipeline_module.shutil, 'which', lambda _executable: '/usr/bin/conda')

    command = Pipeline(args)._collector_command()

    assert command[command.index('--worker-timeout-seconds') + 1] == '1800.0'


def test_pipeline_passes_fixed_layout_seed_contract_to_collector(tmp_path: Path, monkeypatch):
    args = SimpleNamespace(
        raw_dir=tmp_path / 'raw',
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        recipe_fingerprint='test-recipe-fingerprint',
        qualification_seeds=[4906, 485, 34, 12],
        collection_layout_seeds=[4906, 485, 34, 12],
        isaac_env='isaac',
        expected_episodes=2000,
        collection_max_attempts=10000,
        recipe='recipe',
        scene_profile='scene',
        collection_min_available_memory_gib=5.5,
        collection_abort_available_memory_gib=0.5,
        collection_worker_timeout_seconds=1800.0,
    )
    monkeypatch.setattr(pipeline_module.shutil, 'which', lambda _executable: '/usr/bin/conda')

    command = Pipeline(args)._collector_command()

    seed_start = command.index('--layout-seeds') + 1
    assert command[seed_start : seed_start + 4] == ['4906', '485', '34', '12']


def test_pipeline_rejects_episode_outside_fixed_layout_contract():
    manifest = _manifest()
    manifest['collection_layout_seeds'] = [4906, 485, 34, 12]
    manifest['successful_episodes']['0']['layout_seed'] = 4906
    manifest['successful_episodes']['1']['layout_seed'] = 17

    with pytest.raises(ValueError, match='outside the allowed layout seeds'):
        _validate_collection_manifest(
            manifest,
            expected_episodes=2,
            expected_layout_seeds=[4906, 485, 34, 12],
        )


def test_pipeline_does_not_restart_a_failed_recipe_qualification(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    (raw_dir / 'qualification_status.json').write_text(
        json.dumps(
            {
                'recipe_fingerprint': 'failed-recipe',
                'passed': False,
                'failed': True,
                'failed_result': {'seed': 17, 'terminal_reason': 'physical-contact-failed'},
            }
        ),
        encoding='utf-8',
    )
    args = SimpleNamespace(
        raw_dir=raw_dir,
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        expected_episodes=1,
        poll_seconds=0.01,
        supervise_collection=True,
        recipe_fingerprint='failed-recipe',
    )
    pipeline = Pipeline(args)
    starts = []
    monkeypatch.setattr(pipeline_module, '_lock_is_held', lambda _path: False)
    monkeypatch.setattr(pipeline, '_start_collector', lambda: starts.append(True))

    with pytest.raises(RuntimeError, match='automatic collection restart is disabled'):
        pipeline.wait_for_collection()

    assert starts == []


def test_pipeline_recovers_stale_resource_aborted_qualification(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    status_path = raw_dir / 'qualification_status.json'
    status_path.write_text(
        json.dumps(
            {
                'recipe_fingerprint': 'test-recipe-fingerprint',
                'passed': False,
                'failed': True,
                'failed_result': {
                    'seed': 26,
                    'terminal_reason': 'missing-qualification-result',
                    'resource_abort': {'reason': 'low-available-memory'},
                },
            }
        ),
        encoding='utf-8',
    )
    args = SimpleNamespace(
        raw_dir=raw_dir,
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        expected_episodes=1,
        poll_seconds=0.01,
        supervise_collection=True,
        recipe_fingerprint='test-recipe-fingerprint',
    )
    pipeline = Pipeline(args)
    starts = []

    def fake_start_collector():
        starts.append(True)
        status_path.write_text(
            json.dumps(
                {
                    'recipe_fingerprint': 'test-recipe-fingerprint',
                    'passed': True,
                    'failed': False,
                }
            ),
            encoding='utf-8',
        )
        (raw_dir / 'collection_manifest.json').write_text(
            json.dumps(_manifest(count=1)),
            encoding='utf-8',
        )

    monkeypatch.setattr(pipeline_module, '_lock_is_held', lambda _path: False)
    monkeypatch.setattr(pipeline, '_start_collector', fake_start_collector)
    monkeypatch.setattr(pipeline_module.time, 'sleep', lambda _seconds: None)

    manifest = pipeline.wait_for_collection()

    assert starts == [True]
    assert manifest['complete'] is True


def test_pipeline_restarts_when_failed_qualification_belongs_to_old_recipe(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    status_path = raw_dir / 'qualification_status.json'
    status_path.write_text(
        json.dumps(
            {
                'recipe_fingerprint': 'old-recipe',
                'passed': False,
                'failed': True,
                'failed_result': {
                    'seed': 26,
                    'terminal_reason': 'physical-contact-failed',
                },
            }
        ),
        encoding='utf-8',
    )
    args = SimpleNamespace(
        raw_dir=raw_dir,
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        expected_episodes=1,
        poll_seconds=0.01,
        supervise_collection=True,
        recipe_fingerprint='test-recipe-fingerprint',
    )
    pipeline = Pipeline(args)
    starts = []

    def fake_start_collector():
        starts.append(True)
        status_path.write_text(
            json.dumps(
                {
                    'recipe_fingerprint': 'test-recipe-fingerprint',
                    'passed': True,
                    'failed': False,
                }
            ),
            encoding='utf-8',
        )
        (raw_dir / 'collection_manifest.json').write_text(
            json.dumps(_manifest(count=1)),
            encoding='utf-8',
        )

    monkeypatch.setattr(pipeline_module, '_lock_is_held', lambda _path: False)
    monkeypatch.setattr(pipeline, '_start_collector', fake_start_collector)
    monkeypatch.setattr(pipeline_module.time, 'sleep', lambda _seconds: None)

    manifest = pipeline.wait_for_collection()

    assert starts == [True]
    assert manifest['complete'] is True


def test_pipeline_restarts_when_qualification_seed_contract_is_superseded(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    status_path = raw_dir / 'qualification_status.json'
    status_path.write_text(
        json.dumps(
            {
                'recipe_fingerprint': 'test-recipe-fingerprint',
                'selected_seeds': [17, 34, 103, 67],
                'passed': False,
                'failed': True,
                'failed_result': {'seed': 103, 'terminal_reason': 'ik_failed'},
            }
        ),
        encoding='utf-8',
    )
    args = SimpleNamespace(
        raw_dir=raw_dir,
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
        expected_episodes=1,
        poll_seconds=0.01,
        supervise_collection=True,
        recipe_fingerprint='test-recipe-fingerprint',
        qualification_seeds=[17, 34, 3, 5],
    )
    pipeline = Pipeline(args)
    starts = []

    def fake_start_collector():
        starts.append(True)
        status_path.write_text(
            json.dumps(
                {
                    'recipe_fingerprint': 'test-recipe-fingerprint',
                    'selected_seeds': [17, 34, 3, 5],
                    'passed': True,
                    'failed': False,
                }
            ),
            encoding='utf-8',
        )
        (raw_dir / 'collection_manifest.json').write_text(
            json.dumps(_manifest(count=1)),
            encoding='utf-8',
        )

    monkeypatch.setattr(pipeline_module, '_lock_is_held', lambda _path: False)
    monkeypatch.setattr(pipeline, '_start_collector', fake_start_collector)
    monkeypatch.setattr(pipeline_module.time, 'sleep', lambda _seconds: None)

    manifest = pipeline.wait_for_collection()

    assert starts == [True]
    assert manifest['complete'] is True


def test_pipeline_replaces_stale_stage_details_and_clears_old_error(tmp_path: Path, monkeypatch):
    args = SimpleNamespace(
        raw_dir=tmp_path / 'raw_v3',
        dataset_dir=tmp_path / 'dataset',
        train_dir=tmp_path / 'train',
        eval_dir=tmp_path / 'eval',
        pipeline_output_dir=tmp_path / 'pipeline',
    )
    pipeline = Pipeline(args)
    pipeline.state.update(
        {
            'complete': False,
            'error': 'old failure',
            'failed_at_unix': 1.0,
            'stages': {'collection': {'manifest_path': '/tmp/raw_v2/collection_manifest.json'}},
        }
    )
    pipeline._set_stage('collection', 'waiting', replace=True, successful_episodes=0)
    assert 'manifest_path' not in pipeline.state['stages']['collection']

    def fake_wait_for_collection():
        assert 'error' not in pipeline.state
        assert 'failed_at_unix' not in pipeline.state
        return {}

    monkeypatch.setattr(pipeline, 'wait_for_collection', fake_wait_for_collection)
    monkeypatch.setattr(pipeline, 'wait_for_resources', lambda _stage: None)
    monkeypatch.setattr(pipeline, 'export', lambda: None)
    monkeypatch.setattr(pipeline, 'train', lambda: tmp_path / 'checkpoint')
    monkeypatch.setattr(pipeline, 'evaluate', lambda _checkpoint: None)

    pipeline.run()

    assert pipeline.state['complete'] is True


def test_conversion_manifest_recovers_dataset_episode_committed_before_manifest(tmp_path: Path):
    episodes = [
        {
            'seed': seed,
            'metadata_path': str(tmp_path / f'episode_{seed}' / 'metadata.json'),
            'frame_count': 100 + seed,
            'domain_randomization': {'enabled': True, 'seed': seed},
        }
        for seed in range(3)
    ]
    manifest = {'episodes': []}

    changed = _reconcile_conversion_manifest(manifest, episodes, dataset_episode_count=2)

    assert changed is True
    assert [item['seed'] for item in manifest['episodes']] == [0, 1]
    assert manifest['total_episodes'] == 2
    assert manifest['total_frames'] == 201


def test_conversion_manifest_rejects_nonprefix_source_order(tmp_path: Path):
    episodes = [
        {
            'seed': seed,
            'metadata_path': str(tmp_path / f'episode_{seed}' / 'metadata.json'),
            'frame_count': 100,
        }
        for seed in range(2)
    ]
    manifest = {
        'episodes': [
            {
                'source_metadata': episodes[1]['metadata_path'],
                'frame_count': 100,
            }
        ]
    }
    with pytest.raises(RuntimeError, match='not a prefix'):
        _reconcile_conversion_manifest(manifest, episodes, dataset_episode_count=1)
