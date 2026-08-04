from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboassemblybench.scripts.collect_fabrica_plumbers_block_parallel import (
    _build_shards,
    _collector_command,
    _episode_counts,
    _merge_manifests,
    _worker_limit,
)


def _args(tmp_path: Path, **overrides):
    values = {
        'output_dir': str(tmp_path),
        'num_episodes': 10,
        'num_workers': 4,
        'gpu_ids': [0, 1],
        'start_seed': 100000,
        'seed_stride': 10000,
        'max_attempts_per_shard': 1000,
        'conda_env': 'internutopia311',
        'recipe': 'fabrica_plumbers_block_ur5e_right_base_prepare',
        'scene_profile': 'taoyuan_grscenes_tabletop',
        'dataset_fps': 30,
        'dataset_frame_stride': 8,
        'rendering_fps': 240,
        'min_available_memory_gib': 12.0,
        'abort_available_memory_gib': 2.0,
        'worker_timeout_seconds': 1800.0,
        'disk_reserve_gib': 100.0,
        'layout_seeds': [4906, 485, 34, 12],
        'skip_qualification': True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_shard_manifest(shard: dict, *, fingerprint: str, duplicate_seed: int | None = None):
    start_seed = int(shard['start_seed']) if duplicate_seed is None else int(duplicate_seed)
    successful = {
        str(start_seed + offset): {
            'seed': start_seed + offset,
            'layout_seed': shard['layout_seeds'][(start_seed + offset) % len(shard['layout_seeds'])],
            'recipe_fingerprint': fingerprint,
            'metadata_path': f'/data/episode_{start_seed + offset}/metadata.json',
        }
        for offset in range(int(shard['target_episodes']))
    }
    manifest = {
        'complete': True,
        'num_successful': int(shard['target_episodes']),
        'num_failed_attempts': 0,
        'recipe_fingerprint': fingerprint,
        'collection_layout_seeds': list(shard['layout_seeds']),
        'timing_contract': {
            'physics_fps': 240,
            'control_fps': 240,
            'dataset_fps': 30,
            'dataset_frame_stride': 8,
        },
        'successful_episodes': successful,
        'failed_attempts': [],
        'finished_at_unix': 123.0,
    }
    path = Path(shard['manifest_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding='utf-8')


def test_episode_counts_are_balanced():
    assert _episode_counts(10, 4) == [3, 3, 2, 2]
    assert _episode_counts(8, 4) == [2, 2, 2, 2]


def test_worker_limit_defaults_to_all_shards_and_accepts_a_lower_cap(tmp_path):
    args = _args(tmp_path)
    assert _worker_limit(args) == 4

    args.max_concurrent_workers = 2
    assert _worker_limit(args) == 2


def test_build_shards_assigns_disjoint_seed_ranges_and_gpus(tmp_path):
    args = _args(tmp_path)
    shards = _build_shards(args)

    assert [item['target_episodes'] for item in shards] == [3, 3, 2, 2]
    assert [item['gpu_id'] for item in shards] == [0, 1, 0, 1]
    assert [item['start_seed'] for item in shards] == [100000, 110000, 120000, 130000]
    assert [item['layout_seeds'] for item in shards] == [
        [4906, 485, 34],
        [12, 4906, 485],
        [34, 12],
        [4906, 485],
    ]
    command = _collector_command(args, shards[0])
    assert command[command.index('--num-episodes') + 1] == '3'
    assert command[command.index('--layout-seeds') + 1 : -1] == ['4906', '485', '34']
    assert command[-1] == '--skip-qualification'


def test_build_shards_assigns_one_layout_to_each_single_episode_worker(tmp_path):
    args = _args(
        tmp_path,
        num_episodes=4,
        num_workers=4,
        layout_seeds=[101, 202, 303, 404],
    )

    shards = _build_shards(args)

    assert [item['layout_seeds'] for item in shards] == [[101], [202], [303], [404]]


def test_build_shards_rejects_overlapping_attempt_ranges(tmp_path):
    args = _args(tmp_path, seed_stride=1000, max_attempts_per_shard=1000)
    with pytest.raises(ValueError, match='seed ranges cannot overlap'):
        _build_shards(args)


def test_merge_manifests_builds_standard_collection_contract(tmp_path):
    args = _args(tmp_path)
    shards = _build_shards(args)
    fingerprint = 'recipe-fingerprint'
    for shard in shards:
        _write_shard_manifest(shard, fingerprint=fingerprint)

    manifest = _merge_manifests(args, shards, fingerprint)

    assert manifest['complete'] is True
    assert manifest['num_successful'] == 10
    assert manifest['target_successful_episodes'] == 10
    assert manifest['single_worker'] is False
    assert manifest['parallel_workers'] == 4
    assert len(manifest['successful_episodes']) == 10
    assert (tmp_path / 'collection_manifest.json').is_file()


def test_merge_manifests_rejects_duplicate_episode_seeds(tmp_path):
    args = _args(tmp_path, num_episodes=2, num_workers=2)
    shards = _build_shards(args)
    fingerprint = 'recipe-fingerprint'
    _write_shard_manifest(shards[0], fingerprint=fingerprint, duplicate_seed=42)
    _write_shard_manifest(shards[1], fingerprint=fingerprint, duplicate_seed=42)

    with pytest.raises(ValueError, match='Duplicate episode seeds'):
        _merge_manifests(args, shards, fingerprint)
