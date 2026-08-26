from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from roboassemblybench.core.domain_randomization import normalize_randomization_profile
try:
    from roboassemblybench.datasets.cartesian_episode import expected_replay_joint_widths
except ImportError:
    # Older, stable collection branches predate this optional replay signature check.
    expected_replay_joint_widths = None
from roboassemblybench.scripts.collect_fabrica_plumbers_block_2k import (
    _exclusive_collection_lock,
    _quality_check_episode,
    _run_worker_with_resource_monitor,
    _wait_for_resources,
    _write_json_atomic,
)
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe


REPO_ROOT = Path(os.environ.get('RAB_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
MANIFEST_NAME = 'replay_manifest.json'
VISUAL_PROFILES = ('object_distractors', 'texture', 'lighting', 'table_color', 'scene')


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _source_episodes(
    source_dir: Path,
    *,
    expected_recipe_fingerprint: str,
    expected_joint_widths: list[int] | None,
) -> list[dict[str, Any]]:
    manifest_path = source_dir / 'collection_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Stage-1 collection manifest is missing: {manifest_path}.')
    manifest = _load_json(manifest_path)
    successful = list((manifest.get('successful_episodes') or {}).values())
    episodes = []
    for item in successful:
        metadata_path = Path(item.get('metadata_path') or '').resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Stage-1 metadata is missing: {metadata_path}.')
        metadata = _load_json(metadata_path)
        if metadata.get('recording_mode') not in {'trajectory_only', 'rendered'}:
            raise ValueError(f'Stage-1 source is not an authoritative rollout: {metadata_path}.')
        if not bool((metadata.get('metrics') or {}).get('success', False)):
            raise ValueError(f'Stage-1 manifest references an unsuccessful trajectory: {metadata_path}.')
        source_fingerprint = str(metadata.get('recipe_fingerprint', '') or '')
        if source_fingerprint != expected_recipe_fingerprint:
            raise ValueError(
                f'Stage-1 recipe fingerprint {source_fingerprint!r} does not match '
                f'{expected_recipe_fingerprint!r}: {metadata_path}.'
            )
        trajectory_path = Path(metadata.get('trajectory_path') or '')
        if not trajectory_path.is_file():
            raise FileNotFoundError(f'Stage-1 trajectory is missing: {trajectory_path}.')
        if expected_joint_widths is not None:
            with np.load(trajectory_path) as trajectory:
                widths = [int(value) for value in np.asarray(trajectory['replay_joint_widths']).tolist()]
            if widths != expected_joint_widths:
                raise ValueError(
                    f'Stage-1 joint widths {widths} do not match the Franka articulation '
                    f'signature {expected_joint_widths}: {trajectory_path}.'
                )
        episodes.append(
            {
                'seed': int(metadata['seed']),
                'layout_seed': int(metadata['layout_seed']),
                'metadata_path': str(metadata_path),
                'frame_count': int(metadata['frame_count']),
            }
        )
    episodes.sort(key=lambda item: item['seed'])
    if len({item['seed'] for item in episodes}) != len(episodes):
        raise ValueError('Stage-1 source manifest contains duplicate seeds.')
    return episodes


def _replay_command(args, *, sources: list[dict[str, Any]], batch_dir: Path, results_path: Path) -> list[str]:
    thread_count = os.environ.get('ISAACSIM_OMP_NUM_THREADS', '1')
    environment = [
        'PYTHONNOUSERSITE=1',
        'PYTHONUNBUFFERED=1',
        f'OMP_NUM_THREADS={thread_count}',
        f'MKL_NUM_THREADS={thread_count}',
        f'OPENBLAS_NUM_THREADS={thread_count}',
        f'NUMEXPR_NUM_THREADS={thread_count}',
    ]
    if args.isaac_python:
        executable = Path(args.isaac_python).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        command = ['env', *environment, str(executable)]
    else:
        conda = shutil.which('conda')
        if conda is None:
            raise RuntimeError('conda executable was not found; pass --isaac-python.')
        command = [conda, 'run', '--no-capture-output', '-n', args.conda_env, 'env', *environment, 'python']
    command.extend(
        [
            str(REPO_ROOT / 'roboassemblybench' / 'scripts' / 'generate_demos.py'),
            '--worker-mode',
            'replay',
            '--worker-recipe',
            args.recipe,
            '--worker-scene-profile',
            args.scene_profile,
            '--worker-results-path',
            str(results_path),
            '--worker-replay-sources',
            *[item['metadata_path'] for item in sources],
            '--output-dir',
            str(batch_dir),
            '--rendering-fps',
            str(args.rendering_fps),
            '--domain-randomization',
            '--randomization-profile',
            args.randomization_profile,
            '--video-codec',
            args.video_codec,
            '--video-crf',
            str(args.video_crf),
            '--video-preset',
            args.video_preset,
            '--depth-compression-level',
            str(args.depth_compression_level),
            '--headless',
        ]
    )
    return command


def _initial_manifest(args, source_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'schema_version': 'roboassemblybench_visual_replay_collection_v1',
        'recipe': args.recipe,
        'scene_profile': args.scene_profile,
        'randomization_profile': args.randomization_profile,
        'source_dir': str(Path(args.source_dir).resolve()),
        'source_count': len(source_episodes),
        'target_episodes': min(int(args.num_episodes), len(source_episodes)),
        'rendering_fps': int(args.rendering_fps),
        'video': {
            'codec': args.video_codec,
            'crf': int(args.video_crf),
            'preset': args.video_preset,
        },
        'depth': {
            'dtype': 'uint16',
            'depth_scale': 0.001,
            'compression': 'zstd',
            'filter': 'bitshuffle',
            'compression_level': int(args.depth_compression_level),
        },
        'successful_episodes': {},
        'failed_attempts': [],
        'batches': [],
        'complete': False,
    }


def collect(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = load_task_recipe(args.recipe, scene_profile=args.scene_profile)
    recipe_fingerprint = str(recipe['recipe_fingerprint'])
    expected_joint_widths = (
        expected_replay_joint_widths(recipe.get('robots', [])) if expected_replay_joint_widths is not None else None
    )
    source_episodes = _source_episodes(
        Path(args.source_dir).resolve(),
        expected_recipe_fingerprint=recipe_fingerprint,
        expected_joint_widths=expected_joint_widths,
    )
    target = min(int(args.num_episodes), len(source_episodes))
    if target <= 0:
        raise RuntimeError('No Stage-1 trajectories are available for replay.')
    source_episodes = source_episodes[:target]
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        contract = (
            manifest.get('randomization_profile'),
            Path(manifest.get('source_dir') or '').resolve(),
        )
        expected = (args.randomization_profile, Path(args.source_dir).resolve())
        if contract != expected:
            raise RuntimeError(f'Replay manifest contract mismatch: {contract} != {expected}.')
        previous_target = int(manifest.get('target_episodes', 0))
        if target < previous_target:
            raise RuntimeError(
                f'Replay source set shrank from {previous_target} to {target}; '
                'Stage-2 incremental replay requires an append-only Stage-1 manifest.'
            )
        # A new Stage-1 success extends the source set. Existing successful
        # replays are retained; only source seeds absent from ``completed`` run.
        manifest['source_count'] = len(source_episodes)
        manifest['target_episodes'] = target
        if target > previous_target:
            manifest['complete'] = False
            manifest['finished_at_unix'] = None
    else:
        manifest = _initial_manifest(args, source_episodes)

    completed = manifest.setdefault('successful_episodes', {})
    for metadata_path in output_dir.rglob('episode_*_cartesian_raw/metadata.json'):
        quality = _quality_check_episode(
            metadata_path,
            expected_recipe_fingerprint=recipe_fingerprint,
            allowed_layout_seeds=None,
            require_extended_observations=True,
            require_visual_quality=bool(args.require_visual_quality),
            expected_randomization_profile=args.randomization_profile,
        )
        if quality['valid']:
            completed[str(quality['seed'])] = quality
    _write_json_atomic(manifest_path, manifest)

    pending = [item for item in source_episodes if str(item['seed']) not in completed]
    while pending:
        _wait_for_resources(args, output_dir, len(pending))
        sources = pending[: int(args.batch_size)]
        batch_name = f'batch_{sources[0]["seed"]:06d}_{sources[-1]["seed"]:06d}'
        batch_dir = output_dir / 'batches' / batch_name
        if batch_dir.exists():
            batch_dir = output_dir / 'batches' / f'{batch_name}_retry_{int(time.time())}'
        batch_dir.mkdir(parents=True, exist_ok=False)
        results_path = batch_dir / 'replay_results.json'
        log_path = batch_dir / 'worker.log'
        command = _replay_command(args, sources=sources, batch_dir=batch_dir, results_path=results_path)
        batch_record = {
            'seeds': [item['seed'] for item in sources],
            'sources': [item['metadata_path'] for item in sources],
            'batch_dir': str(batch_dir),
            'log_path': str(log_path),
            'started_at_unix': time.time(),
            'status': 'running',
        }
        manifest['batches'].append(batch_record)
        _write_json_atomic(manifest_path, manifest)
        if args.dry_run:
            print(' '.join(command), flush=True)
            batch_record['status'] = 'dry_run'
            _write_json_atomic(manifest_path, manifest)
            break

        with log_path.open('w', encoding='utf-8') as log_file:
            returncode, resource_abort = _run_worker_with_resource_monitor(command, log_file=log_file, args=args)
        batch_record['returncode'] = returncode
        batch_record['resource_abort'] = resource_abort
        batch_record['finished_at_unix'] = time.time()
        batch_record['status'] = 'resource_aborted' if resource_abort else ('completed' if returncode == 0 else 'failed')
        valid_seeds = set()
        quality_records = []
        for metadata_path in batch_dir.rglob('episode_*_cartesian_raw/metadata.json'):
            quality = _quality_check_episode(
                metadata_path,
                expected_recipe_fingerprint=recipe_fingerprint,
                allowed_layout_seeds=None,
                require_extended_observations=True,
                require_visual_quality=bool(args.require_visual_quality),
                expected_randomization_profile=args.randomization_profile,
            )
            quality_records.append(quality)
            if quality['valid']:
                valid_seeds.add(int(quality['seed']))
                completed[str(quality['seed'])] = quality
        batch_record['quality'] = quality_records
        for source in sources:
            if source['seed'] not in valid_seeds:
                manifest['failed_attempts'].append(
                    {
                        'seed': source['seed'],
                        'source_metadata': source['metadata_path'],
                        'batch_dir': str(batch_dir),
                        'returncode': returncode,
                        'resource_abort': resource_abort,
                    }
                )
        manifest['num_successful'] = len(completed)
        _write_json_atomic(manifest_path, manifest)
        if returncode != 0 and not quality_records and resource_abort is None:
            raise RuntimeError(f'Replay worker failed; inspect {log_path}.')
        pending = [item for item in source_episodes if str(item['seed']) not in completed]
        failed_counts = {
            seed: sum(int(item.get('seed', -1)) == seed for item in manifest['failed_attempts'])
            for seed in {item['seed'] for item in pending}
        }
        exhausted = [seed for seed, count in failed_counts.items() if count >= int(args.max_replay_attempts)]
        if exhausted:
            raise RuntimeError(
                f'Replay quality failed {args.max_replay_attempts} times for seeds {sorted(exhausted)}; '
                f'inspect {manifest_path}.'
            )

    manifest['complete'] = len(completed) >= target
    manifest['num_successful'] = len(completed)
    manifest['finished_at_unix'] = time.time() if manifest['complete'] else None
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Render successful Fabrica trajectories with one visual profile.')
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-episodes', type=int, default=1430)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-replay-attempts', type=int, default=2)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--scene-profile', default='taoyuan_grscenes_tabletop')
    parser.add_argument('--randomization-profile', choices=VISUAL_PROFILES, required=True)
    parser.add_argument('--rendering-fps', type=int, default=80)
    parser.add_argument('--conda-env', default='internutopia311')
    parser.add_argument('--isaac-python', default=None)
    parser.add_argument('--video-codec', choices=['h264', 'h265'], default='h264')
    parser.add_argument('--video-crf', type=int, default=23)
    parser.add_argument('--video-preset', default='veryfast')
    parser.add_argument('--depth-compression-level', type=int, default=5)
    parser.add_argument('--require-visual-quality', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--min-available-memory-gib', type=float, default=32.0)
    parser.add_argument('--abort-available-memory-gib', type=float, default=16.0)
    parser.add_argument('--resource-poll-seconds', type=float, default=10.0)
    parser.add_argument('--resource-wait-seconds', type=float, default=60.0)
    parser.add_argument('--low-memory-grace-polls', type=int, default=3)
    parser.add_argument('--worker-timeout-seconds', type=float, default=1800.0)
    parser.add_argument('--worker-stall-timeout-seconds', type=float, default=600.0)
    parser.add_argument('--estimated-episode-mib', type=float, default=96.0)
    parser.add_argument('--disk-reserve-gib', type=float, default=256.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.randomization_profile = normalize_randomization_profile(args.randomization_profile)
    if args.num_episodes <= 0 or args.batch_size <= 0 or args.max_replay_attempts <= 0 or args.rendering_fps != 80:
        parser.error('num-episodes and batch-size must be positive; the two-stage pipeline requires 80 Hz.')
    if not 0 <= args.video_crf <= 51 or not 1 <= args.depth_compression_level <= 22:
        parser.error('video-crf must be 0..51 and depth-compression-level must be 1..22.')
    with _exclusive_collection_lock(Path(args.output_dir).resolve()):
        manifest = collect(args)
    print(json.dumps({'complete': manifest['complete'], 'num_successful': manifest['num_successful']}, indent=2))


if __name__ == '__main__':
    main()
