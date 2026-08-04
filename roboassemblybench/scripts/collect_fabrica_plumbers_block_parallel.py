from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from roboassemblybench.core.process_lock import exclusive_process_lock
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = REPO_ROOT / 'roboassemblybench' / 'scripts' / 'collect_fabrica_plumbers_block_2k.py'
DEFAULT_RECIPE = 'fabrica_plumbers_block_ur5e_right_base_prepare'
DEFAULT_SCENE_PROFILE = 'taoyuan_grscenes_tabletop'
DEFAULT_OUTPUT = REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e_right_base_prepare_2k_parallel_raw_v3'
FINGERPRINT_ROOT_ENV = 'ROBOASSEMBLYBENCH_FINGERPRINT_REPO_ROOT'


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def _exclusive_lock(path: Path):
    with exclusive_process_lock(path, description='parallel collector'):
        yield


def _episode_counts(total: int, workers: int) -> list[int]:
    quotient, remainder = divmod(int(total), int(workers))
    return [quotient + int(index < remainder) for index in range(int(workers))]


def _worker_limit(args) -> int:
    configured = getattr(args, 'max_concurrent_workers', None)
    return min(int(configured or args.num_workers), int(args.num_workers))


def _build_shards(args) -> list[dict[str, Any]]:
    counts = _episode_counts(args.num_episodes, args.num_workers)
    shards = []
    episode_offset = 0
    for index, count in enumerate(counts):
        start_seed = int(args.start_seed) + index * int(args.seed_stride)
        gpu_id = int(args.gpu_ids[index % len(args.gpu_ids)])
        max_attempts = max(int(args.max_attempts_per_shard), count)
        if max_attempts >= int(args.seed_stride):
            raise ValueError(
                f'--max-attempts-per-shard ({max_attempts}) must be smaller than '
                f'--seed-stride ({args.seed_stride}) so shard seed ranges cannot overlap.'
            )
        shard_name = f'shard_{index:02d}_gpu_{gpu_id}'
        shard_dir = Path(args.output_dir).resolve() / 'shards' / shard_name
        assigned_layout_count = min(int(count), len(args.layout_seeds))
        shard_layout_seeds = [
            int(args.layout_seeds[(episode_offset + offset) % len(args.layout_seeds)])
            for offset in range(assigned_layout_count)
        ]
        shards.append(
            {
                'index': index,
                'name': shard_name,
                'gpu_id': gpu_id,
                'target_episodes': count,
                'start_seed': start_seed,
                'max_attempts': max_attempts,
                'layout_seeds': shard_layout_seeds,
                'output_dir': str(shard_dir),
                'manifest_path': str(shard_dir / 'collection_manifest.json'),
                'log_path': str(Path(args.output_dir).resolve() / 'logs' / f'{shard_name}.log'),
            }
        )
        episode_offset += int(count)
    return shards


def _shard_status(shard: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(shard['manifest_path'])
    status = {
        'complete': False,
        'num_successful': 0,
        'num_failed_attempts': 0,
        'manifest_path': str(manifest_path),
    }
    if not manifest_path.is_file():
        return status
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        status['manifest_error'] = f'{type(exc).__name__}: {exc}'
        return status
    status.update(
        {
            'complete': bool(manifest.get('complete', False))
            and int(manifest.get('num_successful', -1)) == int(shard['target_episodes']),
            'num_successful': int(manifest.get('num_successful', 0)),
            'num_failed_attempts': int(manifest.get('num_failed_attempts', 0)),
            'next_seed': int(manifest.get('next_seed', shard['start_seed'])),
        }
    )
    return status


def _collector_command(args, shard: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(COLLECTOR),
        '--output-dir',
        str(shard['output_dir']),
        '--num-episodes',
        str(shard['target_episodes']),
        '--start-seed',
        str(shard['start_seed']),
        '--max-attempts',
        str(shard['max_attempts']),
        '--batch-size',
        '1',
        '--conda-env',
        str(args.conda_env),
        '--recipe',
        str(args.recipe),
        '--scene-profile',
        str(args.scene_profile),
        '--dataset-fps',
        str(args.dataset_fps),
        '--dataset-frame-stride',
        str(args.dataset_frame_stride),
        '--rendering-fps',
        str(args.rendering_fps),
        '--min-available-memory-gib',
        str(args.min_available_memory_gib),
        '--abort-available-memory-gib',
        str(args.abort_available_memory_gib),
        '--worker-timeout-seconds',
        str(args.worker_timeout_seconds),
        '--disk-reserve-gib',
        str(args.disk_reserve_gib),
        '--layout-seeds',
        *[str(seed) for seed in shard['layout_seeds']],
    ]
    if args.skip_qualification:
        command.append('--skip-qualification')
    return command


def _start_shard(args, shard: dict[str, Any], restart_count: int) -> dict[str, Any]:
    log_path = Path(shard['log_path'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open('a', encoding='utf-8')
    command = _collector_command(args, shard)
    log_file.write(
        f'\nParallel manager launch at {time.time():.3f}; restart={restart_count}; '
        f'gpu={shard["gpu_id"]}; command={command!r}\n'
    )
    log_file.flush()
    env = os.environ.copy()
    env.update(
        {
            'PYTHONNOUSERSITE': '1',
            'PYTHONUNBUFFERED': '1',
            'OMNI_KIT_ACCEPT_EULA': 'YES',
            'ISAACSIM_ACTIVE_GPU': str(shard['gpu_id']),
            'ISAACSIM_PHYSICS_GPU': str(shard['gpu_id']),
            'ROBOASSEMBLYBENCH_SHARD_INDEX': str(shard['index']),
        }
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        'process': process,
        'log_file': log_file,
        'command': command,
        'started_at_unix': time.time(),
    }


def _stop_process(record: dict[str, Any], timeout: float = 30.0) -> None:
    process = record['process']
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    record['log_file'].close()


def _merge_manifests(args, shards: list[dict[str, Any]], recipe_fingerprint: str) -> dict[str, Any]:
    successful: dict[str, Any] = {}
    failed_attempts = []
    timing_contract = None
    shard_summaries = []
    for shard in shards:
        manifest = _load_json(Path(shard['manifest_path']))
        shard_successful = manifest.get('successful_episodes') or {}
        if not bool(manifest.get('complete', False)):
            raise ValueError(f'{shard["name"]} is not complete.')
        if int(manifest.get('num_successful', -1)) != int(shard['target_episodes']):
            raise ValueError(f'{shard["name"]} does not contain its requested episode count.')
        if str(manifest.get('recipe_fingerprint') or '') != recipe_fingerprint:
            raise ValueError(f'{shard["name"]} recipe fingerprint mismatch.')
        if [int(seed) for seed in manifest.get('collection_layout_seeds') or []] != list(
            shard['layout_seeds']
        ):
            raise ValueError(f'{shard["name"]} layout seed contract mismatch.')
        if timing_contract is None:
            timing_contract = manifest.get('timing_contract') or {}
        elif (manifest.get('timing_contract') or {}) != timing_contract:
            raise ValueError(f'{shard["name"]} timing contract mismatch.')
        duplicates = set(successful).intersection(shard_successful)
        if duplicates:
            raise ValueError(f'Duplicate episode seeds across shards: {sorted(duplicates)[:10]}.')
        successful.update(shard_successful)
        failed_attempts.extend(manifest.get('failed_attempts') or [])
        shard_summaries.append(
            {
                **shard,
                'num_successful': len(shard_successful),
                'num_failed_attempts': len(manifest.get('failed_attempts') or []),
                'finished_at_unix': manifest.get('finished_at_unix'),
            }
        )
    if len(successful) != int(args.num_episodes):
        raise ValueError(
            f'Merged shards contain {len(successful)} episodes, expected {args.num_episodes}.'
        )
    manifest = {
        'schema_version': 'roboassemblybench_position_2k_collection_v2',
        'recipe': args.recipe,
        'scene_profile': args.scene_profile,
        'recipe_fingerprint': recipe_fingerprint,
        'target_successful_episodes': int(args.num_episodes),
        'start_seed': int(args.start_seed),
        'next_seed': max(int(seed) for seed in successful) + 1,
        'batch_size': 1,
        'dataset_fps': int(args.dataset_fps),
        'dataset_frame_stride': int(args.dataset_frame_stride),
        'rendering_fps': int(args.rendering_fps),
        'worker_timeout_seconds': float(args.worker_timeout_seconds),
        'timing_contract': timing_contract or {},
        'domain_randomization': True,
        'collection_layout_seeds': list(args.layout_seeds),
        'layout_assignment': 'parallel_shards_stratified_round_robin',
        'single_worker': False,
        'parallel_workers': int(args.num_workers),
        'parallel_shards': shard_summaries,
        'successful_episodes': {
            seed: successful[seed] for seed in sorted(successful, key=lambda value: int(value))
        },
        'failed_attempts': failed_attempts,
        'num_successful': len(successful),
        'num_failed_attempts': len(failed_attempts),
        'complete': True,
        'finished_at_unix': time.time(),
    }
    _write_json_atomic(Path(args.output_dir).resolve() / 'collection_manifest.json', manifest)
    return manifest


def _manager_state(
    args,
    shards: list[dict[str, Any]],
    running: dict[int, dict[str, Any]],
    restart_counts: dict[int, int],
) -> dict[str, Any]:
    statuses = []
    for shard in shards:
        status = _shard_status(shard)
        record = running.get(int(shard['index']))
        statuses.append(
            {
                **shard,
                **status,
                'pid': None if record is None else int(record['process'].pid),
                'restart_count': int(restart_counts.get(int(shard['index']), 0)),
            }
        )
    return {
        'schema_version': 'roboassemblybench_parallel_collection_manager_v1',
        'recipe': args.recipe,
        'target_episodes': int(args.num_episodes),
        'num_workers': int(args.num_workers),
        'gpu_ids': list(args.gpu_ids),
        'num_successful': sum(int(status['num_successful']) for status in statuses),
        'num_failed_attempts': sum(int(status['num_failed_attempts']) for status in statuses),
        'complete_shards': sum(bool(status['complete']) for status in statuses),
        'shards': statuses,
        'updated_at_unix': time.time(),
    }


def run(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fingerprint_repo_root:
        os.environ[FINGERPRINT_ROOT_ENV] = str(Path(args.fingerprint_repo_root).expanduser())
    recipe = load_task_recipe(args.recipe, scene_profile=args.scene_profile)
    recipe_fingerprint = str(recipe['recipe_fingerprint'])
    shards = _build_shards(args)
    worker_limit = _worker_limit(args)
    running: dict[int, dict[str, Any]] = {}
    restart_counts = {int(shard['index']): 0 for shard in shards}
    restart_after = {int(shard['index']): 0.0 for shard in shards}
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested:
            now = time.time()
            for shard in shards:
                index = int(shard['index'])
                record = running.get(index)
                if record is not None and record['process'].poll() is not None:
                    returncode = int(record['process'].wait())
                    record['log_file'].close()
                    del running[index]
                    status = _shard_status(shard)
                    if status['complete']:
                        print(
                            f'{shard["name"]} completed {status["num_successful"]}/'
                            f'{shard["target_episodes"]}.',
                            flush=True,
                        )
                    else:
                        restart_counts[index] += 1
                        if restart_counts[index] > int(args.max_restarts):
                            raise RuntimeError(
                                f'{shard["name"]} exceeded max_restarts={args.max_restarts}; '
                                f'last return code was {returncode}. Inspect {shard["log_path"]}.'
                            )
                        restart_after[index] = now + float(args.restart_delay_seconds)
                        print(
                            f'{shard["name"]} exited with {returncode}; restart '
                            f'{restart_counts[index]}/{args.max_restarts} scheduled.',
                            flush=True,
                        )

                status = _shard_status(shard)
                if (
                    status['complete']
                    or index in running
                    or now < restart_after[index]
                    or len(running) >= worker_limit
                ):
                    continue
                running[index] = _start_shard(args, shard, restart_counts[index])
                print(
                    f'Started {shard["name"]} pid={running[index]["process"].pid} '
                    f'gpu={shard["gpu_id"]} target={shard["target_episodes"]}.',
                    flush=True,
                )

            state = _manager_state(args, shards, running, restart_counts)
            _write_json_atomic(output_dir / 'parallel_collection_state.json', state)
            print(
                f'Parallel collection: {state["num_successful"]}/{args.num_episodes}, '
                f'failed={state["num_failed_attempts"]}, '
                f'complete_shards={state["complete_shards"]}/{args.num_workers}.',
                flush=True,
            )
            if int(state['complete_shards']) == int(args.num_workers):
                return _merge_manifests(args, shards, recipe_fingerprint)
            time.sleep(float(args.poll_seconds))
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        for record in list(running.values()):
            _stop_process(record)
    raise RuntimeError('Parallel collection stopped before completion.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run resumable, GPU-pinned parallel shards for plumbers-block collection.'
    )
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--max-concurrent-workers',
        type=int,
        default=None,
        help='Maximum live Isaac workers; defaults to --num-workers.',
    )
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--start-seed', type=int, default=100000)
    parser.add_argument('--seed-stride', type=int, default=100000)
    parser.add_argument('--max-attempts-per-shard', type=int, default=10000)
    parser.add_argument('--max-restarts', type=int, default=20)
    parser.add_argument('--restart-delay-seconds', type=float, default=60.0)
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    parser.add_argument('--conda-env', default='internutopia311')
    parser.add_argument('--recipe', default=DEFAULT_RECIPE)
    parser.add_argument('--scene-profile', default=DEFAULT_SCENE_PROFILE)
    parser.add_argument(
        '--fingerprint-repo-root',
        default=None,
        help='Canonical repository root used to preserve recipe fingerprints across machines.',
    )
    parser.add_argument('--layout-seeds', type=int, nargs='+', default=None)
    parser.add_argument('--dataset-fps', type=int, default=30)
    parser.add_argument('--dataset-frame-stride', type=int, default=8)
    parser.add_argument('--rendering-fps', type=int, default=240)
    parser.add_argument('--min-available-memory-gib', type=float, default=12.0)
    parser.add_argument('--abort-available-memory-gib', type=float, default=2.0)
    parser.add_argument('--worker-timeout-seconds', type=float, default=1800.0)
    parser.add_argument('--disk-reserve-gib', type=float, default=100.0)
    parser.add_argument(
        '--skip-qualification',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Skip per-shard qualification only after a smoke/qualification run passed this recipe.',
    )
    args = parser.parse_args()

    if args.fingerprint_repo_root:
        os.environ[FINGERPRINT_ROOT_ENV] = str(Path(args.fingerprint_repo_root).expanduser())
    recipe = load_task_recipe(args.recipe, scene_profile=args.scene_profile)
    if args.layout_seeds is None:
        args.layout_seeds = [
            int(seed) for seed in (recipe.get('collection') or {}).get('layout_seeds', [])
        ]
    if (
        args.num_episodes <= 0
        or args.num_workers <= 0
        or args.num_workers > args.num_episodes
        or (
            args.max_concurrent_workers is not None
            and (
                args.max_concurrent_workers <= 0
                or args.max_concurrent_workers > args.num_workers
            )
        )
        or not args.gpu_ids
        or not args.layout_seeds
        or args.seed_stride <= 0
        or args.max_attempts_per_shard <= 0
        or args.max_restarts < 0
        or args.restart_delay_seconds < 0
        or args.poll_seconds <= 0
    ):
        parser.error('Invalid episode, worker, GPU, seed, restart, poll, or layout configuration.')

    with _exclusive_lock(Path(args.output_dir).resolve() / '.parallel_collection.lock.d'):
        manifest = run(args)
    print(
        json.dumps(
            {
                'complete': bool(manifest.get('complete', False)),
                'num_successful': int(manifest.get('num_successful', 0)),
                'num_failed_attempts': int(manifest.get('num_failed_attempts', 0)),
                'manifest': str(Path(args.output_dir).resolve() / 'collection_manifest.json'),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
