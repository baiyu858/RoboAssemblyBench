#!/usr/bin/env python3
"""Run bounded stage-2 Fabrica replays as stage-1 successes become available.

This supervisor deliberately owns only replay processes.  Existing stage-1
collectors keep running unchanged, while each committed Stage-1 success is
rendered with the visual randomization profiles one at a time. A node-local
lock, resource guards, and a retry backoff keep the replay side from competing
with collection.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(os.environ.get('RAB_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
REPLAY_SCRIPT = Path(
    os.environ.get(
        'RAB_REPLAY_SCRIPT',
        REPO_ROOT / 'roboassemblybench' / 'scripts' / 'replay_fabrica_successful_trajectories.py',
    )
).resolve()
VISUAL_PROFILES = ('object_distractors', 'texture', 'lighting', 'table_color', 'scene')
PAUSE_MARKER_NAME = 'STAGE2_PAUSED_BY_OPERATOR.json'


@dataclass(frozen=True)
class ReplayJob:
    task: str
    profile: str
    shard_name: str
    source_dir: Path
    output_dir: Path
    recipe: str
    scene_profile: str
    target_episodes: int
    ready_at: float

    @property
    def key(self) -> str:
        return f'{self.task}/{self.profile}/{self.shard_name}'


@dataclass
class ActiveReplay:
    job: ReplayJob
    gpu: int
    slot: int
    process: subprocess.Popen[Any]
    log_handle: Any
    started_at_unix: float


_active_children: dict[str, ActiveReplay] = {}
_stop_requested = False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def _manifest_success_count(payload: dict[str, Any]) -> int:
    successful = payload.get('successful_episodes') or {}
    return int(payload.get('num_successful', len(successful)))


def _stage1_replay_target(payload: dict[str, Any]) -> int:
    """Return the number of source demos committed by the Stage-1 manifest.

    ``successful_episodes`` is the authoritative source set: each entry has
    already passed the Stage-1 quality checks. Do not use the shard's configured
    target here, because Stage-2 is intentionally allowed to trail an active
    shard and incrementally render its growing set of successes.
    """
    successful = payload.get('successful_episodes')
    return len(successful) if isinstance(successful, dict) else 0


def _replay_complete(path: Path, target: int) -> bool:
    payload = _load_json(path)
    return bool(payload and payload.get('complete') and _manifest_success_count(payload) >= target)


def _read_source_contract(source_dir: Path, payload: dict[str, Any]) -> tuple[str, str] | None:
    successful = payload.get('successful_episodes') or {}
    if not isinstance(successful, dict):
        return None
    for item in successful.values():
        if not isinstance(item, dict):
            continue
        metadata_path = Path(str(item.get('metadata_path') or ''))
        if not metadata_path.is_file():
            continue
        metadata = _load_json(metadata_path)
        if metadata is None:
            continue
        recipe = str(metadata.get('recipe') or '')
        scene_profile = str(metadata.get('scene_profile') or '')
        if recipe and scene_profile:
            return recipe, scene_profile
    return None


def _discover_jobs(output_root: Path) -> list[ReplayJob]:
    jobs: list[ReplayJob] = []
    for manifest_path in sorted(output_root.glob('stage1/*/shards/shard_*/collection_manifest.json')):
        stage1 = _load_json(manifest_path)
        if stage1 is None:
            continue
        target = _stage1_replay_target(stage1)
        if target <= 0:
            continue
        contract = _read_source_contract(manifest_path.parent, stage1)
        if contract is None:
            continue
        recipe, scene_profile = contract
        task = manifest_path.parents[2].name
        shard_name = manifest_path.parent.name
        for profile in VISUAL_PROFILES:
            output_dir = output_root / 'rendered' / task / profile / 'shards' / shard_name
            if _replay_complete(output_dir / 'replay_manifest.json', target):
                continue
            jobs.append(
                ReplayJob(
                    task=task,
                    profile=profile,
                    shard_name=shard_name,
                    source_dir=manifest_path.parent,
                    output_dir=output_dir,
                    recipe=recipe,
                    scene_profile=scene_profile,
                    target_episodes=target,
                    ready_at=manifest_path.stat().st_mtime,
                )
            )
    return sorted(jobs, key=lambda job: (job.ready_at, job.task, job.shard_name, VISUAL_PROFILES.index(job.profile)))


def _available_memory_gib() -> float:
    with Path('/proc/meminfo').open(encoding='utf-8') as handle:
        values = {
            key.rstrip(':'): int(value) for key, value, *_ in (line.split() for line in handle) if value.isdigit()
        }
    return float(values.get('MemAvailable', 0)) / (1024.0 * 1024.0)


def _gpu_free_mib(gpu_ids: list[int]) -> dict[int, int]:
    command = [
        'nvidia-smi',
        '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {}
    result: dict[int, int] = {}
    allowed = set(gpu_ids)
    for line in output.splitlines():
        try:
            index_text, free_text = (part.strip() for part in line.split(',', maxsplit=1))
            index = int(index_text)
            if index in allowed:
                result[index] = int(free_text)
        except (TypeError, ValueError):
            continue
    return result


def _cache_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / float(1024**3)


def _resource_ready(
    args: argparse.Namespace,
    *,
    occupied_slots: set[tuple[int, int]],
) -> tuple[int | None, int | None, str]:
    if _available_memory_gib() < args.min_available_memory_gib:
        return None, None, 'host_memory_guard'
    if _cache_free_gib(args.portable_base) < args.min_cache_free_gib:
        return None, None, 'cache_disk_guard'
    if _cache_free_gib(args.output_root) < args.min_output_free_gib:
        return None, None, 'output_disk_guard'
    cpu_count = max(1, os.cpu_count() or 1)
    if os.getloadavg()[0] > args.max_load_per_cpu * cpu_count:
        return None, None, 'cpu_load_guard'
    free_by_gpu = _gpu_free_mib(args.gpu_ids)
    active_by_gpu = {
        gpu: sum((gpu, slot) in occupied_slots for slot in range(args.max_replays_per_gpu))
        for gpu in args.gpu_ids
    }
    eligible = [
        (active_by_gpu[gpu], free_mib, gpu)
        for gpu, free_mib in free_by_gpu.items()
        if active_by_gpu[gpu] < args.max_replays_per_gpu and free_mib >= args.min_gpu_free_mib
    ]
    if not eligible:
        return None, None, 'gpu_memory_guard'
    # Fill every GPU evenly before assigning a second replay to any of them.
    _, _, gpu = min(eligible, key=lambda item: (item[0], -item[1], item[2]))
    slot = next(slot for slot in range(args.max_replays_per_gpu) if (gpu, slot) not in occupied_slots)
    return gpu, slot, 'ready'


def _command_for_job(args: argparse.Namespace, job: ReplayJob, gpu: int) -> list[str]:
    command = [str(args.isaac_python), str(REPLAY_SCRIPT)]
    command.extend(
        [
            '--source-dir',
            str(job.source_dir),
            '--output-dir',
            str(job.output_dir),
            '--num-episodes',
            str(job.target_episodes),
            '--batch-size',
            str(args.batch_size),
            '--isaac-python',
            str(args.isaac_python),
            '--recipe',
            job.recipe,
            '--scene-profile',
            job.scene_profile,
            '--randomization-profile',
            job.profile,
            '--rendering-fps',
            '80',
            '--video-codec',
            args.video_codec,
            '--video-crf',
            str(args.video_crf),
            '--video-preset',
            args.video_preset,
            '--depth-compression-level',
            str(args.depth_compression_level),
            '--require-visual-quality',
            '--min-available-memory-gib',
            str(args.min_available_memory_gib),
            '--abort-available-memory-gib',
            str(args.abort_available_memory_gib),
            '--worker-timeout-seconds',
            str(args.worker_timeout_seconds),
            '--worker-stall-timeout-seconds',
            str(args.worker_stall_timeout_seconds),
            '--disk-reserve-gib',
            str(args.min_output_free_gib),
        ]
    )
    # Replays yield CPU scheduling priority to the always-on stage-1 collectors.
    return ['nice', '-n', str(args.nice_increment), *command]


def _job_environment(args: argparse.Namespace, job: ReplayJob, gpu: int, slot: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            'PYTHONNOUSERSITE': '1',
            'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'RAB_FFMPEG_THREADS': '1',
            'ISAACSIM_ACTIVE_GPU': str(gpu),
            'ISAACSIM_PHYSICS_GPU': str(gpu),
            'ISAACSIM_THREAD_COUNT': '1',
            'ISAAC_SIM_ROOT': str(args.isaac_python.parent),
            'ISAAC_ASSETS_ROOT': str(args.assets_root),
            # Each concurrent Isaac process needs its own writable Kit/cache
            # area. A slot is reused only after its replay process exits.
            'ISAACSIM_PORTABLE_ROOT': str(args.portable_base / f'pipeline_stage2_gpu_{gpu}_slot_{slot}'),
            'PYTHONPATH': f'{REPO_ROOT / ".runtime_python"}:{args.isaac_python.parent / "python_packages"}:{REPO_ROOT}'
            + (f':{environment["PYTHONPATH"]}' if environment.get('PYTHONPATH') else ''),
        }
    )
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    return environment


def _log(path: Path, event: str, **fields: Any) -> None:
    payload = {'timestamp_unix': time.time(), 'event': event, **fields}
    line = json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def _terminate_active_child(_signum: int, _frame: Any) -> None:
    global _stop_requested
    _stop_requested = True
    for active in list(_active_children.values()):
        if active.process.poll() is not None:
            continue
        try:
            os.killpg(active.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue


def _load_state(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {'schema_version': 'fabrica_stage2_pipeline_v1', 'jobs': {}}
    payload.setdefault('jobs', {})
    return payload


def _select_job(
    jobs: list[ReplayJob],
    state: dict[str, Any],
    now: float,
    *,
    active_job_keys: set[str],
) -> ReplayJob | None:
    for job in jobs:
        if job.key in active_job_keys:
            continue
        record = state['jobs'].get(job.key, {})
        if float(record.get('retry_after_unix', 0.0)) <= now:
            return job
    return None


def _record_finished_job(
    args: argparse.Namespace,
    state: dict[str, Any],
    event_log: Path,
    active: ActiveReplay,
) -> None:
    return_code = active.process.returncode
    active.log_handle.close()
    replay_done = _replay_complete(active.job.output_dir / 'replay_manifest.json', active.job.target_episodes)
    record = state['jobs'].setdefault(active.job.key, {})
    record.update({'last_return_code': return_code, 'last_finished_at_unix': time.time()})
    if replay_done:
        record.update({'status': 'complete', 'retry_after_unix': 0.0})
        _log(event_log, 'job_complete', job=active.job.key, gpu=active.gpu)
        return
    failures = int(record.get('supervisor_failures', 0)) + 1
    record.update(
        {
            'status': 'retry_backoff',
            'supervisor_failures': failures,
            'retry_after_unix': time.time() + args.failure_backoff_seconds,
        }
    )
    _log(
        event_log,
        'job_incomplete',
        job=active.job.key,
        gpu=active.gpu,
        return_code=return_code,
        failures=failures,
    )


def _reap_finished_children(args: argparse.Namespace, state: dict[str, Any], event_log: Path) -> None:
    for job_key, active in list(_active_children.items()):
        if active.process.poll() is None:
            continue
        _record_finished_job(args, state, event_log, active)
        del _active_children[job_key]


def _active_state() -> list[dict[str, Any]]:
    return [
        {
            'job': active.job.key,
            'gpu': active.gpu,
            'slot': active.slot,
            'pid': active.process.pid,
            'started_at_unix': active.started_at_unix,
        }
        for active in sorted(_active_children.values(), key=lambda item: (item.gpu, item.job.key))
    ]


def _request_active_children_stop() -> None:
    for active in list(_active_children.values()):
        if active.process.poll() is not None:
            continue
        try:
            os.killpg(active.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue


def _pause_requested(output_root: Path) -> bool:
    """Return whether an operator has paused replay dispatch for this dataset."""
    return (output_root / PAUSE_MARKER_NAME).is_file()


def _run(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.portable_base.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.runtime_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    # NFS locking can block indefinitely during a transient lock-service fault.
    # The supervisor is node-local, so keep control-plane files on its local disk.
    lock_path = args.runtime_dir / '.fabrica_stage2_pipeline.lock'
    state_path = args.runtime_dir / 'stage2_pipeline_state.json'
    event_log = logs_dir / 'stage2_pipeline.log'

    with lock_path.open('w', encoding='utf-8') as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Another stage-2 pipeline supervisor already owns {args.output_root}.')

        signal.signal(signal.SIGTERM, _terminate_active_child)
        signal.signal(signal.SIGINT, _terminate_active_child)
        _log(
            event_log,
            'supervisor_start',
            output_root=str(args.output_root),
            runtime_dir=str(args.runtime_dir),
            gpu_ids=args.gpu_ids,
        )
        while True:
            state = _load_state(state_path)
            _reap_finished_children(args, state, event_log)
            paused = _pause_requested(args.output_root)
            if _stop_requested or paused:
                _request_active_children_stop()
                if not _active_children:
                    _log(
                        event_log,
                        'supervisor_paused' if paused else 'supervisor_stop_requested',
                        pause_marker=str(args.output_root / PAUSE_MARKER_NAME) if paused else None,
                    )
                    return
                state.update(
                    {
                        'updated_at_unix': time.time(),
                        'status': 'paused' if paused else 'stopping',
                        'active_jobs': _active_state(),
                    }
                )
                _write_json_atomic(state_path, state)
                time.sleep(1)
                continue
            now = time.time()
            jobs = _discover_jobs(args.output_root)
            resource_state = 'ready'
            while len(_active_children) < args.max_concurrent_replays:
                gpu, slot, resource_state = _resource_ready(
                    args,
                    occupied_slots={(active.gpu, active.slot) for active in _active_children.values()},
                )
                job = (
                    _select_job(jobs, state, now, active_job_keys=set(_active_children)) if gpu is not None else None
                )
                if job is None or slot is None:
                    break
                job_log = logs_dir / f'replay_{job.task}_{job.profile}_{job.shard_name}.log'
                command = _command_for_job(args, job, gpu)
                _log(event_log, 'job_start', job=job.key, gpu=gpu, command=command)
                if args.dry_run:
                    return
                handle = job_log.open('a', encoding='utf-8')
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=_job_environment(args, job, gpu, slot),
                    start_new_session=True,
                )
                _active_children[job.key] = ActiveReplay(
                    job=job,
                    gpu=gpu,
                    slot=slot,
                    process=process,
                    log_handle=handle,
                    started_at_unix=time.time(),
                )
            state.update(
                {
                    'updated_at_unix': time.time(),
                    'status': 'running' if _active_children else resource_state,
                    'active_jobs': _active_state(),
                    'ready_jobs': [candidate.key for candidate in jobs],
                }
            )
            _write_json_atomic(state_path, state)
            if args.once:
                return
            time.sleep(args.poll_seconds)


def _parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in value.split(',') if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('GPU_IDS must be comma-separated non-negative integers.') from error
    if not gpu_ids or any(gpu < 0 for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError('GPU_IDS must be a non-empty unique list of non-negative integers.')
    return gpu_ids


def main() -> None:
    parser = argparse.ArgumentParser(description='Pipeline bounded visual replays after Stage-1 successes become available.')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--isaac-python', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--gpu-ids', type=_parse_gpu_ids, required=True)
    parser.add_argument('--portable-base', type=Path, default=Path('/tmp/roboassemblybench_stage2'))
    parser.add_argument(
        '--runtime-dir',
        type=Path,
        default=None,
        help='Node-local directory for the supervisor lock, state, and detailed logs.',
    )
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument(
        '--max-concurrent-replays',
        type=int,
        default=1,
        help='Node-wide cap for Stage-2 Isaac replay processes.',
    )
    parser.add_argument(
        '--max-replays-per-gpu',
        type=int,
        default=1,
        help='Maximum number of Stage-2 replay processes assigned to one GPU.',
    )
    parser.add_argument('--min-gpu-free-mib', type=int, default=16384)
    parser.add_argument('--min-available-memory-gib', type=float, default=64.0)
    parser.add_argument('--abort-available-memory-gib', type=float, default=48.0)
    parser.add_argument('--min-cache-free-gib', type=float, default=24.0)
    parser.add_argument('--min-output-free-gib', type=float, default=256.0)
    parser.add_argument('--max-load-per-cpu', type=float, default=0.85)
    parser.add_argument('--failure-backoff-seconds', type=int, default=1800)
    parser.add_argument('--worker-timeout-seconds', type=int, default=7200)
    parser.add_argument('--worker-stall-timeout-seconds', type=int, default=900)
    parser.add_argument('--video-codec', choices=['h264', 'h265'], default='h265')
    parser.add_argument('--video-crf', type=int, default=30)
    parser.add_argument('--video-preset', default='veryfast')
    parser.add_argument('--depth-compression-level', type=int, default=8)
    parser.add_argument('--nice-increment', type=int, default=15)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.isaac_python = args.isaac_python.resolve()
    args.assets_root = args.assets_root.resolve()
    args.portable_base = args.portable_base.resolve()
    args.runtime_dir = (args.runtime_dir or args.portable_base / 'pipeline_runtime').resolve()
    if not args.isaac_python.is_file():
        parser.error(f'--isaac-python does not exist: {args.isaac_python}')
    if not REPLAY_SCRIPT.is_file():
        parser.error(f'Replay script is missing: {REPLAY_SCRIPT}')
    if (
        args.poll_seconds < 10
        or args.batch_size < 1
        or args.max_concurrent_replays < 1
        or args.max_replays_per_gpu < 1
        or args.max_concurrent_replays > len(args.gpu_ids) * args.max_replays_per_gpu
        or args.max_load_per_cpu <= 0
        or args.nice_increment < 0
    ):
        parser.error(
            'poll-seconds must be >= 10; batch-size, max-concurrent-replays, and max-replays-per-gpu must be '
            'positive; max-concurrent-replays cannot exceed --gpu-ids times --max-replays-per-gpu; '
            'max-load-per-cpu must be positive; '
            'nice-increment must be non-negative.'
        )
    _run(args)


if __name__ == '__main__':
    main()
