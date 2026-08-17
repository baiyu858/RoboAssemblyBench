from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = (
    'beam',
    'car',
    'cooling_manifold',
    'duct',
    'gamepad',
    'stool_circular',
)


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


def _recipe_request(task: str) -> str:
    task = str(task).strip()
    if task.startswith('fabrica_'):
        return task
    return f'fabrica_{task}_ur5e_staged'


def _conda_executable() -> str:
    """Resolve Conda for both interactive shells and detached workers."""
    candidates = []
    configured = os.environ.get('CONDA_EXE')
    if configured:
        candidates.append(Path(configured))
    discovered = shutil.which('conda')
    if discovered:
        candidates.append(Path(discovered))

    prefix = Path(sys.prefix)
    candidates.extend(
        [
            prefix.parent / 'bin' / 'conda',
            prefix.parent.parent / 'bin' / 'conda',
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        'Conda executable was not found. Set CONDA_EXE or activate the configured environment.'
    )


def _worker_command(args, recipes: list[str], results_path: Path) -> list[str]:
    conda = _conda_executable()
    command = [
        conda,
        'run',
        '--no-capture-output',
        '-n',
        args.conda_env,
        'env',
        'PYTHONNOUSERSITE=1',
        'PYTHONUNBUFFERED=1',
        f'OMP_NUM_THREADS={int(args.num_threads)}',
        f'MKL_NUM_THREADS={int(args.num_threads)}',
        f'OPENBLAS_NUM_THREADS={int(args.num_threads)}',
        'python',
        str(REPO_ROOT / 'roboassemblybench' / 'scripts' / 'generate_demos.py'),
        '--worker-mode',
        'collect',
        '--worker-recipes',
        *recipes,
        '--worker-seeds',
        str(int(args.seed)),
        '--worker-layout-seeds',
        str(int(args.layout_seed)),
        '--worker-scene-profile',
        args.scene_profile,
        '--worker-results-path',
        str(results_path),
        '--output-dir',
        str(Path(args.output_dir).resolve()),
        '--headless',
        '--skip-episode-steps',
        '--runtime-constraint-monitor',
        '--constraint-check-stride',
        str(int(args.constraint_check_stride)),
        '--assembly-sequence-precheck',
        '--stage-trajectory-precheck',
        '--stage-precheck-stride',
        str(int(args.stage_precheck_stride)),
        '--stage-precheck-waypoints',
        str(int(args.stage_precheck_waypoints)),
    ]
    if args.domain_randomization:
        command.append('--domain-randomization')
    if args.record_live_video:
        command.extend(
            [
                '--record-live-video',
                '--live-video-fps',
                str(int(args.live_video_fps)),
                '--live-video-frame-stride',
                str(int(args.live_video_frame_stride)),
            ]
        )
    else:
        command.extend(
            [
                '--worker-rendering-interval',
                str(int(args.rendering_interval)),
            ]
        )
    return command


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_worker(command: list[str], *, log_path: Path, args) -> tuple[int, dict | None, dict]:
    started_at = time.monotonic()
    minimum_available = _available_memory_gib()
    low_memory_polls = 0
    resource_abort = None
    next_status_at = started_at
    with log_path.open('w', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                available_memory = _available_memory_gib()
                minimum_available = min(minimum_available, available_memory)
                if available_memory < float(args.abort_available_memory_gib):
                    low_memory_polls += 1
                else:
                    low_memory_polls = 0
                elapsed = time.monotonic() - started_at
                if low_memory_polls >= int(args.low_memory_grace_polls):
                    resource_abort = {
                        'reason': 'low-available-memory',
                        'available_memory_gib': available_memory,
                        'threshold_gib': float(args.abort_available_memory_gib),
                        'consecutive_polls': low_memory_polls,
                    }
                elif elapsed >= float(args.worker_timeout_seconds):
                    resource_abort = {
                        'reason': 'worker-wall-timeout',
                        'elapsed_seconds': elapsed,
                        'threshold_seconds': float(args.worker_timeout_seconds),
                    }
                if resource_abort is not None:
                    _terminate_process_group(process)
                    break
                now = time.monotonic()
                if now >= next_status_at:
                    print(
                        '[validation] '
                        f'elapsed={elapsed:.0f}s available_memory={available_memory:.2f}GiB '
                        f'log={log_path}',
                        flush=True,
                    )
                    next_status_at = now + 30.0
                time.sleep(max(float(args.resource_poll_seconds), 1.0))
        except BaseException:
            _terminate_process_group(process)
            raise
    elapsed = max(time.monotonic() - started_at, 0.0)
    return int(process.wait()), resource_abort, {
        'elapsed_seconds': elapsed,
        'minimum_available_memory_gib': minimum_available,
    }


def _phase_at_step(result: dict, step: int | None) -> str | None:
    phase_history = list(result.get('phase_history') or [])
    phase = next((item for item in phase_history if item not in {'failed', 'complete'}), None)
    if step is None:
        return phase
    for transition in result.get('phase_transition_history') or []:
        transition_step = transition.get('step_counter')
        if transition_step is None or int(transition_step) > int(step):
            continue
        candidate = transition.get('to_phase')
        if candidate not in {None, 'failed', 'complete'}:
            phase = candidate
    return phase


def _terminal_phase(result: dict) -> tuple[str | None, int | None]:
    transitions = list(result.get('phase_transition_history') or [])
    if transitions:
        transition = transitions[-1]
        phase = transition.get('to_phase')
        phase_index = transition.get('to_phase_index')
        if phase in {'failed', 'complete', None}:
            phase = transition.get('from_phase')
            phase_index = transition.get('from_phase_index')
        return phase, None if phase_index is None else int(phase_index)
    phase_history = list(result.get('phase_history') or [])
    phase = next((item for item in reversed(phase_history) if item not in {'failed', 'complete'}), None)
    return phase, None


def _collision_summary(result: dict, metric_key: str, events_key: str = 'events') -> dict:
    report = result.get(metric_key) or {}
    phase_counts = Counter()
    for event in report.get(events_key) or []:
        phase = _phase_at_step(result, event.get('step')) or 'unknown'
        phase_counts[phase] += 1
    return {
        'checks': int(report.get('checks', 0)),
        'violation_total': int(report.get('violation_total', 0)),
        'minimum_distance': report.get('minimum_distance'),
        'violations_by_phase': dict(phase_counts),
        'monitor_error': list(report.get('monitor_error') or []),
    }


def _summarize_results(
    *,
    requested_recipes: list[str],
    results: list[dict],
    worker_exit_code: int,
    resource_abort: dict | None,
    resource_usage: dict,
) -> dict:
    task_results = []
    for index, result in enumerate(results):
        terminal_phase, terminal_phase_index = _terminal_phase(result)
        task_results.append(
            {
                'recipe_request': requested_recipes[index] if index < len(requested_recipes) else None,
                'recipe': result.get('recipe')
                or (requested_recipes[index] if index < len(requested_recipes) else None),
                'seed': result.get('seed'),
                'success': bool(result.get('success', False)),
                'failed': bool(result.get('failed', False)),
                'steps': int(result.get('steps', 0)),
                'terminal_reason': result.get('terminal_reason'),
                'terminal_phase': terminal_phase,
                'terminal_phase_index': terminal_phase_index,
                'phase_status': result.get('phase_status'),
                'timeout_count': int(result.get('timeout_count', 0)),
                'recovery_count': int(result.get('recovery_count', 0)),
                'runtime_collisions': _collision_summary(result, 'runtime_constraint_monitor'),
                'stage_precheck_collisions': _collision_summary(result, 'stage_trajectory_precheck'),
                'assembly_sequence_precheck': result.get('assembly_sequence_precheck') or {},
            }
        )
    missing_recipes = requested_recipes[len(task_results) :]
    return {
        'schema_version': 'roboassemblybench_fabrica_canonical_validation_v1',
        'requested_recipes': requested_recipes,
        'worker_exit_code': int(worker_exit_code),
        'resource_abort': resource_abort,
        'resource_usage': resource_usage,
        'num_requested': len(requested_recipes),
        'num_completed': len(task_results),
        'num_successful': sum(item['success'] for item in task_results),
        'missing_recipes': missing_recipes,
        'runtime_collision_total': sum(
            item['runtime_collisions']['violation_total'] for item in task_results
        ),
        'stage_precheck_collision_total': sum(
            item['stage_precheck_collisions']['violation_total'] for item in task_results
        ),
        'tasks': task_results,
    }


def _tail(path: Path, line_count: int = 80) -> str:
    if not path.is_file():
        return ''
    return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-line_count:])


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate the canonical Fabrica UR5e tasks in one Isaac process.')
    parser.add_argument('--tasks', nargs='+', default=list(DEFAULT_TASKS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layout-seed', type=int, default=0)
    parser.add_argument('--scene-profile', default='taoyuan_grscenes_tabletop')
    parser.add_argument('--conda-env', default='internutopia311')
    parser.add_argument('--domain-randomization', action='store_true')
    parser.add_argument(
        '--output-dir',
        default=str(REPO_ROOT / 'outputs' / 'fabrica_canonical_ur5e_validation'),
    )
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--rendering-interval', type=int, default=2399)
    parser.add_argument('--record-live-video', action='store_true')
    parser.add_argument('--live-video-fps', type=int, default=30)
    parser.add_argument('--live-video-frame-stride', type=int, default=8)
    parser.add_argument('--minimum-start-memory-gib', type=float, default=2.5)
    parser.add_argument('--abort-available-memory-gib', type=float, default=1.5)
    parser.add_argument('--low-memory-grace-polls', type=int, default=2)
    parser.add_argument('--resource-poll-seconds', type=float, default=1.0)
    parser.add_argument('--worker-timeout-seconds', type=float, default=10800.0)
    parser.add_argument('--constraint-check-stride', type=int, default=32)
    parser.add_argument('--stage-precheck-stride', type=int, default=128)
    parser.add_argument('--stage-precheck-waypoints', type=int, default=4)
    args = parser.parse_args()

    if args.live_video_fps <= 0 or args.live_video_frame_stride <= 0:
        parser.error('--live-video-fps and --live-video-frame-stride must be positive.')

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'validation_results.json'
    summary_path = output_dir / 'validation_summary.json'
    log_path = output_dir / 'validation.log'
    results_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)

    available_memory = _available_memory_gib()
    if available_memory < float(args.minimum_start_memory_gib):
        raise RuntimeError(
            f'Only {available_memory:.2f} GiB memory is available; '
            f'{float(args.minimum_start_memory_gib):.2f} GiB is required to start Isaac safely.'
        )

    recipes = [_recipe_request(task) for task in args.tasks]
    command = _worker_command(args, recipes, results_path)
    _write_json_atomic(
        output_dir / 'validation_config.json',
        {
            'recipes': recipes,
            'seed': int(args.seed),
            'layout_seed': int(args.layout_seed),
            'scene_profile': args.scene_profile,
            'command': command,
        },
    )
    print(f'[validation] starting {len(recipes)} recipes in one worker', flush=True)
    worker_exit_code, resource_abort, resource_usage = _run_worker(command, log_path=log_path, args=args)
    results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.is_file() else []
    summary = _summarize_results(
        requested_recipes=recipes,
        results=results,
        worker_exit_code=worker_exit_code,
        resource_abort=resource_abort,
        resource_usage=resource_usage,
    )
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if resource_abort is not None or worker_exit_code != 0 or summary['num_completed'] != len(recipes):
        log_tail = _tail(log_path)
        if log_tail:
            print(f'\n[validation] worker log tail:\n{log_tail}', file=sys.stderr, flush=True)
        return 2
    return 0 if summary['num_successful'] == len(recipes) else 1


if __name__ == '__main__':
    raise SystemExit(main())
