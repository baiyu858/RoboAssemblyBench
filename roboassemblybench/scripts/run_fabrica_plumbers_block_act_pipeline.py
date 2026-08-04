from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from roboassemblybench.core.process_lock import (
    exclusive_process_lock,
    process_lock_is_held,
)
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = 'fabrica_plumbers_block_ur5e_right_base_prepare'
DEFAULT_SCENE_PROFILE = 'taoyuan_grscenes_tabletop'
DEFAULT_RAW_DIR = REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_v3'
DEFAULT_DATASET_DIR = REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e_right_base_prepare_2k_lerobot_v3'
DEFAULT_TRAIN_DIR = REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e_right_base_prepare_act'
DEFAULT_EVAL_DIR = REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_ur5e_act_eval_50ep'
DEFAULT_REPO_ID = 'baiyu858/roboassemblybench_fabrica_plumbers_block_ur5e_2k'


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def _lock_is_held(path: Path) -> bool:
    return process_lock_is_held(path.with_name(f'{path.name}.d'))


def _available_memory_gib() -> float:
    values = {}
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, raw_value = line.split(':', 1)
        values[key] = int(raw_value.strip().split()[0])
    return float(values.get('MemAvailable', 0)) / (1024.0**2)


def _validate_collection_manifest(
    manifest: dict[str, Any],
    *,
    expected_episodes: int,
    expected_recipe_fingerprint: str | None = None,
    expected_layout_seeds: list[int] | None = None,
) -> None:
    successful = manifest.get('successful_episodes') or {}
    recipe_fingerprint = str(manifest.get('recipe_fingerprint') or '')
    if not bool(manifest.get('complete', False)):
        raise ValueError('Collection manifest is not complete.')
    if int(manifest.get('target_successful_episodes', -1)) != expected_episodes:
        raise ValueError('Collection target does not match the pipeline target.')
    if int(manifest.get('num_successful', -1)) != expected_episodes or len(successful) != expected_episodes:
        raise ValueError(f'Expected exactly {expected_episodes} successful episodes.')
    if len({int(seed) for seed in successful}) != expected_episodes:
        raise ValueError('Collection manifest contains duplicate successful seeds.')
    if not recipe_fingerprint:
        raise ValueError('Collection manifest does not contain a recipe fingerprint.')
    if expected_recipe_fingerprint is not None and recipe_fingerprint != expected_recipe_fingerprint:
        raise ValueError('Collection manifest recipe fingerprint does not match the current recipe.')
    if any(str(item.get('recipe_fingerprint') or '') != recipe_fingerprint for item in successful.values()):
        raise ValueError('Collection contains episodes from a different recipe fingerprint.')
    if expected_layout_seeds is not None:
        expected_layout_seeds = [int(seed) for seed in expected_layout_seeds]
        manifest_layout_seeds = [int(seed) for seed in manifest.get('collection_layout_seeds') or []]
        if manifest_layout_seeds != expected_layout_seeds:
            raise ValueError('Collection manifest uses a different layout seed contract.')
        allowed_layout_seeds = set(expected_layout_seeds)
        if any(int(item.get('layout_seed', -1)) not in allowed_layout_seeds for item in successful.values()):
            raise ValueError('Collection contains an episode outside the allowed layout seeds.')
    timing = manifest.get('timing_contract') or {}
    expected_timing = {
        'physics_fps': 240,
        'control_fps': 240,
        'dataset_fps': 30,
        'dataset_frame_stride': 8,
        'rendering_interval': 7,
        'camera_render_period_steps': 8,
    }
    if any(int(timing.get(key, -1)) != value for key, value in expected_timing.items()):
        raise ValueError('Collection timing contract is not 240/30/8 with rendering_interval=7.')
    if not bool(timing.get('camera_state_action_aligned', False)):
        raise ValueError('Collection does not claim camera/state/action alignment.')


def _resolve_checkpoint(train_dir: Path) -> Path:
    last = train_dir / 'checkpoints' / 'last' / 'pretrained_model'
    if (last / 'config.json').is_file():
        return last.resolve()
    candidates = sorted(
        (
            path / 'pretrained_model'
            for path in (train_dir / 'checkpoints').glob('[0-9]*')
            if path.name.isdigit() and (path / 'pretrained_model' / 'config.json').is_file()
        ),
        key=lambda path: int(path.parent.name),
    )
    if not candidates:
        raise FileNotFoundError(f'No completed ACT checkpoint was found under {train_dir}.')
    return candidates[-1].resolve()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.raw_dir = Path(args.raw_dir).resolve()
        self.dataset_dir = Path(args.dataset_dir).resolve()
        self.train_dir = Path(args.train_dir).resolve()
        self.eval_dir = Path(args.eval_dir).resolve()
        self.output_dir = Path(args.pipeline_output_dir).resolve()
        self.state_path = self.output_dir / 'pipeline_state.json'
        self.state = _load_json(self.state_path) if self.state_path.is_file() else {'stages': {}}
        configured_fingerprint = getattr(args, 'recipe_fingerprint', None)
        configured_qualification_seeds = getattr(args, 'qualification_seeds', None)
        configured_layout_seeds = getattr(args, 'collection_layout_seeds', None)
        self.qualification_seeds = (
            None if configured_qualification_seeds is None else [int(seed) for seed in configured_qualification_seeds]
        )
        self.collection_layout_seeds = (
            None if configured_layout_seeds is None else [int(seed) for seed in configured_layout_seeds]
        )
        if configured_fingerprint:
            self.recipe_fingerprint = str(configured_fingerprint)
        else:
            recipe = str(getattr(args, 'recipe', DEFAULT_RECIPE))
            scene_profile = str(getattr(args, 'scene_profile', DEFAULT_SCENE_PROFILE))
            recipe_spec = load_task_recipe(recipe, scene_profile=scene_profile)
            self.recipe_fingerprint = str(recipe_spec['recipe_fingerprint'])
            qualification_spec = recipe_spec.get('qualification') or {}
            qualification_count = int(qualification_spec.get('seed_count', 0))
            required_seeds = [int(seed) for seed in qualification_spec.get('required_seeds', [])]
            if qualification_count > 0 and len(required_seeds) >= qualification_count:
                self.qualification_seeds = required_seeds[:qualification_count]
            if self.collection_layout_seeds is None:
                collection_spec = recipe_spec.get('collection') or {}
                self.collection_layout_seeds = [int(seed) for seed in collection_spec.get('layout_seeds', [])] or None
        self._collector_process: subprocess.Popen | None = None

    def _set_stage(self, name: str, status: str, *, replace: bool = False, **details: Any) -> None:
        stages = self.state.setdefault('stages', {})
        payload = {'status': status, 'updated_at_unix': time.time(), **details}
        if replace or name not in stages:
            stages[name] = payload
        else:
            stages[name].update(payload)
        self.state['updated_at_unix'] = time.time()
        _write_json_atomic(self.state_path, self.state)

    def _run(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
        log_path = self.output_dir / f'{name}.log'
        self._set_stage(name, 'running', command=command, log_path=str(log_path))
        with log_path.open('a', encoding='utf-8') as log_file:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            self._set_stage(name, 'failed', returncode=int(completed.returncode))
            raise RuntimeError(f'{name} failed with return code {completed.returncode}; inspect {log_path}.')
        self._set_stage(name, 'completed', returncode=0)

    def _collector_command(self) -> list[str]:
        conda = shutil.which('conda')
        if conda is None:
            raise RuntimeError('conda executable was not found.')
        command = [
            conda,
            'run',
            '--no-capture-output',
            '-n',
            self.args.isaac_env,
            'env',
            'PYTHONNOUSERSITE=1',
            'PYTHONUNBUFFERED=1',
            'python',
            'roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py',
            '--num-episodes',
            str(int(self.args.expected_episodes)),
            '--batch-size',
            '1',
            '--start-seed',
            '0',
            '--max-attempts',
            str(int(self.args.collection_max_attempts)),
            '--recipe',
            str(getattr(self.args, 'recipe', DEFAULT_RECIPE)),
            '--scene-profile',
            str(getattr(self.args, 'scene_profile', DEFAULT_SCENE_PROFILE)),
            '--min-available-memory-gib',
            str(float(self.args.collection_min_available_memory_gib)),
            '--abort-available-memory-gib',
            str(float(self.args.collection_abort_available_memory_gib)),
            '--worker-timeout-seconds',
            str(float(getattr(self.args, 'collection_worker_timeout_seconds', 1800.0))),
            '--output-dir',
            str(self.raw_dir),
        ]
        if self.collection_layout_seeds is not None:
            command.extend(['--layout-seeds', *[str(seed) for seed in self.collection_layout_seeds]])
        return command

    def _start_collector(self) -> subprocess.Popen:
        if self._collector_process is not None:
            if self._collector_process.poll() is None:
                return self._collector_process
            self._collector_process.wait()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        command = self._collector_command()
        log_path = self.raw_dir / 'collector.log'
        env = os.environ.copy()
        with log_path.open('a', encoding='utf-8') as log_file:
            log_file.write(f'Pipeline starting resumable collector at {time.time():.3f}.\n')
            log_file.flush()
            self._collector_process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._set_stage(
            'collection',
            'restarting',
            collector_pid=int(self._collector_process.pid),
            collector_command=command,
            collector_log=str(log_path),
        )
        return self._collector_process

    def wait_for_collection(self) -> dict[str, Any]:
        manifest_path = self.raw_dir / 'collection_manifest.json'
        qualification_status_path = self.raw_dir / 'qualification_status.json'
        collection_lock = self.raw_dir / '.collection.lock'
        while True:
            collection_lock_held = _lock_is_held(collection_lock)
            if qualification_status_path.is_file():
                qualification = _load_json(qualification_status_path)
                qualification_fingerprint = str(qualification.get('recipe_fingerprint') or '')
                current_qualification = qualification_fingerprint == self.recipe_fingerprint
                qualification_seeds = [int(seed) for seed in qualification.get('selected_seeds') or []]
                current_seed_contract = (
                    self.qualification_seeds is None or qualification_seeds == self.qualification_seeds
                )
                if not current_qualification:
                    self._set_stage(
                        'qualification',
                        'superseded',
                        replace=True,
                        status_path=str(qualification_status_path),
                        stale_recipe_fingerprint=qualification_fingerprint,
                        recipe_fingerprint=self.recipe_fingerprint,
                    )
                elif not current_seed_contract:
                    self._set_stage(
                        'qualification',
                        'seed_contract_superseded',
                        replace=True,
                        status_path=str(qualification_status_path),
                        recipe_fingerprint=self.recipe_fingerprint,
                        stale_selected_seeds=qualification_seeds,
                        selected_seeds=self.qualification_seeds,
                    )
                elif bool(qualification.get('failed', False)):
                    failure = qualification.get('failed_result') or {}
                    resource_abort = failure.get('resource_abort')
                    if resource_abort is not None:
                        self._set_stage(
                            'qualification',
                            'recovering_resource_abort',
                            replace=True,
                            status_path=str(qualification_status_path),
                            failed_seed=failure.get('seed'),
                            resource_abort=resource_abort,
                        )
                    else:
                        self._set_stage(
                            'qualification',
                            'failed',
                            replace=True,
                            status_path=str(qualification_status_path),
                            **qualification,
                        )
                        raise RuntimeError(
                            'Recipe qualification failed and automatic collection restart is disabled for '
                            f"this fingerprint: seed={failure.get('seed')} "
                            f"reason={failure.get('terminal_reason')}."
                        )
                if current_qualification and current_seed_contract and bool(qualification.get('passed', False)):
                    self._set_stage(
                        'qualification',
                        'completed',
                        replace=True,
                        status_path=str(qualification_status_path),
                        recipe_fingerprint=qualification.get('recipe_fingerprint'),
                        selected_seeds=qualification.get('selected_seeds') or [],
                    )
                elif current_qualification and current_seed_contract and not bool(qualification.get('failed', False)):
                    self._set_stage(
                        'qualification',
                        'running',
                        replace=True,
                        status_path=str(qualification_status_path),
                        recipe_fingerprint=qualification_fingerprint,
                        selected_seeds=qualification.get('selected_seeds') or [],
                        num_passed=int(qualification.get('num_passed', 0)),
                        num_resource_aborts=int(qualification.get('num_resource_aborts', 0)),
                    )
            if manifest_path.is_file():
                manifest = _load_json(manifest_path)
                successful = len(manifest.get('successful_episodes') or {})
                self._set_stage(
                    'collection',
                    'completed' if manifest.get('complete') else 'waiting',
                    replace=True,
                    successful_episodes=successful,
                    target_episodes=int(self.args.expected_episodes),
                    failed_attempts=len(manifest.get('failed_attempts') or []),
                    manifest_path=str(manifest_path),
                    collector_lock_held=collection_lock_held,
                )
                if manifest.get('complete'):
                    _validate_collection_manifest(
                        manifest,
                        expected_episodes=int(self.args.expected_episodes),
                        expected_recipe_fingerprint=self.recipe_fingerprint,
                        expected_layout_seeds=self.collection_layout_seeds,
                    )
                    if collection_lock_held:
                        time.sleep(float(self.args.poll_seconds))
                        continue
                    return manifest
            else:
                self._set_stage(
                    'collection',
                    'waiting',
                    replace=True,
                    successful_episodes=0,
                    target_episodes=int(self.args.expected_episodes),
                    collector_lock_held=collection_lock_held,
                )
            if not collection_lock_held:
                if bool(getattr(self.args, 'external_collection', False)):
                    self._set_stage(
                        'collection',
                        'waiting_for_external_collector',
                        replace=True,
                        successful_episodes=(
                            len(manifest.get('successful_episodes') or {}) if manifest_path.is_file() else 0
                        ),
                        target_episodes=int(self.args.expected_episodes),
                        manifest_path=str(manifest_path),
                    )
                elif not bool(self.args.supervise_collection):
                    raise RuntimeError('Collector exited before completing the requested episodes.')
                else:
                    self._start_collector()
            time.sleep(float(self.args.poll_seconds))

    def wait_for_resources(self, stage: str) -> None:
        while _available_memory_gib() < float(self.args.min_available_memory_gib):
            self._set_stage(stage, 'waiting_for_memory', available_memory_gib=_available_memory_gib())
            time.sleep(float(self.args.poll_seconds))

    def export(self) -> None:
        summary_path = self.dataset_dir / 'roboassemblybench_export_summary.json'
        if summary_path.is_file():
            summary = _load_json(summary_path)
            if int(summary.get('total_episodes', -1)) == int(self.args.expected_episodes):
                self._set_stage('export', 'completed', summary_path=str(summary_path), resumed=True)
                return
        conda = shutil.which('conda')
        if conda is None:
            raise RuntimeError('conda executable was not found.')
        self._run(
            'export',
            [
                conda,
                'run',
                '--no-capture-output',
                '-n',
                self.args.act_env,
                'env',
                'PYTHONNOUSERSITE=1',
                'OMP_NUM_THREADS=2',
                'MKL_NUM_THREADS=2',
                'python',
                'roboassemblybench/scripts/export_fabrica_plumbers_block_lerobot_v3.py',
                '--input-dir',
                str(self.raw_dir),
                '--output-dir',
                str(self.dataset_dir),
                '--repo-id',
                self.args.dataset_repo_id,
                '--encoder-threads',
                str(int(self.args.encoder_threads)),
                '--resume',
            ],
        )
        summary = _load_json(summary_path)
        if int(summary.get('total_episodes', -1)) != int(self.args.expected_episodes):
            raise RuntimeError('LeRobot export did not produce exactly the requested episode count.')

    def train(self) -> Path:
        if self.state.get('stages', {}).get('train', {}).get('status') == 'completed':
            return _resolve_checkpoint(self.train_dir)
        has_checkpoint = any((self.train_dir / 'checkpoints').glob('*/pretrained_model/config.json'))
        env = os.environ.copy()
        env.update(
            {
                'ACT_ENV': self.args.act_env,
                'DATASET_ROOT': str(self.dataset_dir),
                'DATASET_REPO_ID': self.args.dataset_repo_id,
                'OUTPUT_DIR': str(self.train_dir),
                'STEPS': str(int(self.args.train_steps)),
                'BATCH_SIZE': str(int(self.args.batch_size)),
                'NUM_WORKERS': str(int(self.args.num_workers)),
                'RESUME': 'true' if has_checkpoint else 'false',
            }
        )
        self._run('train', ['bash', 'roboassemblybench/scripts/train_fabrica_plumbers_block_act.sh'], env=env)
        checkpoint = _resolve_checkpoint(self.train_dir)
        self._set_stage('train', 'completed', checkpoint=str(checkpoint), returncode=0)
        return checkpoint

    def evaluate(self, checkpoint: Path) -> None:
        summary_path = self.eval_dir / 'success_rate.json'
        if summary_path.is_file():
            summary = _load_json(summary_path)
            if (
                bool(summary.get('complete'))
                and int(summary.get('num_episodes', -1)) == int(self.args.eval_episodes)
                and (
                    self.collection_layout_seeds is None
                    or [int(seed) for seed in summary.get('layout_seeds') or []] == self.collection_layout_seeds
                )
            ):
                self._set_stage('evaluation', 'completed', summary_path=str(summary_path), resumed=True)
                return
        env = os.environ.copy()
        env.update(
            {
                'ACT_ENV': self.args.act_env,
                'ISAAC_ENV': self.args.isaac_env,
                'CHECKPOINT': str(checkpoint),
                'OUTPUT_DIR': str(self.eval_dir),
                'NUM_EPISODES': str(int(self.args.eval_episodes)),
                'START_SEED': str(int(self.args.eval_start_seed)),
                'LAYOUT_SEEDS': ' '.join(str(seed) for seed in (self.collection_layout_seeds or [])),
            }
        )
        self._run('evaluation', ['bash', 'roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.sh'], env=env)
        summary = _load_json(summary_path)
        if not bool(summary.get('complete')) or int(summary.get('num_episodes', -1)) != int(self.args.eval_episodes):
            raise RuntimeError('Online evaluation did not complete the requested episode count.')

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state['complete'] = False
        self.state.pop('error', None)
        self.state.pop('failed_at_unix', None)
        self.state.pop('finished_at_unix', None)
        self.state['started_at_unix'] = time.time()
        _write_json_atomic(self.state_path, self.state)
        try:
            self.wait_for_collection()
            self.wait_for_resources('export')
            self.export()
            self.wait_for_resources('train')
            checkpoint = self.train()
            self.wait_for_resources('evaluation')
            self.evaluate(checkpoint)
        except Exception as exc:
            self.state['complete'] = False
            self.state['error'] = f'{type(exc).__name__}: {exc}'
            self.state['failed_at_unix'] = time.time()
            _write_json_atomic(self.state_path, self.state)
            raise
        self.state['complete'] = True
        self.state.pop('error', None)
        self.state['finished_at_unix'] = time.time()
        _write_json_atomic(self.state_path, self.state)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the 2k LeRobot v3, ACT, and 50-episode pipeline.')
    parser.add_argument('--raw-dir', default=str(DEFAULT_RAW_DIR))
    parser.add_argument('--dataset-dir', default=str(DEFAULT_DATASET_DIR))
    parser.add_argument('--train-dir', default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument('--eval-dir', default=str(DEFAULT_EVAL_DIR))
    parser.add_argument('--pipeline-output-dir', default=str(REPO_ROOT / 'outputs' / 'fabrica_plumbers_block_pipeline'))
    parser.add_argument('--dataset-repo-id', default=DEFAULT_REPO_ID)
    parser.add_argument('--recipe', default=DEFAULT_RECIPE)
    parser.add_argument('--scene-profile', default=DEFAULT_SCENE_PROFILE)
    parser.add_argument('--expected-episodes', type=int, default=2000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--eval-start-seed', type=int, default=10000)
    parser.add_argument('--train-steps', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--encoder-threads', type=int, default=2)
    parser.add_argument('--poll-seconds', type=float, default=60.0)
    parser.add_argument('--min-available-memory-gib', type=float, default=4.0)
    parser.add_argument('--supervise-collection', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--external-collection',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Wait for an external or parallel collector without launching the built-in collector.',
    )
    parser.add_argument('--collection-max-attempts', type=int, default=10000)
    parser.add_argument('--collection-min-available-memory-gib', type=float, default=5.5)
    parser.add_argument('--collection-abort-available-memory-gib', type=float, default=1.5)
    parser.add_argument('--collection-worker-timeout-seconds', type=float, default=1800.0)
    parser.add_argument('--collection-layout-seeds', type=int, nargs='+', default=None)
    parser.add_argument('--act-env', default='roboassemblybench-act')
    parser.add_argument('--isaac-env', default='internutopia311')
    args = parser.parse_args()

    if (
        args.expected_episodes <= 0
        or args.eval_episodes <= 0
        or args.poll_seconds <= 0
        or args.collection_max_attempts < args.expected_episodes
        or args.collection_worker_timeout_seconds < 0
    ):
        parser.error(
            'Episode counts and poll-seconds must be positive; collection-max-attempts must cover '
            'expected-episodes; collection-worker-timeout-seconds must be non-negative.'
        )
    output_dir = Path(args.pipeline_output_dir).resolve()
    with exclusive_process_lock(output_dir / '.pipeline.lock.d', description='ACT pipeline'):
        Pipeline(args).run()


if __name__ == '__main__':
    main()
