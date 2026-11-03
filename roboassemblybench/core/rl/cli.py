"""Guarded capture, qualification, training and deployment of insertion policies."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / 'roboassemblybench/scripts/run_insertion_rl.py'


def full_flow_passed(results):
    return bool(results and results[0].get('success') and not results[0].get('failed')
                and results[0].get('phase_status') in ('success', 'complete'))


def resumed_insertion_passed(metrics, settle_phase):
    """A verified post-insertion window can finish directly in terminal success."""
    settle = metrics.get('physical_settle') or {}
    return bool(metrics.get('success') and not metrics.get('failed')
                and metrics.get('phase_status') in ('success', 'complete')
                and settle.get('phase') == settle_phase and settle.get('reason') is None
                and settle.get('required_stable_steps', 0) >= 2
                and settle.get('stable_steps', 0) >= settle['required_stable_steps']
                and any(t.get('from_phase') == settle_phase
                and t.get('transition_type') in ('advance', 'terminal')
                for t in metrics.get('phase_transition_history', [])))


def resumed_success_criteria(criteria, end_spec):
    """An intermediate supported check keeps its explicit release ownership.

    The final unloaded gate retains release requirements for every body.
    Position, orientation and static tolerances are left unchanged.
    """
    entries = {entry['object']: entry for entry in end_spec['advance']['objects']}
    result = []
    for original in criteria:
        if original.get('object') not in entries:
            continue
        item = dict(original)
        if end_spec.get('physical_supported_verification'):
            item['require_released'] = entries[item['object']].get('require_released', True)
        result.append(item)
    return result


def post_insertion_phase(phases, binding):
    """Find the real release continuation for an optional seated checkpoint."""
    names = [p['name'] for p in phases]
    replaced = [binding.rl_phase] if binding.rl_phase in names else binding.replace_phases
    if any(name not in names for name in replaced):
        return None
    indices = [names.index(name) for name in replaced if name in names]
    if not indices or max(indices) + 1 >= len(names):
        return None
    return names[max(indices) + 1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('capture', 'validate', 'baseline', 'train', 'play', 'full', 'pipeline', 'resume'), required=True)
    p.add_argument('--recipe', help='Task identifier/path, optionally @profile')
    p.add_argument('--stage', help='Insertion stage/part key from the recipe')
    p.add_argument('--through-stage', help='Resume through this later insertion stage, preserving intermediate support')
    p.add_argument('--binding', type=Path, help='Explicit InsertionTask JSON for a custom task')
    for field in ('snapshot', 'qualification', 'checkpoint', 'output-dir'):
        p.add_argument('--' + field, type=Path)
    p.add_argument('--steps', type=int, default=2048, help='Policy decisions, not physics steps')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seed', type=int, default=4906)
    p.add_argument('--layout-seed', type=int)
    p.add_argument('--video', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.steps < 1 or a.episodes < 1:
        p.error('--steps and --episodes must be positive')
    if a.through_stage is not None and (a.mode != 'resume' or not a.recipe):
        p.error('--through-stage requires --mode resume and --recipe')
    if a.mode in ('capture', 'full') and not a.recipe:
        p.error('--recipe is required')
    if a.mode not in ('capture', 'full') and (not a.snapshot or not a.snapshot.is_file()):
        p.error('A physical --snapshot is required')
    if a.mode in ('play', 'resume') and (not a.checkpoint or not a.checkpoint.is_file()):
        p.error('A trained --checkpoint is required')
    if a.mode == 'train' and (not a.qualification or not a.qualification.is_file()):
        p.error('A passing --qualification is required')
    for field in ('snapshot', 'qualification', 'checkpoint', 'binding'):
        path = getattr(a, field)
        if path:
            if not path.is_file():
                p.error(f'Missing --{field}: {path}')
            setattr(a, field, path.resolve())
    if not a.output_dir:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        a.output_dir = ROOT / 'outputs' / f'insertion_rl_{a.mode}_{stamp}'
    a.output_dir = a.output_dir.resolve()
    return a


def worker(a):
    from .snapshot import json_value, read_snapshot
    explicit = json.loads(a.binding.read_text()) if a.binding else None
    if a.mode == 'capture':
        from .capture import capture_entry
        capture_entry(a.recipe, stage_id=a.stage, explicit_binding=explicit,
                      output_dir=a.output_dir, seed=a.seed, layout_seed=a.layout_seed)
        return
    if a.mode == 'full':
        from .capture import policy_overrides, request_binding
        from .task import InsertionTask
        from roboassemblybench.core.task_registry import load_task_recipe
        from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode
        from toolkits.factory_dual_franka_assembly.generate_demos import _run_task_sequence
        payload = load_task_recipe(a.recipe)
        overrides = None
        if a.checkpoint:
            snapshot = read_snapshot(a.snapshot) if a.snapshot else None
            binding = (InsertionTask.from_dict(snapshot['binding']) if snapshot else request_binding(payload, a.stage, explicit))
            fields = dict(controller='residual_rl', backend='native_contact_insertion', binding=binding.to_dict(),
                          checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
            if snapshot and snapshot.get('stage_settings', {}).get('controller_config'):
                fields['controller_config'] = snapshot['stage_settings']['controller_config']
            overrides = policy_overrides(payload, binding.stage_id, fields)
        effective = load_task_recipe(a.recipe, overrides=overrides)
        rates = {s['physics_hz'] for s in effective.get('precision_insertion_resolved', {}).values()}
        if len(rates) > 1:
            raise ValueError('One physics scene cannot run policies trained at different physics rates')
        physics_hz = next(iter(rates), 240)
        cfg = build_dual_franka_assembly_episode(a.recipe, seed=a.seed, layout_seed=a.layout_seed,
                                                 attach_runtime_cameras=a.video, recipe_overrides=overrides, control_fps=physics_hz)
        if not a.video:
            for robot in cfg.robots:
                robot.sensors = []
            cfg.camera_metadata = []
        _run_task_sequence([cfg], headless=True, output_dir=a.output_dir,
                           results_output_path=a.output_dir / 'full_results.json', record_episode_steps=False,
                           record_live_video=a.video, live_video_fps=15, live_video_frame_stride=16,
                           live_video_encoder='stream', rendering_fps=physics_hz,
                           rendering_interval_override=15 if a.video else 999999)
        return
    from .env import NativeInsertionEnv
    from .validate import qualify, evaluate, QualificationConfig
    env = NativeInsertionEnv(a.snapshot, output_dir=a.output_dir,
                             record_video=a.video and a.mode in ('validate', 'baseline', 'play', 'resume'),
                             recipe=a.recipe if a.mode == 'resume' else None)
    try:
        if a.mode == 'resume':
            from .integration import configure_full_flow
            from toolkits.factory_dual_franka_assembly.demo_policy import DualFrankaAssemblyDemoPolicy
            from roboassemblybench.core.task_registry import load_task_recipe
            env.reset()
            settings = env.snapshot.get('stage_settings', {})
            if a.recipe:
                settings = load_task_recipe(a.recipe).get('precision_insertion_resolved', {}).get(env.binding.stage_id, settings)
            configure_full_flow(env.task.cfg, a.checkpoint, binding=env.binding,
                                controller_config=settings.get('controller_config'), retreat_control=settings.get('retreat_control'),
                                released_rotation_tolerance=settings.get('released_rotation_tolerance'),
                                torque_soft_limit=settings.get('torque_soft_limit'),
                                torque_hard_limit=settings.get('torque_hard_limit'), axis_alignment=settings.get('axis_alignment'))
            task = env.task
            end_binding = env.binding
            if a.through_stage is not None:
                from .capture import request_binding
                end_binding = request_binding(load_task_recipe(a.recipe), a.through_stage)
            end = next(i for i, p in enumerate(task.cfg.phase_specs) if p['name'] == end_binding.settle_phase)
            end_spec = task.cfg.phase_specs[end]
            supported = bool(end_spec.get('physical_supported_verification'))
            index = next(i for i, p in enumerate(task.cfg.phase_specs) if p['name'] == env.binding.rl_phase)
            if end < index:
                raise ValueError('--through-stage precedes the snapshot insertion stage')
            task.phase_specs = task.cfg.phase_specs[:end + 1]
            task.cfg.phase_specs = task.phase_specs
            task.cfg.success_criteria = resumed_success_criteria(task.cfg.success_criteria, end_spec)
            task._set_phase(index, reason='resume-insertion-snapshot', transition_type='resume')
            policy = DualFrankaAssemblyDemoPolicy()
            for _ in range(sum(int(p.get('timeout_steps', 1200)) for p in task.phase_specs[index:]) + 240):
                if task.failed or task.success:
                    break
                env._physics_step(policy.act(task))
            metrics = task.calculate_metrics()
            passed = resumed_insertion_passed(metrics, end_binding.settle_phase)
            from .flow import save_flow_snapshot
            save_flow_snapshot(task, a.output_dir / 'continuation_snapshot.json', resume_phase=end_binding.settle_phase)
            (a.output_dir / 'insertion_resume_results.json').write_text(json.dumps(dict(
                scope=('entry_through_selected_insertion_supported_stability' if supported
                       else 'entry_through_selected_insertion_unloaded_stability'), passed=passed,
                whole_assembly_success=False, support_retained=supported, binding=env.binding.to_dict(),
                end_binding=end_binding.to_dict(), metrics=metrics), default=json_value, indent=2))
            return
        if a.mode == 'baseline':
            import numpy as np
            report = evaluate(env, SimpleNamespace(act=lambda obs: np.zeros(6)), episodes=a.episodes,
                              output_path=a.output_dir / 'local_evaluation.json')
            continuation = post_insertion_phase(env.task.cfg.phase_specs, env.binding)
            if continuation and report['episodes'][-1]['local_seated']:
                from .flow import save_flow_snapshot
                state = save_flow_snapshot(env.task, a.output_dir / 'seated_flow_snapshot.json', resume_phase=continuation)
                state['source_controller'] = 'zero_residual_geometric_guide_and_admittance; training_state_only'
                (a.output_dir / 'seated_flow_snapshot.json').write_text(json.dumps(state, default=json_value, indent=2))
            return
        if a.mode in ('validate', 'pipeline'):
            qualification = a.output_dir / 'qualification.json'
            report = qualify(env, a.snapshot, qualification, config=QualificationConfig(
                **env.snapshot.get('stage_settings', {}).get('qualification_config', {})))
            if not report['passed'] or a.mode == 'validate':
                return
        else:
            qualification = a.qualification
        from .ppo import train, load_policy
        checkpoint = a.checkpoint
        if a.mode in ('train', 'pipeline'):
            checkpoint = train(env, steps=a.steps, output_dir=a.output_dir / 'ppo', qualification_path=qualification,
                               snapshot_path=a.snapshot, seed=a.seed, checkpoint=checkpoint)
            if a.mode == 'train':
                return
        model, config = load_policy(checkpoint, binding=env.binding)
        if config != env.config:
            raise ValueError('Checkpoint and environment controller configurations differ')
        from .control import effective_torque_soft_limit
        env.torque_soft_limit = effective_torque_soft_limit(
            env.torque_soft_limit, getattr(model, 'torque_soft_limit', None), env.config)
        if env.torque_hard_limit is None:
            env.torque_hard_limit = getattr(model, 'torque_hard_limit', None)
        if env.axis_alignment is None:
            env.axis_alignment = getattr(model, 'axis_alignment', None)
        evaluate(env, model, episodes=a.episodes, output_path=a.output_dir / 'local_evaluation.json')
    except BaseException:
        (a.output_dir / 'worker_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        env.close()


def result_passed(a):
    def read(name, default=None):
        path = a.output_dir / name
        return json.loads(path.read_text()) if path.exists() else default
    if (a.output_dir / 'worker_error.txt').exists():
        return False
    if a.mode == 'capture':
        return bool(read('capture_results.json', {}).get('captured') and (a.output_dir / 'entry_snapshot.json').exists())
    if a.mode == 'validate':
        return read('qualification.json', {}).get('passed') is True
    if a.mode == 'train':
        return (a.output_dir / 'ppo/policy.pt').is_file()
    if a.mode == 'full':
        return full_flow_passed(read('full_results.json', []))
    if a.mode == 'resume':
        return read('insertion_resume_results.json', {}).get('passed') is True
    evaluation = read('local_evaluation.json', {})
    return bool(evaluation.get('episode_count') == a.episodes and evaluation.get('local_held_successes') == a.episodes)


def main():
    a = parse_args()
    if not a.worker and a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit(f'Output directory is not empty: {a.output_dir}')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.worker:
        try:
            worker(a)
        except BaseException:
            (a.output_dir / 'worker_error.txt').write_text(traceback.format_exc())
            raise
        return 0 if result_passed(a) else 2
    from roboassemblybench.scripts.validate_fabrica_canonical_ur5e import _run_worker
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PYTHONNOUSERSITE'], os.environ['PYTHONUNBUFFERED'] = '1', '1'
    options = SimpleNamespace(abort_available_memory_gib=1.5, low_memory_grace_polls=3,
                              worker_timeout_seconds=21600, resource_poll_seconds=2)
    command = [sys.executable, str(ENTRY), *sys.argv[1:], '--worker', '--output-dir', str(a.output_dir)]
    code, abort, usage = _run_worker(command, log_path=a.output_dir / 'run.log', args=options)
    passed = code == 0 and abort is None and result_passed(a)
    summary = dict(mode=a.mode, worker_exit_code=code, result_exit_code=0 if passed else 2,
                   passed=passed, resource_abort=abort, resource_usage=usage, output_dir=str(a.output_dir))
    if a.mode == 'pipeline' and passed:
        from .snapshot import read_snapshot
        snapshot = read_snapshot(a.snapshot)
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        full = a.output_dir / f'full_flow_{stamp}'
        full.mkdir()
        command = [sys.executable, str(ENTRY), '--mode', 'full', '--worker', '--recipe', a.recipe or snapshot['recipe'],
                   '--snapshot', str(a.snapshot), '--checkpoint', str(a.output_dir / 'ppo/policy.pt'),
                   '--seed', str(snapshot['seed']), '--layout-seed', str(a.layout_seed if a.layout_seed is not None else snapshot['layout_seed']),
                   '--output-dir', str(full), '--video']
        full_code, full_abort, full_usage = _run_worker(command, log_path=full / 'run.log', args=options)
        full_args = SimpleNamespace(mode='full', output_dir=full)
        passed = full_code == 0 and full_abort is None and result_passed(full_args)
        summary.update(passed=passed, result_exit_code=0 if passed else 2, whole_assembly_success=passed,
                       full_flow=dict(output_dir=str(full), passed=passed, worker_exit_code=full_code,
                                      resource_abort=full_abort, resource_usage=full_usage))
    (a.output_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2
