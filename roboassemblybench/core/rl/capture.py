"""Execute the normal predecessor flow and capture an arbitrary insertion entry."""
from dataclasses import asdict
from pathlib import Path
import time
import json

from roboassemblybench.core.scene_profiles import deep_merge
from .control import InsertionConfig
from .recipe import infer_binding
from .snapshot import save_snapshot
from .task import InsertionTask


def policy_overrides(payload, stage_id, fields):
    settings = {'precision_insertion': {str(stage_id): fields}}
    return {'fabrica_canonical': settings} if payload.get('fabrica_canonical') is not None else settings


def requested_stage_settings(payload, stage_id):
    active = payload.get('precision_insertion_resolved', {}).get(str(stage_id))
    if active is not None:
        return dict(active)
    from .validate import QualificationConfig
    raw = (payload.get('fabrica_canonical') or payload).get('precision_insertion', {}).get(str(stage_id), {})
    return dict(raw, controller_config=asdict(InsertionConfig(**raw.get('controller_config', {}))),
                qualification_config=asdict(QualificationConfig(**raw.get('qualification_config', {}))))


def request_binding(payload, stage_id=None, explicit=None):
    if explicit:
        return InsertionTask.from_dict(explicit)
    stages = payload.get('precision_insertion_resolved', {})
    if stage_id is None:
        if len(stages) != 1:
            raise ValueError('Specify --stage or --binding for the insertion to capture')
        stage_id = next(iter(stages))
    if str(stage_id) in stages:
        return InsertionTask.from_dict(stages[str(stage_id)]['binding'])
    settings = requested_stage_settings(payload, stage_id)
    return infer_binding(payload, str(stage_id), settings.get('entry_waypoint', 1), settings.get('binding'))


def capture_entry(recipe, *, stage_id=None, explicit_binding=None, output_dir, seed=4906, layout_seed=None):
    from .snapshot import json_value
    from roboassemblybench.core.task_registry import load_task_recipe
    from toolkits.factory_dual_franka_assembly.generate_demos import _build_env
    from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode
    from toolkits.factory_dual_franka_assembly.demo_policy import DualFrankaAssemblyDemoPolicy

    payload = load_task_recipe(recipe)
    binding = request_binding(payload, stage_id, explicit_binding)
    settings = requested_stage_settings(payload, binding.stage_id)
    names = [p['name'] for p in payload['phases']]
    entry_index = names.index(binding.rl_phase if binding.rl_phase in names else binding.entry_phase)
    overrides = {}
    # Later policy files need not exist to capture an earlier entry. Previous
    # learned insertions remain enabled when they are needed to reach this one.
    for key, value in payload.get('precision_insertion_resolved', {}).items():
        if names.index(value['phase']) >= entry_index:
            overrides = deep_merge(overrides, policy_overrides(payload, key, {'controller': 'planned'}))
    config = InsertionConfig(**settings.get('controller_config', {}))
    cfg = build_dual_franka_assembly_episode(recipe, seed=seed, layout_seed=layout_seed,
                                             attach_runtime_cameras=False, control_fps=config.physics_hz, recipe_overrides=overrides)
    for robot in cfg.robots:
        robot.sensors = []
    cfg.camera_metadata = []
    env = _build_env([cfg], headless=True, rendering_fps=config.physics_hz, rendering_interval=15)
    output = Path(output_dir)
    started = time.monotonic()
    try:
        env.reset()
        task = next(iter(env.runner.current_tasks.values()))
        policy = DualFrankaAssemblyDemoPolicy()
        while not task.failed and not task.success and task.step_counter < task.max_steps:
            if task.phase == binding.entry_phase:
                save_snapshot(task, output / 'entry_snapshot.json', binding=binding,
                              stage_settings=dict(settings, controller_config=asdict(config)))
                result = dict(scope='entry_capture_only', captured=True, benchmark_success=False,
                              binding=binding.to_dict(), source_step=int(task.step_counter), elapsed_seconds=time.monotonic() - started)
                (output / 'capture_results.json').write_text(json.dumps(result, default=json_value, indent=2))
                return result
            env.step([policy.act(task)])
            if task.step_counter % 240 == 0:
                print(f'[insertion-entry] step={task.step_counter} phase={task.phase}', flush=True)
        raise RuntimeError(f'Entry capture failed at {task.phase}: {task.terminal_reason}')
    except BaseException:
        import traceback
        (output / 'worker_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        env.close()
