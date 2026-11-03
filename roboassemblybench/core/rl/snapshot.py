"""Capture/restore contracts for arbitrary physical insertion entry states."""
from __future__ import annotations

import json
from pathlib import Path

from .task import InsertionTask
from . import legacy

FORMAT = 'roboassemblybench-insertion-reset-v2'


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def _validate_grasps(data, binding):
    for robot, name in binding.participants:
        attachment = data['attachments'].get(name, {})
        if (attachment.get('mode') != 'pure_physical_grasp' or attachment.get('robot_name') != robot
                or attachment.get('joint_path') or attachment.get('collision_disabled')
                or attachment.get('kinematic_peel_carry')
                or not attachment.get('attach_spec', {}).get('continuous_physics')):
            raise ValueError(f'{name}: reset contains a missing/nonphysical grasp')
    if any(name not in data['objects'] for name in binding.objects):
        raise ValueError('Reset snapshot lacks insertion bodies')


def save_snapshot(task, path, *, binding, stage_settings=None):
    if task.phase != binding.entry_phase and task.phase != binding.rl_phase:
        raise ValueError(f'Capture requires insertion entry {binding.entry_phase} or {binding.rl_phase}')
    if set(binding.objects).intersection(task._locked_targets):
        raise ValueError('Insertion reset cannot contain locked task bodies')
    states = task.get_tracked_object_states()
    for name in binding.objects:
        if name not in states:
            position, orientation = task._resolve_object(name).get_pose()
            states[name] = dict(position=position, orientation=orientation)
    for robot, obj in binding.participants:
        hold = states[obj].get('attachment') or {}
        if hold.get('mode') != 'pure_physical_grasp' or not hold.get('physical_hold_valid'):
            raise ValueError(f'{obj} has no qualified real-contact grasp')
    cfg = task.config
    data = dict(format=FORMAT, binding=binding.to_dict(), stage_settings=stage_settings or {}, recipe=cfg.spec_path or cfg.recipe,
        recipe_overrides=(cfg.task_metadata or {}).get('recipe_overrides', {}),
        scene_profile=cfg.scene_profile, seed=cfg.seed, layout_seed=cfg.layout_seed,
        phase=task.phase, source_step=int(task.step_counter),
        robots={name: dict(dof_names=list(robot.articulation.dof_names),
            joint_positions=robot.articulation.get_joint_positions(),
            joint_velocities=robot.articulation.get_joint_velocities()) for name, robot in task.robots.items()},
        objects=states, attachments=task._attachments, target_poses=task.target_poses,
        hold_commands=getattr(task, '_ur5e_contact_hold_commands', {}), phase_specs=task.phase_specs, reset_only=True)
    _validate_grasps(data, binding)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, default=json_value, indent=2), encoding='utf-8')
    temp.replace(path)
    return data


def read_snapshot(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('reset_only') is not True:
        raise ValueError('Not a reset-only insertion snapshot')
    if data.get('format') == legacy.SNAPSHOT_FORMAT:
        binding = legacy.validate_legacy_snapshot(data)
        data['binding'] = binding.to_dict()
        data['legacy_artifact'] = True
    elif data.get('format') == FORMAT:
        binding = InsertionTask.from_dict(data['binding'])
        if data.get('phase') not in (binding.entry_phase, binding.rl_phase):
            raise ValueError('Snapshot phase and insertion binding disagree')
    else:
        raise ValueError('Unsupported physical insertion reset format')
    _validate_grasps(data, binding)
    return data
