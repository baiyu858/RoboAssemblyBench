"""Policy handover for any declared insertion binding."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from .control import InsertionConfig, effective_torque_soft_limit, validate_torque_hard_limit
from .recipe import replace_insertion
from .runtime import InsertionController
from .task import InsertionTask


def configure_full_flow(task_config, checkpoint, *, binding, controller_config=None, retreat_control=None,
                        released_rotation_tolerance=None, torque_soft_limit=None, torque_hard_limit=None, axis_alignment=None):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    settings = dict(checkpoint=str(checkpoint), timeout_steps=6000,
                    controller_config=controller_config or asdict(InsertionConfig()),
                    retreat_control=retreat_control or {})
    if torque_soft_limit is not None:
        from .control import validate_torque_soft_limit
        settings['torque_soft_limit'] = validate_torque_soft_limit(
            torque_soft_limit, InsertionConfig(**settings['controller_config']))
    if released_rotation_tolerance is not None:
        settings['released_rotation_tolerance'] = released_rotation_tolerance
    if torque_hard_limit is not None:
        validate_torque_hard_limit(torque_hard_limit, InsertionConfig(**settings['controller_config']),
                                   settings.get('torque_soft_limit'))
        settings['torque_hard_limit'] = torque_hard_limit
    if axis_alignment is not None:
        from .axis_alignment import alignment_settings
        settings['axis_alignment'] = alignment_settings(axis_alignment)
    existing = next((p for p in task_config.phase_specs if p['name'] == binding.rl_phase), None)
    if existing is not None:
        for skill in existing['local_skills'].values():
            if skill.get('controller_config') != settings['controller_config']:
                raise ValueError('Rebuild the recipe with the requested controller configuration')
            skill.update(checkpoint=str(checkpoint), checkpoint_sha256=None, binding=binding.to_dict())
            if torque_soft_limit is not None:
                skill['torque_soft_limit'] = settings['torque_soft_limit']
            if torque_hard_limit is not None:
                skill['torque_hard_limit'] = settings['torque_hard_limit']
            if axis_alignment is not None:
                skill['axis_alignment'] = settings['axis_alignment']
        return task_config
    task_config.phase_specs, task_config.success_criteria = replace_insertion(
        task_config.phase_specs, task_config.success_criteria, binding, settings)
    return task_config


class InsertionPolicyAdapter:
    def __init__(self, spec=None, *, bridge_factory=None):
        self.sessions = {}
        self.models = {}
        self.bridge_factory = bridge_factory

    def act(self, *, task, robot_name, checkpoint_path, skill_spec=None, **kwargs):
        from .ppo import load_policy
        spec = skill_spec or {}
        if not spec.get('binding'):
            raise ValueError('Insertion policy requires an explicit task binding')
        binding = InsertionTask.from_dict(spec['binding'])
        if robot_name not in {r for r, _ in binding.participants}:
            raise ValueError('Robot is not a participant in this insertion')
        key = (checkpoint_path, repr(binding.policy_contract()))
        if key not in self.models:
            expected = spec.get('checkpoint_sha256')
            if expected and hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() != expected:
                raise ValueError('Recipe insertion checkpoint_sha256 mismatch')
            model, config = load_policy(checkpoint_path, binding=binding)
            if spec.get('controller_config') is not None and spec['controller_config'] != asdict(config):
                raise ValueError('Recipe and checkpoint insertion controller configurations differ')
            self.models[key] = (model, config)
        model, config = self.models[key]
        soft_limit = effective_torque_soft_limit(
            spec.get('torque_soft_limit'), getattr(model, 'torque_soft_limit', None), config)
        hard_limit = spec.get('torque_hard_limit')
        if hard_limit is None:
            hard_limit = getattr(model, 'torque_hard_limit', None)
        validate_torque_hard_limit(hard_limit, config, soft_limit)
        alignment = spec.get('axis_alignment')
        if alignment is None:
            alignment = getattr(model, 'axis_alignment', None)
        session_key = (id(task), task.phase_index, task.phase_entry_step, binding.stage_id, key, soft_limit, hard_limit, repr(alignment))
        if session_key not in self.sessions:
            bridge = None if self.bridge_factory is None else self.bridge_factory(task, binding)
            self.sessions = {session_key: dict(controller=InsertionController(
                task, binding=binding, config=config, bridge=bridge,
                torque_soft_limit=soft_limit, torque_hard_limit=hard_limit, axis_alignment=alignment), decision_step=None)}
        session = self.sessions[session_key]
        controller = session['controller']
        sample = controller.sample()
        if sample['failure']:
            raise ValueError(sample['failure'])
        if sample['local_seated']:
            for robot, _ in binding.participants:
                task.mark_local_skill_complete(robot_name=robot, skill_name='contact_insert_rl',
                                               detail=dict(scope='local_held_seating', benchmark_success=False,
                                                           measurement={k: v.tolist() if isinstance(v, np.ndarray) else v
                                                                        for k, v in sample['measurement'].items()}))
            return controller.last_actions[robot_name]
        step = int(task.step_counter)
        if session['decision_step'] is None or step - session['decision_step'] >= config.decimation:
            controller.set_action(model.act(sample['observation']))
            session['decision_step'] = step
        return controller.tick()[robot_name]
