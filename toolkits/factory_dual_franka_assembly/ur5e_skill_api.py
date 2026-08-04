from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable


UR5E_ASSEMBLY_ADAPTER = (
    'toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills:'
    'UR5eAssemblyAtomicSkillAdapter'
)


@dataclass(frozen=True)
class AtomicSkillDefinition:
    name: str
    runtime_name: str
    description: str
    required_parameters: tuple[str, ...] = ()


@dataclass
class AtomicSkillCall:
    skill: str
    robot: str
    parameters: dict[str, Any] = field(default_factory=dict)
    phase_name: str | None = None
    timeout_steps: int | None = None
    phase_actions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'AtomicSkillCall':
        payload = copy.deepcopy(payload)
        skill = payload.pop('skill', payload.pop('name', None))
        robot = payload.pop('robot', None)
        if not skill or not robot:
            raise ValueError('An atomic skill call requires both skill and robot.')
        phase_name = payload.pop('phase_name', None)
        timeout_steps = payload.pop('timeout_steps', None)
        phase_actions = payload.pop('phase_actions', {})
        parameters = payload.pop('parameters', {})
        parameters = {**payload, **dict(parameters)}
        return cls(
            skill=str(skill),
            robot=str(robot),
            parameters=parameters,
            phase_name=None if phase_name is None else str(phase_name),
            timeout_steps=None if timeout_steps is None else int(timeout_steps),
            phase_actions=dict(phase_actions),
        )


class UR5eAssemblySkillAPI:
    """Planner-facing compiler for the proven UR5e assembly atoms.

    The API emits the same ``local_skill`` phase contract consumed by
    ``LocalSkillExecutor``. A planner therefore deals in typed skill calls and
    never needs to know adapter import paths or completion-condition syntax.
    """

    definitions = (
        AtomicSkillDefinition(
            'move_above_part',
            'ur5e_move_above_part',
            'Move to an object-relative pre-grasp pose with the gripper open.',
            ('object',),
        ),
        AtomicSkillDefinition(
            'descend_to_grasp',
            'ur5e_descend_to_grasp',
            'Servo from the pre-grasp pose to the contact-consistent grasp pose.',
            ('object',),
        ),
        AtomicSkillDefinition(
            'preshape_gripper',
            'ur5e_preshape_gripper',
            'Hold the arm and settle the gripper at a pre-grasp opening.',
            ('object', 'gripper_openness'),
        ),
        AtomicSkillDefinition(
            'retreat_vertical',
            'ur5e_retreat_vertical',
            'Retreat the tool vertically from its current pose before cross-workspace transit.',
        ),
        AtomicSkillDefinition(
            'close_gripper',
            'ur5e_close_gripper',
            'Close the gripper while holding the validated grasp pose.',
            ('object',),
        ),
        AtomicSkillDefinition(
            'move_part_to_hover',
            'ur5e_move_part_to_table_hover',
            'Transport an attached part to an intermediate object-pose target.',
            ('object', 'target_object_target'),
        ),
        AtomicSkillDefinition(
            'move_part_to_target',
            'ur5e_move_part_to_staging',
            'Transport an attached part to its assembly or staging target.',
            ('object', 'target_object_target'),
        ),
        AtomicSkillDefinition(
            'hold_part_end',
            'ur5e_hold_part_end',
            'Hold the end of an attached part while the partner arm operates.',
            ('object', 'target_object_target'),
        ),
    )
    _by_name = {
        alias: definition
        for definition in definitions
        for alias in (definition.name, definition.runtime_name)
    }
    _motion_defaults = {
        'cartesian_servo': True,
        'cartesian_position_step': 0.015,
        'cartesian_orientation_step': 0.030,
        'guard_ik_branch_jump': True,
        'ik_branch_jump_limit': 0.30,
        'default_max_command_joint_step': 0.060,
        'default_max_command_wrist_joint_step': 0.040,
        'limit_command_to_measured_state': True,
        'max_command_tracking_error': 0.18,
        'max_wrist_command_tracking_error': 0.12,
    }
    _transport_motion_defaults = {
        **_motion_defaults,
        'cartesian_orientation_step': 0.030,
        'default_max_command_joint_step': 0.060,
        'default_max_command_wrist_joint_step': 0.040,
        'servo_target_object_pose': True,
    }
    _hold_motion_defaults = {
        **_motion_defaults,
        'cartesian_position_step': 0.001,
        'cartesian_orientation_step': 0.015,
        'default_max_command_joint_step': 0.015,
        'default_max_command_wrist_joint_step': 0.010,
    }
    _descend_motion_defaults = {
        **_motion_defaults,
        'cartesian_position_step': 0.004,
        'cartesian_orientation_step': 0.015,
        'default_max_command_joint_step': 0.035,
        'default_max_command_wrist_joint_step': 0.015,
    }
    _retreat_motion_defaults = {
        **_motion_defaults,
        'relative_to_current_tcp': True,
        'offset': [0.0, 0.0, 0.15],
        'offset_frame': 'world',
        'lock_target_position': True,
        'lock_target_orientation': True,
        'cartesian_position_step': 0.008,
        'default_max_command_joint_step': 0.035,
        'default_max_command_wrist_joint_step': 0.020,
    }

    @classmethod
    def describe(cls) -> list[dict[str, Any]]:
        return [
            {
                'name': definition.name,
                'runtime_name': definition.runtime_name,
                'description': definition.description,
                'required_parameters': list(definition.required_parameters),
                'adapter': UR5E_ASSEMBLY_ADAPTER,
            }
            for definition in cls.definitions
        ]

    @classmethod
    def compile_call(cls, call: AtomicSkillCall | dict[str, Any], *, index: int = 0) -> dict[str, Any]:
        if isinstance(call, dict):
            call = AtomicSkillCall.from_dict(call)
        definition = cls._by_name.get(call.skill)
        if definition is None:
            supported = ', '.join(definition.name for definition in cls.definitions)
            raise ValueError(f'Unknown UR5e assembly skill {call.skill!r}. Supported skills: {supported}.')

        missing = [name for name in definition.required_parameters if call.parameters.get(name) is None]
        if missing:
            raise ValueError(f'UR5e assembly skill {definition.name!r} is missing parameters: {missing}.')

        if definition.runtime_name in {'ur5e_close_gripper', 'ur5e_preshape_gripper'}:
            defaults = {}
        elif definition.runtime_name in {'ur5e_move_part_to_table_hover', 'ur5e_move_part_to_staging'}:
            defaults = cls._transport_motion_defaults
        elif definition.runtime_name == 'ur5e_hold_part_end':
            defaults = cls._hold_motion_defaults
        elif definition.runtime_name == 'ur5e_descend_to_grasp':
            defaults = cls._descend_motion_defaults
        elif definition.runtime_name == 'ur5e_retreat_vertical':
            defaults = cls._retreat_motion_defaults
        else:
            defaults = cls._motion_defaults
        local_skill = {
            'name': definition.runtime_name,
            'robot': call.robot,
            **copy.deepcopy(defaults),
            **copy.deepcopy(call.parameters),
        }
        phase = {
            'name': call.phase_name or f'{index:02d}_{call.robot}_{definition.name}',
            'robot_targets': {},
            'local_skill': local_skill,
            'advance': {
                'type': 'local_skill_complete',
                'robot': call.robot,
                'skill': definition.runtime_name,
                'min_steps': int(call.parameters.get('min_steps', 1)),
            },
        }
        if call.timeout_steps is not None:
            phase['timeout_steps'] = int(call.timeout_steps)
        phase.update(copy.deepcopy(call.phase_actions))
        return phase

    @classmethod
    def compile_plan(cls, calls: Iterable[AtomicSkillCall | dict[str, Any]]) -> list[dict[str, Any]]:
        return [cls.compile_call(call, index=index) for index, call in enumerate(calls)]


def compile_ur5e_skill_plan(calls: Iterable[AtomicSkillCall | dict[str, Any]]) -> list[dict[str, Any]]:
    return UR5eAssemblySkillAPI.compile_plan(calls)
