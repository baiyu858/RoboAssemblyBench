from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from roboassemblybench.robobrain.models import as_list, slugify_task_name

DEFAULT_EE_ORIENTATION = [3.1415926536, 0.0, 0.0]


@dataclass
class SkillCompileResult:
    targets: list[dict[str, Any]] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    success: list[dict[str, Any]] = field(default_factory=list)
    primitive_plan: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SkillLibrary:
    """Composable RoboBrain skill compiler for RoboAssemblyBench phase recipes.

    The LLM can emit a compact skill plan instead of fully spelling out every phase. This compiler
    expands the skill plan into executable target and phase dictionaries that use the existing
    dual-Franka task runtime.
    """

    supported_skills = ('move', 'pick', 'place', 'insert', 'lift', 'press', 'handover')

    @classmethod
    def prompt_payload(cls) -> str:
        payload = {
            'supported_skills': list(cls.supported_skills),
            'skill_schema': {
                'pick': {
                    'skill': 'pick',
                    'robot': 'franka_right',
                    'object': 'object_name',
                    'approach_offset': [0.0, 0.0, 0.12],
                    'grasp_offset': [0.0, 0.0, 0.0],
                    'lift_offset': [0.0, 0.0, 0.14],
                },
                'place': {
                    'skill': 'place',
                    'robot': 'franka_right',
                    'object': 'object_name',
                    'target': 'existing_or_generated_object_goal',
                    'target_position': [0.48, 0.0, 0.06],
                    'approach_height': 0.16,
                    'release_height': 0.08,
                },
                'insert': {
                    'skill': 'insert',
                    'robot': 'franka_right',
                    'object': 'object_name',
                    'target': 'existing_goal_target',
                    'approach_height': 0.18,
                    'insert_height': 0.03,
                    'position_tolerance': 0.035,
                },
                'lift': {
                    'skill': 'lift',
                    'robots': ['franka_left', 'franka_right'],
                    'object': 'object_name',
                    'height': 0.20,
                    'lateral_grasp_offset': 0.22,
                },
                'press': {
                    'skill': 'press',
                    'robot': 'franka_right',
                    'object': 'button_or_fixture',
                    'target': 'press_target',
                    'press_offset': [0.0, 0.0, 0.0],
                },
                'handover': {
                    'skill': 'handover',
                    'from_robot': 'franka_left',
                    'to_robot': 'franka_right',
                    'object': 'object_name',
                    'handoff_position': [0.38, 0.0, 0.36],
                },
            },
            'notes': [
                'Use skills for new task compositions; use raw phases only when the exact phase plan is needed.',
                'All objects must exist in the selected template recipe. New targets may be world or object-relative.',
                'The compiler automatically adds conservative approach, grasp, release, settle, and retreat phases.',
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def __init__(self, base_recipe: dict[str, Any]):
        self.base_recipe = base_recipe
        self.object_by_name = {
            str(item.get('name')): item
            for item in base_recipe.get('objects', [])
            if isinstance(item, dict) and item.get('name')
        }
        self.base_target_by_name = {
            str(item.get('name')): item
            for item in base_recipe.get('targets', [])
            if isinstance(item, dict) and item.get('name')
        }
        self.generated_target_by_name: dict[str, dict[str, Any]] = {}
        self.phase_names: set[str] = set()

    def compile(self, skills: list[dict[str, Any]]) -> SkillCompileResult:
        result = SkillCompileResult()
        carried_object_by_robot: dict[str, str] = {}

        for raw_skill in skills:
            if not isinstance(raw_skill, dict):
                result.warnings.append(f'Ignoring non-dict skill entry: {raw_skill!r}.')
                continue
            skill_name = str(raw_skill.get('skill') or raw_skill.get('primitive') or '').strip().lower()
            if skill_name not in self.supported_skills:
                result.warnings.append(f'Unsupported skill {skill_name!r}; supported skills are {self.supported_skills}.')
                continue

            before_phase_count = len(result.phases)
            if skill_name == 'move':
                self._compile_move(raw_skill, result)
            elif skill_name == 'pick':
                picked = self._compile_pick(raw_skill, result)
                if picked:
                    carried_object_by_robot[picked[0]] = picked[1]
            elif skill_name == 'place':
                placed = self._compile_place_like(raw_skill, result, mode='place')
                if placed:
                    carried_object_by_robot.pop(placed[0], None)
            elif skill_name == 'insert':
                placed = self._compile_place_like(raw_skill, result, mode='insert')
                if placed:
                    carried_object_by_robot.pop(placed[0], None)
            elif skill_name == 'lift':
                lifted = self._compile_lift(raw_skill, result)
                if lifted:
                    carried_object_by_robot[lifted[0]] = lifted[1]
            elif skill_name == 'press':
                self._compile_press(raw_skill, result)
            elif skill_name == 'handover':
                handed = self._compile_handover(raw_skill, result)
                if handed:
                    carried_object_by_robot[handed[0]] = handed[1]

            result.primitive_plan.append(
                {
                    'skill': skill_name,
                    'input': copy.deepcopy(raw_skill),
                    'phase_count': len(result.phases) - before_phase_count,
                }
            )

        result.targets = list(self.generated_target_by_name.values())
        return result

    def _compile_move(self, spec: dict[str, Any], result: SkillCompileResult):
        robot = self._robot(spec.get('robot'))
        target = spec.get('target')
        if not target and spec.get('position') is not None:
            target = self._ensure_world_target(
                name=spec.get('target_name') or f'{self._robot_short(robot)}_move_target',
                position=spec.get('position'),
                orientation_euler=spec.get('orientation_euler'),
            )
        if not target:
            result.warnings.append(f'Move skill for {robot} does not define a target.')
            return
        phase = self._phase(
            name=spec.get('name') or f'{self._robot_short(robot)}_move',
            robot_targets=self._with_waiting_robot(
                robot,
                {robot: {'target': target, 'position_tolerance': float(spec.get('position_tolerance', 0.06))}},
            ),
            gripper_commands=self._with_waiting_grippers(robot, spec.get('gripper', 'open')),
            advance={'type': 'robot_targets_reached', 'min_steps': int(spec.get('min_steps', 12)), 'tolerance': 0.06},
        )
        result.phases.append(phase)

    def _compile_pick(self, spec: dict[str, Any], result: SkillCompileResult) -> tuple[str, str] | None:
        object_name = self._object_name(spec, result)
        if object_name is None:
            return None
        robot = self._robot(spec.get('robot') or self._default_robot_for_object(object_name))
        prefix = self._skill_prefix(robot, object_name)
        approach = self._ensure_object_target(
            name=spec.get('approach_target') or f'{prefix}_approach',
            reference=object_name,
            offset=spec.get('approach_offset', [0.0, 0.0, 0.12]),
            orientation_euler=spec.get('orientation_euler'),
        )
        grasp_open = self._ensure_object_target(
            name=spec.get('open_target') or f'{prefix}_grasp_open',
            reference=object_name,
            offset=spec.get('open_offset', spec.get('grasp_open_offset', [0.0, 0.0, 0.035])),
            orientation_euler=spec.get('orientation_euler'),
        )
        grasp_close = self._ensure_object_target(
            name=spec.get('grasp_target') or f'{prefix}_grasp_close',
            reference=object_name,
            offset=spec.get('grasp_offset', [0.0, 0.0, 0.0]),
            orientation_euler=spec.get('orientation_euler'),
        )
        lift_target = self._ensure_object_target(
            name=spec.get('lift_target') or f'{prefix}_lift',
            reference=object_name,
            offset=spec.get('lift_offset', [0.0, 0.0, 0.14]),
            orientation_euler=spec.get('orientation_euler'),
        )
        attach = self._attach_spec(object_name=object_name, robot=robot, target=grasp_close, spec=spec)

        result.phases.extend(
            [
                self._phase(
                    name=f'{prefix}_approach',
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': approach, 'position_tolerance': float(spec.get('approach_tolerance', 0.07))}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'open'),
                    advance={'type': 'robot_targets_reached', 'min_steps': 12, 'tolerance': 0.06},
                ),
                self._phase(
                    name=f'{prefix}_descend',
                    timeout_steps=int(spec.get('descend_timeout_steps', 260)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': grasp_open, 'position_tolerance': float(spec.get('open_tolerance', 0.04))}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'open'),
                    advance={'type': 'robot_targets_reached', 'min_steps': 10, 'tolerance': 0.04},
                ),
                self._phase(
                    name=f'{prefix}_grasp',
                    timeout_steps=int(spec.get('grasp_timeout_steps', 340)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': grasp_close, 'position_tolerance': float(spec.get('grasp_tolerance', 0.025))}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'close'),
                    attach=[attach],
                    advance={'type': 'timer', 'min_steps': int(spec.get('grasp_settle_steps', 24))},
                ),
                self._phase(
                    name=f'{prefix}_lift_verify',
                    timeout_steps=int(spec.get('lift_timeout_steps', 260)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': lift_target, 'position_tolerance': float(spec.get('lift_tolerance', 0.08))}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'close'),
                    advance={
                        'type': 'all_of',
                        'min_steps': 12,
                        'conditions': [
                            {'type': 'robot_targets_reached', 'tolerance': 0.08},
                            {
                                'type': 'object_lifted',
                                'object': object_name,
                                'robot': robot,
                                'min_lift': float(spec.get('min_lift', 0.04)),
                            },
                        ],
                    },
                ),
            ]
        )
        return robot, object_name

    def _compile_place_like(self, spec: dict[str, Any], result: SkillCompileResult, *, mode: str) -> tuple[str, str] | None:
        object_name = self._object_name(spec, result)
        if object_name is None:
            return None
        robot = self._robot(spec.get('robot') or self._default_robot_for_object(object_name))
        prefix = self._skill_prefix(robot, object_name, suffix=mode)
        target_name = self._ensure_goal_target(spec=spec, object_name=object_name, mode=mode)
        approach_height = float(spec.get('approach_height', 0.18 if mode == 'insert' else 0.16))
        release_height = float(spec.get('insert_height' if mode == 'insert' else 'release_height', 0.03 if mode == 'insert' else 0.08))
        pre_target = self._ensure_target_offset_from_target(
            source_target=target_name,
            name=spec.get('approach_target') or f'{prefix}_approach',
            z_delta=approach_height,
        )
        final_target = self._ensure_target_offset_from_target(
            source_target=target_name,
            name=spec.get('final_robot_target') or f'{prefix}_final',
            z_delta=release_height,
        )
        tolerance = float(spec.get('position_tolerance', 0.035 if mode == 'insert' else 0.05))
        result.phases.extend(
            [
                self._phase(
                    name=f'{prefix}_approach',
                    timeout_steps=int(spec.get('approach_timeout_steps', 360)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {
                            robot: {
                                'target': pre_target,
                                'payload_object': object_name,
                                'payload_target': target_name,
                                'position_tolerance': max(tolerance, 0.06),
                            }
                        },
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'close'),
                    advance={'type': 'robot_targets_reached', 'min_steps': 12, 'tolerance': max(tolerance, 0.06)},
                ),
                self._phase(
                    name=f'{prefix}_final',
                    timeout_steps=int(spec.get('final_timeout_steps', 420)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {
                            robot: {
                                'target': final_target,
                                'payload_object': object_name,
                                'payload_target': target_name,
                                'position_tolerance': tolerance,
                            }
                        },
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'close'),
                    advance={
                        'type': 'all_of',
                        'min_steps': 18,
                        'conditions': [
                            {'type': 'robot_targets_reached', 'tolerance': tolerance},
                            {
                                'type': 'object_targets_reached',
                                'tolerance': tolerance,
                                'objects': [
                                    {
                                        'object': object_name,
                                        'target': target_name,
                                        'position_tolerance': tolerance,
                                        'orientation_tolerance': float(spec.get('orientation_tolerance', 0.6)),
                                    }
                                ],
                            },
                        ],
                    },
                ),
                self._phase(
                    name=f'{prefix}_release',
                    timeout_steps=int(spec.get('release_timeout_steps', 180)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {
                            robot: {
                                'target': final_target,
                                'payload_object': object_name,
                                'payload_target': target_name,
                                'position_tolerance': max(tolerance, 0.06),
                            }
                        },
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'open'),
                    detach=[object_name],
                    advance={'type': 'timer', 'min_steps': int(spec.get('release_steps', 24))},
                ),
                self._phase(
                    name=f'{prefix}_settle',
                    timeout_steps=int(spec.get('settle_timeout_steps', 260)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': pre_target, 'position_tolerance': 0.08}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'open'),
                    advance={
                        'type': 'all_of',
                        'min_steps': 12,
                        'conditions': [
                            {
                                'type': 'object_targets_reached',
                                'tolerance': tolerance,
                                'objects': [
                                    {
                                        'object': object_name,
                                        'target': target_name,
                                        'position_tolerance': tolerance,
                                        'orientation_tolerance': float(spec.get('orientation_tolerance', 0.6)),
                                    }
                                ],
                            },
                            {
                                'type': 'objects_static',
                                'objects': [object_name],
                                'linear_velocity_threshold': 0.025,
                                'angular_velocity_threshold': 0.8,
                            },
                        ],
                    },
                ),
            ]
        )
        result.success.append(
            {
                'object': object_name,
                'target': target_name,
                'position_tolerance': tolerance,
                'orientation_tolerance': float(spec.get('orientation_tolerance', 0.6)),
                'require_released': True,
                'require_static': True,
                'linear_velocity_threshold': 0.025,
                'angular_velocity_threshold': 0.8,
            }
        )
        return robot, object_name

    def _compile_lift(self, spec: dict[str, Any], result: SkillCompileResult) -> tuple[str, str] | None:
        object_name = self._object_name(spec, result)
        if object_name is None:
            return None
        robots = [self._robot(robot) for robot in as_list(spec.get('robots') or ['franka_left', 'franka_right'])]
        robots = [robot for robot in robots if robot in {'franka_left', 'franka_right'}]
        if not robots:
            robots = [self._default_robot_for_object(object_name)]
        owner_robot = robots[0]
        height = float(spec.get('height', 0.20))
        lateral = float(spec.get('lateral_grasp_offset', 0.22))
        object_scale = self.object_by_name.get(object_name, {}).get('scale') or [0.08, 0.08, 0.08]
        top_z = max(float(object_scale[2]) * 0.5, 0.04)
        target_map = {}
        lift_target_map = {}
        for robot in robots:
            side = -1.0 if robot == 'franka_left' else 1.0
            prefix = self._skill_prefix(robot, object_name, suffix='lift')
            approach = self._ensure_object_target(
                name=f'{prefix}_approach',
                reference=object_name,
                offset=[0.0, side * lateral, top_z + 0.10],
                orientation_euler=spec.get('orientation_euler'),
            )
            grasp = self._ensure_object_target(
                name=f'{prefix}_grasp',
                reference=object_name,
                offset=[0.0, side * lateral, max(top_z - 0.05, 0.0)],
                orientation_euler=spec.get('orientation_euler'),
            )
            lift = self._ensure_object_target(
                name=f'{prefix}_target',
                reference=object_name,
                offset=[0.0, side * lateral, top_z + height],
                orientation_euler=spec.get('orientation_euler'),
            )
            target_map[robot] = {'approach': approach, 'grasp': grasp}
            lift_target_map[robot] = lift
        lifted_target = self._ensure_object_target(
            name=spec.get('lifted_target') or f'{slugify_task_name(object_name)}_lifted_generated',
            reference=object_name,
            offset=[0.0, 0.0, height],
            orientation_euler=[1.0e-06, 0.0, 0.0],
        )
        result.phases.extend(
            [
                self._phase(
                    name=f'{slugify_task_name(object_name)}_dual_lift_approach',
                    robot_targets={
                        robot: {'target': target_map[robot]['approach'], 'position_tolerance': 0.08}
                        for robot in robots
                    },
                    gripper_commands={robot: 'open' for robot in robots},
                    advance={'type': 'robot_targets_reached', 'min_steps': 12, 'tolerance': 0.07},
                ),
                self._phase(
                    name=f'{slugify_task_name(object_name)}_dual_lift_grasp',
                    timeout_steps=int(spec.get('grasp_timeout_steps', 360)),
                    robot_targets={
                        robot: {'target': target_map[robot]['grasp'], 'position_tolerance': 0.04}
                        for robot in robots
                    },
                    gripper_commands={robot: 'close' for robot in robots},
                    attach=[
                        self._attach_spec(
                            object_name=object_name,
                            robot=owner_robot,
                            target=target_map[owner_robot]['grasp'],
                            spec=spec,
                        )
                    ],
                    advance={'type': 'timer', 'min_steps': int(spec.get('grasp_settle_steps', 30))},
                ),
                self._phase(
                    name=f'{slugify_task_name(object_name)}_dual_lift_raise',
                    timeout_steps=int(spec.get('lift_timeout_steps', 520)),
                    robot_targets={
                        robot: {
                            'target': lift_target_map[robot],
                            **(
                                {'payload_object': object_name, 'payload_target': lifted_target}
                                if robot == owner_robot
                                else {}
                            ),
                            'position_tolerance': 0.09,
                        }
                        for robot in robots
                    },
                    gripper_commands={robot: 'close' for robot in robots},
                    advance={
                        'type': 'object_targets_reached',
                        'min_steps': 24,
                        'objects': [
                            {
                                'object': object_name,
                                'target': lifted_target,
                                'position_tolerance': float(spec.get('position_tolerance', 0.10)),
                                'orientation_tolerance': float(spec.get('orientation_tolerance', 1.2)),
                            }
                        ],
                    },
                ),
            ]
        )
        result.success.append(
            {
                'object': object_name,
                'target': lifted_target,
                'position_tolerance': float(spec.get('position_tolerance', 0.10)),
                'orientation_tolerance': float(spec.get('orientation_tolerance', 1.2)),
                'require_attached': True,
                'robot': owner_robot,
                'require_static': True,
                'linear_velocity_threshold': 0.08,
                'angular_velocity_threshold': 2.0,
            }
        )
        return owner_robot, object_name

    def _compile_press(self, spec: dict[str, Any], result: SkillCompileResult):
        object_name = self._object_name(spec, result)
        if object_name is None:
            return
        robot = self._robot(spec.get('robot') or self._default_robot_for_object(object_name))
        prefix = self._skill_prefix(robot, object_name, suffix='press')
        target = spec.get('target')
        if target and self._target_exists(str(target)):
            press_target = str(target)
        else:
            press_target = self._ensure_object_target(
                name=target or f'{prefix}_target',
                reference=object_name,
                offset=spec.get('press_offset', [0.0, 0.0, 0.02]),
                orientation_euler=spec.get('orientation_euler'),
            )
        approach = self._ensure_target_offset_from_target(
            source_target=press_target,
            name=f'{prefix}_approach',
            z_delta=float(spec.get('approach_height', 0.12)),
        )
        result.phases.extend(
            [
                self._phase(
                    name=f'{prefix}_approach',
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': approach, 'position_tolerance': 0.06}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, 'open'),
                    advance={'type': 'robot_targets_reached', 'min_steps': 12, 'tolerance': 0.06},
                ),
                self._phase(
                    name=f'{prefix}_contact',
                    timeout_steps=int(spec.get('press_timeout_steps', 260)),
                    robot_targets=self._with_waiting_robot(
                        robot,
                        {robot: {'target': press_target, 'position_tolerance': float(spec.get('position_tolerance', 0.035))}},
                    ),
                    gripper_commands=self._with_waiting_grippers(robot, spec.get('gripper', 'open')),
                    advance={
                        'type': 'robot_object_contact',
                        'robot': robot,
                        'object': object_name,
                        'min_steps': int(spec.get('min_steps', 18)),
                    },
                ),
            ]
        )

    def _compile_handover(self, spec: dict[str, Any], result: SkillCompileResult) -> tuple[str, str] | None:
        object_name = self._object_name(spec, result)
        if object_name is None:
            return None
        from_robot = self._robot(spec.get('from_robot') or 'franka_left')
        to_robot = self._robot(spec.get('to_robot') or self._other_robot(from_robot))
        handoff_target = self._ensure_world_target(
            name=spec.get('handoff_target') or f'{slugify_task_name(object_name)}_handoff_generated',
            position=spec.get('handoff_position', [0.38, 0.0, 0.36]),
            orientation_euler=spec.get('orientation_euler'),
        )
        receive_target = self._ensure_target_offset_from_target(
            source_target=handoff_target,
            name=f'{slugify_task_name(object_name)}_{self._robot_short(to_robot)}_receive_generated',
            y_delta=0.04 if to_robot == 'franka_right' else -0.04,
        )
        result.phases.extend(
            [
                self._phase(
                    name=f'{slugify_task_name(object_name)}_handover_present',
                    robot_targets={
                        from_robot: {
                            'target': handoff_target,
                            'payload_object': object_name,
                            'payload_target': handoff_target,
                            'position_tolerance': 0.07,
                        },
                        to_robot: {'target': receive_target, 'position_tolerance': 0.08},
                    },
                    gripper_commands={from_robot: 'close', to_robot: 'open'},
                    advance={'type': 'robot_targets_reached', 'min_steps': 16, 'tolerance': 0.07},
                ),
                self._phase(
                    name=f'{slugify_task_name(object_name)}_handover_transfer',
                    timeout_steps=int(spec.get('transfer_timeout_steps', 340)),
                    robot_targets={
                        from_robot: {
                            'target': handoff_target,
                            'payload_object': object_name,
                            'payload_target': handoff_target,
                            'position_tolerance': 0.07,
                        },
                        to_robot: {'target': receive_target, 'position_tolerance': 0.05},
                    },
                    gripper_commands={from_robot: 'open', to_robot: 'close'},
                    attach=[self._attach_spec(object_name=object_name, robot=to_robot, target=receive_target, spec=spec)],
                    detach=[object_name],
                    advance={'type': 'timer', 'min_steps': int(spec.get('transfer_settle_steps', 30))},
                ),
            ]
        )
        return to_robot, object_name

    def _object_name(self, spec: dict[str, Any], result: SkillCompileResult) -> str | None:
        object_name = spec.get('object') or spec.get('target_object')
        if not object_name:
            result.warnings.append(f"Skill {spec.get('skill')!r} does not define object.")
            return None
        object_name = str(object_name)
        if object_name not in self.object_by_name:
            result.warnings.append(f"Skill {spec.get('skill')!r} references unknown object {object_name!r}.")
            return None
        return object_name

    @staticmethod
    def _robot(value: Any) -> str:
        value = str(value or '').strip().lower()
        if value in {'left', 'left_arm', 'agent_1', 'franka_left'}:
            return 'franka_left'
        if value in {'right', 'right_arm', 'agent_2', 'franka_right'}:
            return 'franka_right'
        return 'franka_right'

    @staticmethod
    def _robot_short(robot: str) -> str:
        return 'left' if robot == 'franka_left' else 'right'

    @staticmethod
    def _other_robot(robot: str) -> str:
        return 'franka_right' if robot == 'franka_left' else 'franka_left'

    def _default_robot_for_object(self, object_name: str) -> str:
        position = self.object_by_name.get(object_name, {}).get('position') or [0.0, 0.0, 0.0]
        try:
            return 'franka_right' if float(position[1]) >= 0.0 else 'franka_left'
        except (TypeError, ValueError, IndexError):
            return 'franka_right'

    def _skill_prefix(self, robot: str, object_name: str, suffix: str | None = None) -> str:
        parts = [self._robot_short(robot), slugify_task_name(object_name)]
        if suffix:
            parts.append(slugify_task_name(suffix))
        return '_'.join(parts)

    def _target_exists(self, name: str) -> bool:
        return name in self.base_target_by_name or name in self.generated_target_by_name

    def _get_target(self, name: str) -> dict[str, Any] | None:
        return self.generated_target_by_name.get(name) or self.base_target_by_name.get(name)

    def _unique_name(self, raw_name: Any, *, existing: set[str] | None = None) -> str:
        base = slugify_task_name(str(raw_name), fallback='generated_target')
        used = set(self.base_target_by_name) | set(self.generated_target_by_name) | set(existing or set())
        if base not in used:
            return base
        index = 2
        while f'{base}_{index}' in used:
            index += 1
        return f'{base}_{index}'

    def _ensure_object_target(
        self,
        *,
        name: Any,
        reference: str,
        offset: Any,
        orientation_euler: Any = None,
    ) -> str:
        name = str(name)
        if self._target_exists(name):
            return name
        target_name = self._unique_name(name)
        self.generated_target_by_name[target_name] = {
            'name': target_name,
            'reference': reference,
            'offset': [float(item) for item in as_list(offset)[:3]],
            'orientation_euler': [float(item) for item in as_list(orientation_euler or DEFAULT_EE_ORIENTATION)[:3]],
        }
        return target_name

    def _ensure_world_target(self, *, name: Any, position: Any, orientation_euler: Any = None) -> str:
        name = str(name)
        if self._target_exists(name):
            return name
        target_name = self._unique_name(name)
        self.generated_target_by_name[target_name] = {
            'name': target_name,
            'reference': 'world',
            'position': [float(item) for item in as_list(position)[:3]],
            'orientation_euler': [float(item) for item in as_list(orientation_euler or DEFAULT_EE_ORIENTATION)[:3]],
        }
        return target_name

    def _ensure_goal_target(self, *, spec: dict[str, Any], object_name: str, mode: str) -> str:
        raw_target = spec.get('target') or spec.get('target_name') or f'{slugify_task_name(object_name)}_{mode}_goal'
        target_name = str(raw_target)
        if self._target_exists(target_name):
            return target_name
        if spec.get('target_position') is not None or spec.get('position') is not None:
            return self._ensure_world_target(
                name=target_name,
                position=spec.get('target_position', spec.get('position')),
                orientation_euler=spec.get('target_orientation_euler') or spec.get('orientation_euler') or [1.0e-06, 0.0, 0.0],
            )
        reference = spec.get('reference') or object_name
        offset = spec.get('target_offset') or spec.get('offset') or [0.0, 0.0, 0.0]
        return self._ensure_object_target(
            name=target_name,
            reference=str(reference),
            offset=offset,
            orientation_euler=spec.get('target_orientation_euler') or spec.get('orientation_euler') or [1.0e-06, 0.0, 0.0],
        )

    def _ensure_target_offset_from_target(
        self,
        *,
        source_target: str,
        name: Any,
        x_delta: float = 0.0,
        y_delta: float = 0.0,
        z_delta: float = 0.0,
    ) -> str:
        if self._target_exists(str(name)):
            return str(name)
        source = self._get_target(source_target)
        if not source:
            return self._ensure_world_target(name=name, position=[0.42 + x_delta, y_delta, 0.32 + z_delta])
        target_name = self._unique_name(name)
        target = copy.deepcopy(source)
        target['name'] = target_name
        if target.get('reference', 'world') == 'world' and target.get('position') is not None:
            position = list(target.get('position', [0.0, 0.0, 0.0]))
            target['position'] = [
                float(position[0]) + float(x_delta),
                float(position[1]) + float(y_delta),
                float(position[2]) + float(z_delta),
            ]
        else:
            offset = list(target.get('offset', [0.0, 0.0, 0.0]))
            target['offset'] = [
                float(offset[0]) + float(x_delta),
                float(offset[1]) + float(y_delta),
                float(offset[2]) + float(z_delta),
            ]
        target.setdefault('orientation_euler', DEFAULT_EE_ORIENTATION)
        self.generated_target_by_name[target_name] = target
        return target_name

    def _attach_spec(self, *, object_name: str, robot: str, target: str, spec: dict[str, Any]) -> dict[str, Any]:
        template = self._find_attach_template(object_name=object_name, robot=robot)
        if template is None:
            template = {
                'object': object_name,
                'robot': robot,
                'target': target,
                'position_tolerance': 0.025,
                'attachment_mode': 'physical_joint',
                'disable_collision_on_attach': False,
                'require_physical_contact': True,
                'require_target_reached_for_attach': True,
                'finger_contact_distance': 0.008,
                'physical_attach_surface_gap': 0.008,
                'gripper_closed_threshold': 0.02,
                'gripper_closed_margin': 0.004,
            }
        attach = copy.deepcopy(template)
        attach['object'] = object_name
        attach['robot'] = robot
        attach['target'] = target
        attach['position_tolerance'] = float(spec.get('attach_tolerance', attach.get('position_tolerance', 0.025)))
        attach.setdefault('require_target_reached_for_attach', True)
        attach.setdefault('require_physical_contact', True)
        return attach

    def _find_attach_template(self, *, object_name: str, robot: str) -> dict[str, Any] | None:
        fallback = None
        for phase in self.base_recipe.get('phases', []):
            for attach in as_list(phase.get('attach')):
                if not isinstance(attach, dict):
                    continue
                if attach.get('object') == object_name and attach.get('robot') == robot:
                    return attach
                if attach.get('object') == object_name:
                    fallback = attach
                elif fallback is None:
                    fallback = attach
        return fallback

    def _phase(self, *, name: str, **kwargs) -> dict[str, Any]:
        phase_name = slugify_task_name(name, fallback='skill_phase')
        if phase_name in self.phase_names:
            base = phase_name
            index = 2
            while f'{base}_{index}' in self.phase_names:
                index += 1
            phase_name = f'{base}_{index}'
        self.phase_names.add(phase_name)
        phase = {'name': phase_name}
        phase.update({key: copy.deepcopy(value) for key, value in kwargs.items() if value is not None})
        phase.setdefault('timeout_action', 'fail')
        return phase

    def _with_waiting_robot(self, active_robot: str, targets: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(targets)
        other = self._other_robot(active_robot)
        wait_target = f'{self._robot_short(other)}_wait'
        if wait_target in self.base_target_by_name and other not in merged:
            merged[other] = {'target': wait_target, 'position_tolerance': 0.08}
        return merged

    def _with_waiting_grippers(self, active_robot: str, active_command: str) -> dict[str, str]:
        other = self._other_robot(active_robot)
        commands = {active_robot: str(active_command).lower()}
        if f'{self._robot_short(other)}_wait' in self.base_target_by_name:
            commands[other] = 'open'
        return commands


def compile_skill_plan(skills: list[dict[str, Any]], *, base_recipe: dict[str, Any]) -> SkillCompileResult:
    return SkillLibrary(base_recipe=base_recipe).compile(skills)
