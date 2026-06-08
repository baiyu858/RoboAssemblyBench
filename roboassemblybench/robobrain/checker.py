from __future__ import annotations

import math
from typing import Any

import numpy as np

from roboassemblybench.robobrain.models import CONSTRAINT_CATEGORIES, ROBOT_NAMES, CheckResult, RoboBrainPlan, as_list


AGENT_ALIASES = {
    'agent_1': 'franka_left',
    'agent1': 'franka_left',
    'left': 'franka_left',
    'left_arm': 'franka_left',
    'agent_2': 'franka_right',
    'agent2': 'franka_right',
    'right': 'franka_right',
    'right_arm': 'franka_right',
}


def _as_dict_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get('name')): item for item in items if isinstance(item, dict) and item.get('name')}


def _normalize_agent_name(value: str) -> str:
    value = str(value).strip()
    return AGENT_ALIASES.get(value.lower(), value)


class RoboChecker:
    """Static RoboChecker for generated RoboAssemblyBench recipes.

    The runtime still performs the real physical/contact checks. This checker validates that RoboBrain
    produced a plan that can be loaded by the existing benchmark executor and emits the constraint
    interface calls that correspond to the textual constraints.
    """

    def check(self, *, plan: RoboBrainPlan, recipe: dict[str, Any]) -> CheckResult:
        errors: list[str] = []
        warnings: list[str] = []
        check_code: list[str] = []

        robot_names = {robot.get('name') for robot in recipe.get('robots', []) if robot.get('name')}
        object_by_name = _as_dict_by_name(recipe.get('objects', []))
        target_by_name = _as_dict_by_name(recipe.get('targets', []))

        if not set(ROBOT_NAMES).issubset(robot_names):
            errors.append(f'Recipe must define both RoboAssemblyBench robots: {ROBOT_NAMES}. Found {sorted(robot_names)}.')

        self._check_constraints(plan=plan, robot_names=robot_names, errors=errors, warnings=warnings, check_code=check_code)
        self._check_targets(recipe=recipe, object_by_name=object_by_name, target_by_name=target_by_name, errors=errors)
        self._check_phases(
            recipe=recipe,
            robot_names=robot_names,
            object_by_name=object_by_name,
            target_by_name=target_by_name,
            errors=errors,
            warnings=warnings,
        )
        self._check_success(recipe=recipe, object_by_name=object_by_name, target_by_name=target_by_name, errors=errors)

        return CheckResult(ok=not errors, errors=errors, warnings=warnings, check_code=check_code)

    def _check_constraints(
        self,
        *,
        plan: RoboBrainPlan,
        robot_names: set[str],
        errors: list[str],
        warnings: list[str],
        check_code: list[str],
    ):
        for category in CONSTRAINT_CATEGORIES:
            entries = plan.constraints.get(category, [])
            if not entries:
                warnings.append(f'No {category} constraints were generated.')
                continue
            for index, entry in enumerate(entries):
                agents = entry.get('agents') or entry.get('Agents') or entry.get('Agent') or entry.get('agent')
                normalized_agents = [_normalize_agent_name(agent) for agent in as_list(agents)]
                if not normalized_agents:
                    errors.append(f'{category} constraint #{index} does not mention any agent.')
                    continue
                unknown_agents = sorted(agent for agent in normalized_agents if agent not in robot_names)
                if unknown_agents:
                    errors.append(f'{category} constraint #{index} references unknown agents: {unknown_agents}.')
                validation = entry.get('validation') or self._default_validation_for_category(category)
                check_code.append(f"{validation}(agents={normalized_agents!r}, constraint={entry.get('constraint', '')!r})")

    @staticmethod
    def _default_validation_for_category(category: str) -> str:
        if category == 'Logical':
            return 'Validate_Interaction'
        if category == 'Temporal':
            return 'Validate_Scheduling'
        return 'Validate_Spatial_Occupancy'

    def _check_targets(
        self,
        *,
        recipe: dict[str, Any],
        object_by_name: dict[str, dict[str, Any]],
        target_by_name: dict[str, dict[str, Any]],
        errors: list[str],
    ):
        seen = set()
        for target in recipe.get('targets', []):
            name = target.get('name')
            if not name:
                errors.append('A target entry is missing name.')
                continue
            if name in seen:
                errors.append(f'Duplicate target name: {name}.')
            seen.add(name)

            reference = target.get('reference', 'world')
            if reference != 'world' and reference not in object_by_name:
                errors.append(f"Target {name!r} references unknown object {reference!r}.")
            if reference == 'world' and 'position' not in target:
                errors.append(f"World target {name!r} must define position.")
            if reference != 'world' and 'offset' not in target:
                errors.append(f"Object-relative target {name!r} must define offset.")
            if name not in target_by_name:
                errors.append(f"Target {name!r} could not be indexed.")

    def _check_phases(
        self,
        *,
        recipe: dict[str, Any],
        robot_names: set[str],
        object_by_name: dict[str, dict[str, Any]],
        target_by_name: dict[str, dict[str, Any]],
        errors: list[str],
        warnings: list[str],
    ):
        phases = recipe.get('phases', [])
        if not phases:
            errors.append('RoboBrain generated no phases.')
            return

        phase_names = set()
        for phase in phases:
            phase_name = phase.get('name')
            if not phase_name:
                errors.append('A phase is missing name.')
            elif phase_name in phase_names:
                errors.append(f'Duplicate phase name: {phase_name}.')
            phase_names.add(phase_name)

            robot_targets = phase.get('robot_targets', {})
            for robot_name, target_like in robot_targets.items():
                if robot_name not in robot_names:
                    errors.append(f"Phase {phase_name!r} references unknown robot {robot_name!r}.")
                target_name = self._target_name(target_like)
                if target_name is not None and target_name not in target_by_name:
                    errors.append(f"Phase {phase_name!r} robot {robot_name!r} references unknown target {target_name!r}.")
                if isinstance(target_like, dict) and 'reference' in target_like:
                    reference = target_like.get('reference')
                    if reference != 'world' and reference not in object_by_name:
                        errors.append(
                            f"Phase {phase_name!r} inline target for {robot_name!r} references unknown object {reference!r}."
                        )
                if isinstance(target_like, dict):
                    payload_object = target_like.get('payload_object')
                    payload_target = target_like.get('payload_target')
                    if payload_object and payload_object not in object_by_name:
                        errors.append(
                            f"Phase {phase_name!r} robot {robot_name!r} references unknown payload object {payload_object!r}."
                        )
                    if payload_target and payload_target not in target_by_name:
                        errors.append(
                            f"Phase {phase_name!r} robot {robot_name!r} references unknown payload target {payload_target!r}."
                        )

            for robot_name, command in (phase.get('gripper_commands') or {}).items():
                if robot_name not in robot_names:
                    errors.append(f"Phase {phase_name!r} has gripper command for unknown robot {robot_name!r}.")
                if str(command).lower() not in {'open', 'close', 'none'}:
                    errors.append(f"Phase {phase_name!r} has unsupported gripper command {command!r}.")

            for attach_spec in as_list(phase.get('attach')):
                if not isinstance(attach_spec, dict):
                    continue
                if attach_spec.get('robot') not in robot_names:
                    errors.append(f"Phase {phase_name!r} attach references unknown robot {attach_spec.get('robot')!r}.")
                if attach_spec.get('object') not in object_by_name:
                    errors.append(f"Phase {phase_name!r} attach references unknown object {attach_spec.get('object')!r}.")
                target_name = attach_spec.get('target')
                if target_name and target_name not in target_by_name:
                    errors.append(f"Phase {phase_name!r} attach references unknown target {target_name!r}.")

            for detach_spec in as_list(phase.get('detach')):
                if isinstance(detach_spec, str):
                    object_name = detach_spec
                elif isinstance(detach_spec, dict):
                    object_name = detach_spec.get('object')
                else:
                    continue
                if object_name not in object_by_name:
                    errors.append(f"Phase {phase_name!r} detach references unknown object {object_name!r}.")

            self._warn_close_targets(
                phase=phase,
                object_by_name=object_by_name,
                target_by_name=target_by_name,
                warnings=warnings,
            )
            self._check_advance_condition(
                phase_name=phase_name,
                advance=phase.get('advance', {}),
                robot_names=robot_names,
                object_by_name=object_by_name,
                target_by_name=target_by_name,
                errors=errors,
            )

    def _check_success(
        self,
        *,
        recipe: dict[str, Any],
        object_by_name: dict[str, dict[str, Any]],
        target_by_name: dict[str, dict[str, Any]],
        errors: list[str],
    ):
        for success_spec in recipe.get('success', []):
            if not isinstance(success_spec, dict):
                continue
            object_name = success_spec.get('object')
            target_name = success_spec.get('target')
            if object_name and object_name not in object_by_name:
                errors.append(f"Success criterion references unknown object {object_name!r}.")
            if target_name and target_name not in target_by_name:
                errors.append(f"Success criterion references unknown target {target_name!r}.")

    def _check_advance_condition(
        self,
        *,
        phase_name: str | None,
        advance: dict[str, Any],
        robot_names: set[str],
        object_by_name: dict[str, dict[str, Any]],
        target_by_name: dict[str, dict[str, Any]],
        errors: list[str],
    ):
        if not isinstance(advance, dict) or not advance:
            return
        advance_type = advance.get('type', 'timer')
        if advance_type in {'all_of', 'any_of'}:
            for condition in advance.get('conditions', advance.get('all', advance.get('any', []))):
                if isinstance(condition, dict):
                    self._check_advance_condition(
                        phase_name=phase_name,
                        advance=condition,
                        robot_names=robot_names,
                        object_by_name=object_by_name,
                        target_by_name=target_by_name,
                        errors=errors,
                    )
            return

        if advance_type in {'object_targets_reached', 'object_pose_reached'}:
            for object_target in as_list(advance.get('objects')):
                if not isinstance(object_target, dict):
                    continue
                object_name = object_target.get('object')
                target_name = object_target.get('target')
                if object_name and object_name not in object_by_name:
                    errors.append(f"Phase {phase_name!r} advance references unknown object {object_name!r}.")
                if target_name and target_name not in target_by_name:
                    errors.append(f"Phase {phase_name!r} advance references unknown target {target_name!r}.")

        if advance_type in {'object_lifted', 'objects_lifted', 'objects_static', 'object_static'}:
            object_specs = advance.get('objects')
            if object_specs is None and advance.get('object') is not None:
                object_specs = [advance.get('object')]
            for object_spec in as_list(object_specs):
                object_name = object_spec.get('object') if isinstance(object_spec, dict) else object_spec
                if object_name and object_name not in object_by_name:
                    errors.append(f"Phase {phase_name!r} advance references unknown object {object_name!r}.")
            robot_name = advance.get('robot')
            if robot_name and robot_name not in robot_names:
                errors.append(f"Phase {phase_name!r} advance references unknown robot {robot_name!r}.")

        if advance_type in {'robot_object_contact', 'gripper_object_contact'}:
            contact_specs = advance.get('contacts') or [advance]
            for contact_spec in as_list(contact_specs):
                if not isinstance(contact_spec, dict):
                    continue
                robot_name = contact_spec.get('robot')
                object_name = contact_spec.get('object')
                if robot_name and robot_name not in robot_names:
                    errors.append(f"Phase {phase_name!r} contact advance references unknown robot {robot_name!r}.")
                if object_name and object_name not in object_by_name:
                    errors.append(f"Phase {phase_name!r} contact advance references unknown object {object_name!r}.")

    @staticmethod
    def _target_name(target_like) -> str | None:
        if target_like is None:
            return None
        if isinstance(target_like, str):
            return target_like
        if isinstance(target_like, dict):
            return target_like.get('target')
        return None

    def _target_position(
        self,
        target_like,
        *,
        object_by_name: dict[str, dict[str, Any]],
        target_by_name: dict[str, dict[str, Any]],
    ) -> np.ndarray | None:
        if target_like is None:
            return None
        if isinstance(target_like, str):
            target = target_by_name.get(target_like)
        elif isinstance(target_like, dict) and target_like.get('target'):
            target = target_by_name.get(target_like.get('target'))
        elif isinstance(target_like, dict):
            target = target_like
        else:
            return None
        if not isinstance(target, dict):
            return None
        reference = target.get('reference', 'world')
        if reference == 'world':
            position = target.get('position')
            return None if position is None else np.asarray(position, dtype=float)
        base_object = object_by_name.get(reference)
        if not base_object or base_object.get('position') is None:
            return None
        offset = target.get('offset', [0.0, 0.0, 0.0])
        return np.asarray(base_object.get('position'), dtype=float) + np.asarray(offset, dtype=float)

    def _warn_close_targets(
        self,
        *,
        phase: dict[str, Any],
        object_by_name: dict[str, dict[str, Any]],
        target_by_name: dict[str, dict[str, Any]],
        warnings: list[str],
    ):
        robot_targets = phase.get('robot_targets', {})
        if len(robot_targets) < 2:
            return
        positions = []
        for robot_name, target_like in robot_targets.items():
            position = self._target_position(
                target_like,
                object_by_name=object_by_name,
                target_by_name=target_by_name,
            )
            if position is not None:
                positions.append((robot_name, position))
        for index, (lhs_name, lhs_position) in enumerate(positions):
            for rhs_name, rhs_position in positions[index + 1 :]:
                distance = float(np.linalg.norm(lhs_position - rhs_position))
                if math.isfinite(distance) and distance < 0.08:
                    warnings.append(
                        f"Phase {phase.get('name')!r} places {lhs_name} and {rhs_name} within {distance:.3f} m; "
                        'runtime collision avoidance must handle this.'
                    )
