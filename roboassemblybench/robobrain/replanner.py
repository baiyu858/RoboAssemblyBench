from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from roboassemblybench.robobrain.models import RoboBrainPlan, as_list, slugify_task_name


@dataclass
class LocalReplanResult:
    plan: RoboBrainPlan
    recipe: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


class LocalReplanner:
    """Deterministic online repair pass for runtime RoboChecker failures."""

    def replan(
        self,
        *,
        plan: RoboBrainPlan,
        recipe: dict[str, Any],
        runtime_feedback: dict[str, Any] | None,
        attempt: int = 1,
    ) -> LocalReplanResult | None:
        feedback_items = [
            item for item in (runtime_feedback or {}).get('feedback', []) if isinstance(item, dict)
        ]
        if not feedback_items:
            return None

        patched_recipe = copy.deepcopy(recipe)
        patched_plan = copy.deepcopy(plan)
        patches = []
        for item in feedback_items[-16:]:
            validation = item.get('validation')
            if validation == 'Validate_Scheduling':
                patches.extend(self._patch_scheduling(patched_recipe, item))
            elif validation == 'Validate_Spatial_Occupancy':
                patches.extend(self._patch_spatial(patched_recipe, item, attempt=attempt))
            elif validation == 'Validate_Interaction':
                patches.extend(self._patch_interaction(patched_recipe, item))

        if not patches:
            return None

        patched_plan.task_name = slugify_task_name(f'{plan.task_name}_local_replan_{attempt}')
        patched_plan.phases = copy.deepcopy(patched_recipe.get('phases', []))
        patched_plan.targets = copy.deepcopy(patched_recipe.get('targets', []))
        patched_plan.success = copy.deepcopy(patched_recipe.get('success', []))
        patched_plan.source = f'{plan.source}+local_replan'
        patched_plan.assumptions = [
            *list(plan.assumptions),
            'Local deterministic replanner patched the previous runtime failure before asking the LLM for a full replan.',
        ]
        patched_recipe['task_name'] = patched_plan.task_name
        metadata = copy.deepcopy(patched_recipe.get('metadata', {}))
        robobrain_metadata = copy.deepcopy(metadata.get('robobrain', {}))
        robobrain_metadata['local_replan'] = {
            'attempt': int(attempt),
            'patches': patches,
            'source_feedback_count': len(feedback_items),
        }
        metadata['robobrain'] = robobrain_metadata
        patched_recipe['metadata'] = metadata
        return LocalReplanResult(
            plan=patched_plan,
            recipe=patched_recipe,
            report=robobrain_metadata['local_replan'],
        )

    def _patch_scheduling(self, recipe: dict[str, Any], feedback: dict[str, Any]) -> list[dict[str, Any]]:
        phase = self._phase_by_name(recipe, feedback.get('phase'))
        if phase is None:
            return []
        old_timeout = int(phase.get('timeout_steps') or recipe.get('phase_timeout_steps') or 240)
        new_timeout = max(old_timeout + 120, int(old_timeout * 1.5))
        phase['timeout_steps'] = new_timeout
        phase['timeout_action'] = 'fail'
        advance = phase.get('advance')
        if isinstance(advance, dict) and advance.get('type') == 'robot_targets_reached':
            old_tolerance = float(advance.get('tolerance', 0.05))
            advance['tolerance'] = min(max(old_tolerance * 1.25, old_tolerance + 0.01), 0.12)
        return [
            {
                'type': 'scheduling_timeout_relaxation',
                'phase': phase.get('name'),
                'old_timeout_steps': old_timeout,
                'new_timeout_steps': new_timeout,
            }
        ]

    def _patch_spatial(self, recipe: dict[str, Any], feedback: dict[str, Any], *, attempt: int) -> list[dict[str, Any]]:
        phase = self._phase_by_name(recipe, feedback.get('phase'))
        if phase is None:
            return []
        robot_targets = phase.get('robot_targets') or {}
        if len(robot_targets) < 2:
            return []

        patches = []
        target_by_name = self._target_by_name(recipe)
        for robot_name, target_like in list(robot_targets.items()):
            direction = -1.0 if 'left' in str(robot_name) else 1.0
            y_delta = direction * (0.055 + 0.025 * max(int(attempt) - 1, 0))
            new_target_name = self._shift_target_for_robot(
                recipe=recipe,
                target_by_name=target_by_name,
                phase=phase,
                robot_name=robot_name,
                target_like=target_like,
                y_delta=y_delta,
                attempt=attempt,
            )
            if new_target_name:
                patches.append(
                    {
                        'type': 'spatial_lane_widening',
                        'phase': phase.get('name'),
                        'robot': robot_name,
                        'target': new_target_name,
                        'y_delta': y_delta,
                    }
                )

        if patches:
            phase['timeout_steps'] = max(int(phase.get('timeout_steps') or recipe.get('phase_timeout_steps') or 240), 320)
        return patches

    def _patch_interaction(self, recipe: dict[str, Any], feedback: dict[str, Any]) -> list[dict[str, Any]]:
        phase = self._phase_by_name(recipe, feedback.get('phase'))
        if phase is None:
            return []
        patches = []
        attach_specs = [attach for attach in as_list(phase.get('attach')) if isinstance(attach, dict)]
        evidence = feedback.get('evidence') or {}
        evidence_attach = evidence.get('attach_spec') if isinstance(evidence, dict) else None
        if not attach_specs and isinstance(evidence_attach, dict):
            attach_specs = [copy.deepcopy(evidence_attach)]
            phase['attach'] = attach_specs

        for attach in attach_specs:
            old_tolerance = float(attach.get('position_tolerance', 0.02))
            attach['position_tolerance'] = min(max(old_tolerance * 1.5, old_tolerance + 0.01), 0.08)
            attach['require_target_reached_for_attach'] = True
            attach['require_physical_contact'] = bool(attach.get('require_physical_contact', True))
            if 'finger_contact_distance' in attach:
                attach['finger_contact_distance'] = min(float(attach.get('finger_contact_distance', 0.008)) * 1.35, 0.02)
            if 'physical_attach_surface_gap' in attach:
                attach['physical_attach_surface_gap'] = min(float(attach.get('physical_attach_surface_gap', 0.008)) * 1.25, 0.02)
            patches.append(
                {
                    'type': 'interaction_grasp_relaxation',
                    'phase': phase.get('name'),
                    'object': attach.get('object'),
                    'robot': attach.get('robot'),
                    'old_position_tolerance': old_tolerance,
                    'new_position_tolerance': attach['position_tolerance'],
                }
            )

        old_timeout = int(phase.get('timeout_steps') or recipe.get('phase_timeout_steps') or 240)
        phase['timeout_steps'] = max(old_timeout + 90, int(old_timeout * 1.3))
        advance = phase.get('advance')
        if isinstance(advance, dict) and advance.get('type') == 'timer':
            advance['min_steps'] = max(int(advance.get('min_steps') or 12), 36)
        return patches

    @staticmethod
    def _phase_by_name(recipe: dict[str, Any], phase_name: Any) -> dict[str, Any] | None:
        if not phase_name:
            return None
        for phase in recipe.get('phases', []):
            if isinstance(phase, dict) and phase.get('name') == phase_name:
                return phase
        return None

    @staticmethod
    def _target_by_name(recipe: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(target.get('name')): target
            for target in recipe.get('targets', [])
            if isinstance(target, dict) and target.get('name')
        }

    def _shift_target_for_robot(
        self,
        *,
        recipe: dict[str, Any],
        target_by_name: dict[str, dict[str, Any]],
        phase: dict[str, Any],
        robot_name: str,
        target_like: Any,
        y_delta: float,
        attempt: int,
    ) -> str | None:
        if isinstance(target_like, str):
            target_name = target_like
            target_spec = target_by_name.get(target_name)
            target_ref = {'target': target_name}
        elif isinstance(target_like, dict) and target_like.get('target'):
            target_name = target_like.get('target')
            target_spec = target_by_name.get(target_name)
            target_ref = copy.deepcopy(target_like)
        elif isinstance(target_like, dict) and ('position' in target_like or 'offset' in target_like):
            target_spec = target_like
            target_ref = copy.deepcopy(target_like)
            target_name = target_like.get('name') or f"{phase.get('name')}_{robot_name}_inline"
        else:
            return None
        if not isinstance(target_spec, dict):
            return None

        new_name = self._unique_target_name(
            recipe,
            f"{target_name}_{slugify_task_name(robot_name)}_safe_lane_{attempt}",
        )
        new_target = copy.deepcopy(target_spec)
        new_target['name'] = new_name
        if new_target.get('reference', 'world') == 'world' and new_target.get('position') is not None:
            position = list(new_target.get('position'))
            position[1] = float(position[1]) + float(y_delta)
            new_target['position'] = position
        else:
            offset = list(new_target.get('offset', [0.0, 0.0, 0.0]))
            offset[1] = float(offset[1]) + float(y_delta)
            new_target['offset'] = offset
        recipe.setdefault('targets', []).append(new_target)

        if isinstance(target_ref, dict):
            target_ref['target'] = new_name
            phase.setdefault('robot_targets', {})[robot_name] = target_ref
        else:
            phase.setdefault('robot_targets', {})[robot_name] = new_name
        return new_name

    @staticmethod
    def _unique_target_name(recipe: dict[str, Any], raw_name: str) -> str:
        base = slugify_task_name(raw_name, fallback='local_replan_target')
        existing = {
            str(target.get('name'))
            for target in recipe.get('targets', [])
            if isinstance(target, dict) and target.get('name')
        }
        if base not in existing:
            return base
        index = 2
        while f'{base}_{index}' in existing:
            index += 1
        return f'{base}_{index}'
