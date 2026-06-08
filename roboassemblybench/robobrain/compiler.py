from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from roboassemblybench.robobrain.models import CheckResult, RoboBrainPlan
from roboassemblybench.robobrain.skills import compile_skill_plan


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _merge_targets(base_targets: list[dict[str, Any]], generated_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets_by_name = {target.get('name'): copy.deepcopy(target) for target in base_targets if target.get('name')}
    for target in generated_targets:
        name = target.get('name')
        if not name:
            continue
        targets_by_name[name] = copy.deepcopy(target)
    return list(targets_by_name.values())


def build_primitive_plan(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    primitive_plan = []
    for phase in recipe.get('phases', []):
        primitives = []
        for robot_name, target_spec in (phase.get('robot_targets') or {}).items():
            primitives.append({'primitive': 'MOVE', 'robot': robot_name, 'target': target_spec})
        for robot_name, command in (phase.get('gripper_commands') or {}).items():
            primitives.append({'primitive': 'GRIPPER', 'robot': robot_name, 'command': command})
        for attach_spec in phase.get('attach', []) if isinstance(phase.get('attach', []), list) else [phase.get('attach')]:
            if isinstance(attach_spec, dict):
                primitives.append({'primitive': 'ATTACH', **copy.deepcopy(attach_spec)})
        for detach_spec in phase.get('detach', []) if isinstance(phase.get('detach', []), list) else [phase.get('detach')]:
            if detach_spec:
                primitives.append({'primitive': 'DETACH', 'spec': copy.deepcopy(detach_spec)})
        primitive_plan.append({'phase': phase.get('name'), 'primitives': primitives, 'advance': phase.get('advance', {})})
    return primitive_plan


def compile_plan_to_recipe(
    *,
    plan: RoboBrainPlan,
    base_recipe: dict[str, Any],
    scene_profile: str | None = None,
) -> dict[str, Any]:
    recipe = copy.deepcopy(base_recipe)
    recipe['task_name'] = plan.task_name
    recipe['prompt'] = plan.task_instruction
    recipe['task_description'] = plan.task_instruction
    recipe['scene_profile'] = scene_profile or recipe.get('scene_profile')
    skill_result = compile_skill_plan(plan.skills, base_recipe=recipe) if plan.skills else None
    generated_targets = []
    if skill_result is not None:
        generated_targets.extend(skill_result.targets)
    generated_targets.extend(plan.targets)
    recipe['targets'] = _merge_targets(recipe.get('targets', []), generated_targets)
    if plan.phases:
        recipe['phases'] = copy.deepcopy(plan.phases)
    elif skill_result is not None:
        recipe['phases'] = copy.deepcopy(skill_result.phases)
    if plan.success:
        recipe['success'] = copy.deepcopy(plan.success)
    elif skill_result is not None and skill_result.success:
        recipe['success'] = copy.deepcopy(skill_result.success)

    metadata = copy.deepcopy(recipe.get('metadata', {}))
    metadata['robobrain'] = {
        'selected_template': plan.selected_template,
        'rationale': plan.rationale,
        'assumptions': list(plan.assumptions),
        'subgoals': copy.deepcopy(plan.subgoals),
        'constraints': copy.deepcopy(plan.constraints),
        'skills': copy.deepcopy(plan.skills),
        'skill_compile_warnings': [] if skill_result is None else list(skill_result.warnings),
        'skill_primitive_plan': [] if skill_result is None else copy.deepcopy(skill_result.primitive_plan),
        'grounding': copy.deepcopy(plan.grounding),
        'primitive_plan': build_primitive_plan(recipe),
        'source': plan.source,
    }
    recipe['metadata'] = metadata
    return _jsonable(recipe)


def write_recipe_bundle(
    *,
    output_dir: Path,
    plan: RoboBrainPlan,
    recipe: dict[str, Any],
    check_result: CheckResult,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = output_dir / 'recipe.yaml'
    plan_path = output_dir / 'plan.json'
    checker_path = output_dir / 'checker_report.json'
    primitive_plan_path = output_dir / 'primitive_plan.json'
    annotation_path = output_dir / 'annotation.yaml'

    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding='utf-8')
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
    checker_path.write_text(json.dumps(check_result.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
    primitive_plan_path.write_text(
        json.dumps(build_primitive_plan(recipe), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    annotation_payload = {
        'task_name': plan.task_name,
        'title': plan.task_name.replace('_', ' ').title(),
        'summary': plan.task_instruction,
        'task_description': plan.task_instruction,
        'metadata': {
            'authoring_stack': 'RoboBrain',
            'selected_template': plan.selected_template,
            'subgoals': plan.subgoals,
            'constraints': plan.constraints,
        },
    }
    annotation_path.write_text(yaml.safe_dump(annotation_payload, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return {
        'recipe': recipe_path,
        'plan': plan_path,
        'checker_report': checker_path,
        'primitive_plan': primitive_plan_path,
        'annotation': annotation_path,
    }
