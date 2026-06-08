from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ROBOT_NAMES = ('franka_left', 'franka_right')
CONSTRAINT_CATEGORIES = ('Logical', 'Temporal', 'Spatial')


def slugify_task_name(value: str, *, fallback: str = 'robobrain_task') -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', str(value).strip().lower()).strip('_')
    if not slug:
        return fallback
    if slug[0].isdigit():
        slug = f'task_{slug}'
    return slug[:80]


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_constraint_category(category: str) -> str:
    lookup = {item.lower(): item for item in CONSTRAINT_CATEGORIES}
    return lookup.get(str(category).strip().lower(), str(category).strip())


@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    check_code: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'errors': list(self.errors),
            'warnings': list(self.warnings),
            'check_code': list(self.check_code),
        }


@dataclass
class RoboBrainPlan:
    task_name: str
    task_instruction: str
    selected_template: str
    rationale: str = ''
    assumptions: list[str] = field(default_factory=list)
    subgoals: dict[str, list[str]] = field(default_factory=dict)
    constraints: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    skills: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    success: list[dict[str, Any]] = field(default_factory=list)
    grounding: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    source: str = 'llm'

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, task_instruction: str, source: str = 'llm') -> 'RoboBrainPlan':
        raw_task_name = payload.get('task_name') or payload.get('name') or task_instruction
        subgoals = {}
        for robot_name, robot_subgoals in (payload.get('subgoals') or payload.get('Subgoals') or {}).items():
            subgoals[str(robot_name)] = [str(item) for item in as_list(robot_subgoals)]

        raw_constraints = payload.get('constraints') or payload.get('Constraints') or {}
        constraints: dict[str, list[dict[str, Any]]] = {category: [] for category in CONSTRAINT_CATEGORIES}
        for category, entries in raw_constraints.items():
            normalized_category = normalize_constraint_category(category)
            if normalized_category not in constraints:
                constraints[normalized_category] = []
            for entry in as_list(entries):
                if isinstance(entry, str):
                    constraints[normalized_category].append({'constraint': entry})
                elif isinstance(entry, dict):
                    constraints[normalized_category].append(dict(entry))

        return cls(
            task_name=slugify_task_name(raw_task_name),
            task_instruction=str(payload.get('task_instruction') or payload.get('instruction') or task_instruction),
            selected_template=str(payload.get('selected_template') or payload.get('template') or ''),
            rationale=str(payload.get('rationale') or payload.get('reasoning_summary') or ''),
            assumptions=[str(item) for item in as_list(payload.get('assumptions'))],
            subgoals=subgoals,
            constraints=constraints,
            skills=[
                dict(item)
                for item in as_list(
                    payload.get('skills') or payload.get('skill_plan') or payload.get('primitive_skills')
                )
                if isinstance(item, dict)
            ],
            targets=[dict(item) for item in as_list(payload.get('targets')) if isinstance(item, dict)],
            phases=[dict(item) for item in as_list(payload.get('phases')) if isinstance(item, dict)],
            success=[dict(item) for item in as_list(payload.get('success')) if isinstance(item, dict)],
            grounding=dict(payload.get('grounding') or payload.get('perception_grounding') or {}),
            raw_response=dict(payload),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_name': self.task_name,
            'task_instruction': self.task_instruction,
            'selected_template': self.selected_template,
            'rationale': self.rationale,
            'assumptions': list(self.assumptions),
            'subgoals': self.subgoals,
            'constraints': self.constraints,
            'skills': self.skills,
            'targets': self.targets,
            'phases': self.phases,
            'success': self.success,
            'grounding': self.grounding,
            'source': self.source,
            'raw_response': self.raw_response,
        }
