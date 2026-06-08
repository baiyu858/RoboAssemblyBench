from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from roboassemblybench.core.task_registry import list_task_recipes, load_task_recipe


def _tokens(value: str) -> set[str]:
    return set(re.findall(r'[a-z0-9]+', value.lower()))


def _compact_phase(phase: dict[str, Any]) -> dict[str, Any]:
    robot_targets = {}
    for robot_name, target_spec in (phase.get('robot_targets') or {}).items():
        if isinstance(target_spec, dict):
            robot_targets[robot_name] = {
                key: target_spec[key]
                for key in ('target', 'name', 'reference', 'offset', 'position_tolerance', 'payload_object', 'payload_target')
                if key in target_spec
            }
        else:
            robot_targets[robot_name] = target_spec
    compact = {
        'name': phase.get('name'),
        'robot_targets': robot_targets,
        'gripper_commands': phase.get('gripper_commands', {}),
        'advance': {'type': (phase.get('advance') or {}).get('type')},
    }
    for key in ('attach', 'detach', 'handoff', 'transfer', 'lock'):
        if key in phase:
            compact[key] = phase[key]
    return compact


def _compact_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        'task_name': recipe.get('task_name'),
        'prompt': recipe.get('prompt'),
        'task_description': recipe.get('task_description'),
        'robots': [robot.get('name') for robot in recipe.get('robots', [])],
        'objects': [
            {
                'name': item.get('name'),
                'kind': item.get('kind'),
                'position': item.get('position'),
                'scale': item.get('scale'),
                'tracked': item.get('tracked', True),
            }
            for item in recipe.get('objects', [])
        ],
        'targets': [
            {
                key: target[key]
                for key in ('name', 'reference', 'position', 'offset', 'orientation_euler')
                if key in target
            }
            for target in recipe.get('targets', [])
        ],
        'phases': [_compact_phase(phase) for phase in recipe.get('phases', [])],
        'success': recipe.get('success', []),
        'tags': (recipe.get('metadata') or {}).get('tags', []),
    }


@dataclass
class RoboAssemblyInventory:
    recipes: dict[str, dict[str, Any]]
    compact_recipes: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, *, scene_profile: str | None = None) -> 'RoboAssemblyInventory':
        recipes = {}
        compact = {}
        for recipe_name in list_task_recipes():
            recipe = load_task_recipe(recipe_name, scene_profile=scene_profile)
            recipes[recipe_name] = recipe
            compact[recipe_name] = _compact_recipe(recipe)
        return cls(recipes=recipes, compact_recipes=compact)

    def template_names(self) -> list[str]:
        return sorted(self.recipes)

    def best_template_for(self, task_instruction: str) -> str:
        instruction_tokens = _tokens(task_instruction)
        best_name = self.template_names()[0]
        best_score = -1
        for recipe_name, recipe in self.compact_recipes.items():
            text = json.dumps(recipe, ensure_ascii=False)
            recipe_tokens = _tokens(text)
            score = len(instruction_tokens & recipe_tokens)
            compact_name = recipe_name.replace('_', ' ')
            if compact_name in task_instruction.lower():
                score += 10
            for keyword in recipe_name.split('_'):
                if keyword in instruction_tokens:
                    score += 3
            if score > best_score:
                best_name = recipe_name
                best_score = score
        return best_name

    def prompt_payload(self, *, max_chars: int = 24000) -> str:
        payload = {
            'available_templates': self.template_names(),
            'recipes': self.compact_recipes,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + '\n... TRUNCATED INVENTORY ...'
