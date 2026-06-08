from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from roboassemblybench.robobrain.inventory import RoboAssemblyInventory


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith('{'):
        return json.loads(text)
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError('Model response did not contain a JSON object.')


def _image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or 'image/png'
    payload = base64.b64encode(image_path.read_bytes()).decode('ascii')
    return f'data:{mime_type};base64,{payload}'


class OpenAIRoboBrainClient:
    def __init__(self, *, model: str | None = None, temperature: float = 0.2):
        self.model = model or os.environ.get('ROBOBRAIN_MODEL') or 'gpt-4o'
        self.temperature = float(temperature)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        observation_images: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('The openai package is required for online RoboBrain planning.') from exc

        if not os.environ.get('OPENAI_API_KEY'):
            raise RuntimeError('OPENAI_API_KEY is required for online RoboBrain planning. Use --mock-llm for a local dry run.')

        base_url = os.environ.get('OPENAI_BASE_URL')
        client_kwargs = {'api_key': os.environ.get('OPENAI_API_KEY')}
        if base_url:
            client_kwargs['base_url'] = base_url
        client = OpenAI(**client_kwargs)

        user_content: str | list[dict[str, Any]]
        if observation_images:
            user_content = [{'type': 'text', 'text': user_prompt}]
            for image_path in observation_images:
                user_content.append({'type': 'image_url', 'image_url': {'url': _image_data_url(image_path)}})
        else:
            user_content = user_prompt

        request = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'},
        }
        try:
            response = client.chat.completions.create(**request)
        except TypeError:
            request.pop('response_format', None)
            response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or ''
        return extract_json_object(content)


VISUAL_GROUNDING_SYSTEM_PROMPT = """You are the visual grounding module for RoboBrain.

Analyze RoboAssemblyBench RGB observations together with runtime state summaries. Return one JSON
object only. Keep the output compact and actionable for robot planning. Do not invent unavailable
assets; if something is uncertain, mark confidence below 0.5.

Required JSON shape:
{
  "scene_summary": "short visual summary",
  "visible_objects": [
    {
      "name": "object name if known or descriptive label",
      "category": "tool|fixture|robot|workbench|unknown",
      "confidence": 0.0,
      "image_path": "path if known",
      "bbox_2d": [x_min, y_min, x_max, y_max],
      "spatial_hint": "left/right/front/back/center/near gripper",
      "state_hint": "free|grasped|attached|misaligned|occluded|unknown"
    }
  ],
  "visible_relations": [
    {"relation": "near|touching|inside|above|occluding|misaligned", "subject": "name", "object": "name", "confidence": 0.0}
  ],
  "task_risks": [
    {"risk": "short risk", "evidence": "visual or state evidence", "severity": "low|medium|high"}
  ],
  "suggested_planner_adjustments": ["short actionable adjustment"]
}
"""


class OpenAIVisualGroundingClient:
    def __init__(self, *, model: str | None = None, temperature: float = 0.0):
        self.model = model or os.environ.get('ROBOBRAIN_VISION_MODEL') or os.environ.get('ROBOBRAIN_MODEL') or 'gpt-4o'
        self.temperature = float(temperature)

    def ground(
        self,
        *,
        image_paths: list[str],
        state_grounding: dict[str, Any],
        task_instruction: str | None = None,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('The openai package is required for VLM visual grounding.') from exc

        if not os.environ.get('OPENAI_API_KEY'):
            raise RuntimeError('OPENAI_API_KEY is required for VLM visual grounding.')

        base_url = os.environ.get('OPENAI_BASE_URL')
        client_kwargs = {'api_key': os.environ.get('OPENAI_API_KEY')}
        if base_url:
            client_kwargs['base_url'] = base_url
        client = OpenAI(**client_kwargs)

        user_prompt = {
            'task_instruction': task_instruction or '',
            'state_grounding': state_grounding,
            'image_paths': image_paths,
        }
        user_content: list[dict[str, Any]] = [
            {
                'type': 'text',
                'text': (
                    'Ground these RoboAssemblyBench runtime observations for planning. '
                    'Use image_path values from the supplied list when referring to images.\n'
                    + json.dumps(user_prompt, ensure_ascii=False, indent=2)
                ),
            }
        ]
        for image_path in image_paths:
            user_content.append({'type': 'image_url', 'image_url': {'url': _image_data_url(image_path)}})

        request = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': VISUAL_GROUNDING_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'},
        }
        try:
            response = client.chat.completions.create(**request)
        except TypeError:
            request.pop('response_format', None)
            response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or ''
        return extract_json_object(content)


class MockRoboBrainClient:
    """Deterministic local stand-in used by tests and plan-only smoke runs."""

    def generate(
        self,
        *,
        task_instruction: str,
        inventory: RoboAssemblyInventory,
        selected_template: str | None = None,
        feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        template_name = selected_template or inventory.best_template_for(task_instruction)
        template = inventory.recipes[template_name]
        phases = template.get('phases', [])
        success = template.get('success', [])
        return {
            'task_name': f"robobrain_{template_name}",
            'selected_template': template_name,
            'rationale': 'Local mock plan reuses the closest executable RoboAssemblyBench template.',
            'assumptions': [
                'The requested task can be represented with existing RoboAssemblyBench assets and targets.',
                'The generated demo uses the benchmark dual-Franka execution policy.',
            ],
            'subgoals': {
                'franka_left': [
                    f"Follow the {template_name} left-arm role where applicable.",
                    'Avoid interfering with the right arm while maintaining a safe waiting or manipulation pose.',
                ],
                'franka_right': [
                    f"Follow the {template_name} right-arm role where applicable.",
                    'Execute grasp, transport, insertion, or pressing motions specified by the selected template.',
                ],
            },
            'constraints': {
                'Logical': [
                    {
                        'agents': ['franka_left', 'franka_right'],
                        'constraint': 'Each gripper must interact only with the object/contact target assigned in the active phase.',
                        'validation': 'Validate_Interaction',
                    }
                ],
                'Temporal': [
                    {
                        'agents': ['franka_left', 'franka_right'],
                        'constraint': 'The phase order must be executed sequentially, while robot targets inside one phase may run in parallel.',
                        'validation': 'Validate_Scheduling',
                    }
                ],
                'Spatial': [
                    {
                        'agents': ['franka_left', 'franka_right'],
                        'constraint': 'The arms must keep separate approach lanes and avoid end-effector overlap near the shared workspace.',
                        'validation': 'Validate_Spatial_Occupancy',
                    }
                ],
            },
            'targets': [],
            'phases': phases,
            'success': success,
            'feedback_acknowledged': feedback or [],
        }
