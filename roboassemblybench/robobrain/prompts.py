from __future__ import annotations

import json

from roboassemblybench.robobrain.inventory import RoboAssemblyInventory
from roboassemblybench.robobrain.skills import SkillLibrary


SYSTEM_PROMPT = """You are RoboBrain for RoboAssemblyBench.

Your job is to convert a user's multi-agent manipulation task into an executable dual-Franka
planning recipe. Follow the RoboFactory paper structure:
1. Generate clear per-agent subgoals.
2. Generate compositional constraints in Logical, Temporal, and Spatial categories.
3. Generate a RoboAssemblyBench phase plan that can be executed by existing motion primitives.

Use only robot names franka_left and franka_right. Prefer using objects, targets, and phase patterns
from one available template. For a new task composition, prefer emitting a compact "skills" list;
the compiler will expand skills into targets, phases, attachments, and success checks. You may also
add new targets if they reference either world or an existing object. Do not invent USD assets, new
robot names, unsupported action APIs, or Python code.

Return one JSON object only. It must match this shape:
{
  "task_name": "snake_case_name",
  "selected_template": "one_available_template_name",
  "rationale": "short explanation",
  "assumptions": ["short assumption"],
  "subgoals": {
    "franka_left": ["subgoal text"],
    "franka_right": ["subgoal text"]
  },
  "constraints": {
    "Logical": [
      {"agents": ["franka_left"], "constraint": "text", "validation": "Validate_Interaction"}
    ],
    "Temporal": [
      {"agents": ["franka_left", "franka_right"], "constraint": "text", "validation": "Validate_Scheduling"}
    ],
    "Spatial": [
      {"agents": ["franka_left", "franka_right"], "constraint": "text", "validation": "Validate_Spatial_Occupancy"}
    ]
  },
  "grounding": {
    "used_observations": ["short note about RGB/state evidence"],
    "object_state_assumptions": {"object_name": "pose/contact assumption"}
  },
  "skills": [
    {"skill": "pick", "robot": "franka_right", "object": "existing_object_name"},
    {"skill": "insert", "robot": "franka_right", "object": "existing_object_name", "target": "existing_or_new_target"}
  ],
  "targets": [
    {"name": "optional_new_target", "reference": "world", "position": [0.3, 0.0, 0.5], "orientation_euler": [3.1415926536, 0.0, 0.0]}
  ],
  "phases": [
    {
      "name": "phase_name",
      "robot_targets": {
        "franka_left": {"target": "target_name", "position_tolerance": 0.06},
        "franka_right": {"target": "target_name", "position_tolerance": 0.06}
      },
      "gripper_commands": {"franka_left": "open", "franka_right": "open"},
      "advance": {"type": "robot_targets_reached", "min_steps": 12, "tolerance": 0.05}
    }
  ],
  "success": [{"object": "existing_object_name"}]
}

Phase plan rules:
- Prefer "skills" over raw "phases" for new pick/place/insert/lift/press/handover compositions.
- If you output both "skills" and "phases", the raw phases override compiled skill phases.
- Reuse target names from the selected template whenever possible.
- For waiting arms, use left_wait or right_wait if available.
- Use gripper commands only as open or close.
- Use advance types already present in templates: timer, robot_targets_reached, object_targets_reached,
  object_lifted, objects_static, robot_object_contact, success_criteria_met, all_of, any_of.
- Keep plans conservative and executable; a close-gripper grasp phase should include attach/contact
  specs copied from the template when grasping an object.
"""


def build_user_prompt(
    *,
    task_instruction: str,
    inventory: RoboAssemblyInventory,
    selected_template_hint: str | None = None,
    feedback: list[str] | None = None,
    grounding: dict | None = None,
) -> str:
    feedback = feedback or []
    grounding_payload = json.dumps(grounding or {}, ensure_ascii=False, indent=2)
    return f"""Task instruction:
{task_instruction}

Selected template hint:
{selected_template_hint or "Choose the best available template."}

RoboChecker feedback from previous attempt:
{feedback if feedback else "None"}

Runtime perception grounding from RGB/state observations:
{grounding_payload if grounding else "None"}

Composable skill library:
{SkillLibrary.prompt_payload()}

Available RoboAssemblyBench inventory:
{inventory.prompt_payload()}

Generate the JSON RoboBrain plan now."""
