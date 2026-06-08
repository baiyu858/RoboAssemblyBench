from pathlib import Path

import pytest

from roboassemblybench.core.task_registry import load_task_recipe
from roboassemblybench.robobrain.checker import RoboChecker
from roboassemblybench.robobrain.compiler import compile_plan_to_recipe
from roboassemblybench.robobrain.executor import RoboBrainRunConfig, RoboBrainRunner
from roboassemblybench.robobrain.llm import extract_json_object
from roboassemblybench.robobrain.models import RoboBrainPlan
from roboassemblybench.robobrain.perception import ObservationGrounder
from roboassemblybench.robobrain.replanner import LocalReplanner
from roboassemblybench.robobrain.runtime_monitor import RuntimeRoboChecker


def test_extract_json_object_from_fenced_response():
    payload = extract_json_object('```json\n{"task_name": "demo", "selected_template": "peg_insertion"}\n```')
    assert payload['task_name'] == 'demo'


def test_mock_robobrain_generates_loadable_recipe(tmp_path: Path):
    result = RoboBrainRunner(
        RoboBrainRunConfig(
            output_dir=tmp_path,
            selected_template='peg_insertion',
            mock_llm=True,
            plan_only=True,
        )
    ).run('right Franka inserts a peg into the socket')

    assert result.check_result.ok
    assert result.bundle_paths['recipe'].exists()
    assert result.bundle_paths['plan'].exists()
    assert result.bundle_paths['primitive_plan'].exists()

    recipe = load_task_recipe(str(result.bundle_paths['recipe']), scene_profile='taoyuan_grscenes_tabletop')
    assert recipe['task_name'] == 'robobrain_peg_insertion'
    assert recipe['metadata']['robobrain']['selected_template'] == 'peg_insertion'
    assert recipe['phases']


def test_checker_rejects_unknown_phase_target():
    base_recipe = load_task_recipe('peg_insertion', scene_profile='taoyuan_grscenes_tabletop')
    recipe = dict(base_recipe)
    recipe['phases'] = [
        {
            'name': 'bad_phase',
            'robot_targets': {'franka_right': {'target': 'missing_target'}},
            'gripper_commands': {'franka_right': 'open'},
            'advance': {'type': 'robot_targets_reached'},
        }
    ]
    plan = RoboBrainPlan.from_dict(
        {
            'task_name': 'bad_plan',
            'selected_template': 'peg_insertion',
            'constraints': {
                'Logical': [{'agents': ['franka_right'], 'constraint': 'Interact with target.'}],
                'Temporal': [{'agents': ['franka_right'], 'constraint': 'Execute in order.'}],
                'Spatial': [{'agents': ['franka_right'], 'constraint': 'Avoid collisions.'}],
            },
            'phases': recipe['phases'],
        },
        task_instruction='bad target plan',
    )

    result = RoboChecker().check(plan=plan, recipe=recipe)
    assert not result.ok
    assert any('missing_target' in error for error in result.errors)


class _FakeRuntimeTask:
    phase = 'bad_grasp'
    step_counter = 96

    def get_current_phase_spec(self):
        return {
            'name': 'bad_grasp',
            'attach': [{'object': 'peg', 'robot': 'franka_right', 'target': 'peg_grasp_close'}],
        }

    def get_tracked_object_states(self):
        return {'peg': {'position': [0.4, 0.0, 0.1], 'attached_to': None, 'status': 'free'}}

    def get_tracked_robot_states(self, phase_spec=None):
        return {
            'franka_left': {
                'position': [0.0, 0.0, 0.0],
                'position_error': 0.1,
                'target_reached': False,
                'target_name': 'left_wait',
            },
            'franka_right': {
                'position': [0.01, 0.0, 0.0],
                'position_error': 0.1,
                'target_reached': False,
                'target_name': 'peg_grasp_close',
            },
        }

    def get_phase_runtime_state(self):
        return {'phase_step_counter': 96, 'phase_status': 'running', 'timeout_remaining': 100}


def test_runtime_robochecker_reports_dynamic_feedback():
    checker = RuntimeRoboChecker(capture_rgb=False, check_stride=1, min_steps_before_stop=1)
    result = checker.observe(task=_FakeRuntimeTask(), obs={}, actions={}, episode_idx=0)

    assert result['blocking']
    validations = {item['validation'] for item in result['feedback']}
    assert 'Validate_Interaction' in validations
    assert 'Validate_Spatial_Occupancy' in validations


def test_demo_command_enables_runtime_feedback(tmp_path: Path):
    runner = RoboBrainRunner(RoboBrainRunConfig(output_dir=tmp_path, mock_llm=True, plan_only=True))
    command = runner._demo_command(
        recipe_path=tmp_path / 'recipe.yaml',
        output_dir=tmp_path / 'demo',
        runtime_feedback_path=tmp_path / 'runtime_feedback.json',
        runtime_observation_dir=tmp_path / 'runtime_observations',
    )
    assert '--runtime-robochecker' in command
    assert '--runtime-feedback-path' in command
    assert '--runtime-observation-dir' in command


def test_skill_plan_generates_new_targets_and_phases():
    base_recipe = load_task_recipe('peg_insertion', scene_profile='taoyuan_grscenes_tabletop')
    plan = RoboBrainPlan.from_dict(
        {
            'task_name': 'skill_generated_pick_place',
            'selected_template': 'peg_insertion',
            'constraints': {
                'Logical': [{'agents': ['franka_right'], 'constraint': 'The peg is grasped before it is placed.'}],
                'Temporal': [{'agents': ['franka_right'], 'constraint': 'Pick happens before place.'}],
                'Spatial': [{'agents': ['franka_left', 'franka_right'], 'constraint': 'Keep the idle arm in a wait lane.'}],
            },
            'skills': [
                {'skill': 'pick', 'robot': 'franka_right', 'object': 'peg'},
                {
                    'skill': 'place',
                    'robot': 'franka_right',
                    'object': 'peg',
                    'target': 'peg_generated_place_goal',
                    'target_position': [0.48, 0.10, 0.06],
                },
            ],
        },
        task_instruction='Pick the peg and place it at a new goal.',
    )

    recipe = compile_plan_to_recipe(plan=plan, base_recipe=base_recipe, scene_profile='taoyuan_grscenes_tabletop')
    result = RoboChecker().check(plan=plan, recipe=recipe)

    assert result.ok
    assert any(target['name'] == 'peg_generated_place_goal' for target in recipe['targets'])
    assert recipe['metadata']['robobrain']['skills']
    assert recipe['phases'][0]['name'] == 'right_peg_approach'


def test_observation_grounder_summarizes_runtime_state(tmp_path: Path):
    state_dir = tmp_path / 'episode_0000' / 'state'
    state_dir.mkdir(parents=True)
    state_path = state_dir / 'step_000096.json'
    state_path.write_text(
        """{
          "episode_idx": 0,
          "step_index": 96,
          "phase": "bad_grasp",
          "tracked_objects": {
            "peg": {"position": [0.4, 0.0, 0.1], "attached_to": null, "target_reached": false, "position_error": 0.12}
          },
          "tracked_robots": {
            "franka_right": {"position": [0.4, 0.0, 0.2], "target_name": "peg_grasp_close", "target_reached": false, "position_error": 0.08}
          },
          "runtime_state": {"phase_step_counter": 96}
        }""",
        encoding='utf-8',
    )
    feedback = {
        'feedback': [
            {
                'severity': 'error',
                'validation': 'Validate_Interaction',
                'phase': 'bad_grasp',
                'message': 'Expected attachment did not happen.',
            }
        ],
        'latest_state_paths': [str(state_path)],
        'observation_dir': str(tmp_path),
    }

    grounding = ObservationGrounder(max_images=0).ground(runtime_feedback=feedback)

    assert grounding['state_observations'][0]['objects']['peg']['position_error'] == 0.12
    assert any(item['relation'] == 'robot_not_at_target' for item in grounding['inferred_relations'])
    assert grounding['planner_hints']


def test_local_visual_grounder_detects_colored_regions(tmp_path: Path):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[25:70, 30:75] = [255, 0, 0]
    image[55:105, 95:140] = [0, 255, 0]
    image_path = tmp_path / 'runtime_rgb.png'
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    grounding = ObservationGrounder(
        max_images=1,
        visual_backend='local',
        visual_labels=['peg', 'green fixture'],
    ).ground(
        observation_images=[str(image_path)],
        task_instruction='pick the red peg near the green fixture',
    )

    visual = grounding['visual_grounding']
    assert visual['backend'] == 'local'
    assert 'peg' in visual['labels']
    assert len(visual['detections']) >= 2
    assert all('bbox_2d' in item for item in visual['detections'])


def test_local_replanner_patches_runtime_feedback():
    recipe = load_task_recipe('peg_insertion', scene_profile='taoyuan_grscenes_tabletop')
    plan = RoboBrainPlan.from_dict(
        {
            'task_name': 'local_replan_demo',
            'selected_template': 'peg_insertion',
            'constraints': {
                'Logical': [{'agents': ['franka_right'], 'constraint': 'grasp'}],
                'Temporal': [{'agents': ['franka_right'], 'constraint': 'finish phase'}],
                'Spatial': [{'agents': ['franka_left', 'franka_right'], 'constraint': 'separate lanes'}],
            },
            'phases': recipe['phases'],
            'success': recipe['success'],
        },
        task_instruction='demo',
    )
    feedback = {
        'feedback': [
            {'validation': 'Validate_Scheduling', 'phase': 'right_grasp_peg', 'severity': 'error'},
            {'validation': 'Validate_Interaction', 'phase': 'right_grasp_peg', 'severity': 'error'},
            {'validation': 'Validate_Spatial_Occupancy', 'phase': 'right_approach_peg', 'severity': 'error'},
        ]
    }

    local = LocalReplanner().replan(plan=plan, recipe=recipe, runtime_feedback=feedback, attempt=1)
    assert local is not None
    result = RoboChecker().check(plan=local.plan, recipe=local.recipe)

    assert result.ok
    assert local.report['patches']
    assert local.plan.task_name.endswith('local_replan_1')
