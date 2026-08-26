import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from internutopia_extension.tasks.factory_dual_franka_assembly_task import (
    FactoryDualFrankaAssemblyTask,
)
from roboassemblybench.core.domain_randomization import apply_domain_randomization
from roboassemblybench.datasets.cartesian_episode import CompactCartesianEpisodeRecorder
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import (
    UR5eAssemblyAtomicSkillAdapter,
    UR5ePlumbersBlockAtomicSkillAdapter,
)
from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe
from toolkits.factory_dual_franka_assembly.ur5e_skill_api import UR5eAssemblySkillAPI


def test_position_randomization_is_deterministic_and_group_correlated():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    randomized_a, result_a = apply_domain_randomization(recipe, seed=17, enabled_override=True)
    randomized_b, result_b = apply_domain_randomization(recipe, seed=17, enabled_override=True)
    _, result_c = apply_domain_randomization(recipe, seed=18, enabled_override=True)

    assert result_a == result_b
    assert result_a != result_c
    assert result_a['enabled'] is True
    assert set(result_a['groups']) == {'start_parts', 'assembly_base'}
    assert set(result_a['appearance_groups']) == {'table_surface', 'background'}
    assert 'fabrica_fixture' in result_a['groups']['start_parts']['objects']
    assert result_a['groups']['assembly_base']['objects'] == []

    original_objects = {item['name']: item for item in recipe['objects']}
    randomized_objects = {item['name']: item for item in randomized_a['objects']}
    start_delta = np.asarray(result_a['groups']['start_parts']['translation'])
    for object_name in result_a['groups']['start_parts']['objects']:
        actual_delta = np.asarray(randomized_objects[object_name]['position']) - np.asarray(
            original_objects[object_name]['position']
        )
        np.testing.assert_allclose(actual_delta, start_delta)

    original_targets = {item['name']: item for item in recipe['targets']}
    randomized_targets = {item['name']: item for item in randomized_a['targets']}
    assembly_delta = np.asarray(result_a['groups']['assembly_base']['translation'])
    for target_name in result_a['groups']['assembly_base']['targets']:
        actual_delta = np.asarray(randomized_targets[target_name]['position']) - np.asarray(
            original_targets[target_name]['position']
        )
        np.testing.assert_allclose(actual_delta, assembly_delta)

    np.testing.assert_allclose(
        randomized_objects['optical_board']['position'],
        original_objects['optical_board']['position'],
    )
    assert 'domain_randomization_group' not in randomized_objects['optical_board']

    table_color = result_a['appearance_groups']['table_surface']['color']
    np.testing.assert_allclose(randomized_objects['factory_tabletop_visual']['color'], table_color)
    background_color = result_a['appearance_groups']['background']['color']
    assert result_a['appearance_groups']['background']['objects'] == []
    randomized_lights = {item['name']: item for item in randomized_a['scene_lights']}
    np.testing.assert_allclose(randomized_lights['warehouse_dome_fill']['color'], background_color)
    assert 60.0 <= randomized_lights['warehouse_dome_fill']['intensity'] <= 160.0
    assert -0.35 <= randomized_lights['warehouse_dome_fill']['exposure'] <= 0.0


def test_appearance_randomization_rejects_members_outside_the_allowlist():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    recipe['domain_randomization']['appearance']['groups']['table_surface']['objects'].append('optical_board')

    with pytest.raises(ValueError, match='outside appearance.allowed_objects'):
        apply_domain_randomization(recipe, seed=17, enabled_override=True)


def test_visual_distractors_stay_outside_robot_base_keepouts():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )

    randomized, result = apply_domain_randomization(recipe, seed=17, enabled_override=True)

    keepout_radius = recipe['domain_randomization']['visual_distractors']['robot_keepout_radius']
    workspace_offset = np.asarray(randomized['workspace_offset'], dtype=float)
    robot_positions = []
    for robot in randomized['robots']:
        position = np.asarray(robot['position'], dtype=float)
        if robot.get('apply_workspace_offset', True):
            position = position + workspace_offset
        robot_positions.append(position)
    assert result['visual_distractors']
    for distractor in result['visual_distractors']:
        distractor_xy = np.asarray(distractor['position'][:2], dtype=float)
        assert all(
            np.linalg.norm(distractor_xy - robot_position[:2]) >= keepout_radius
            for robot_position in robot_positions
        )


@pytest.mark.parametrize(
    ('profile', 'appearance_groups'),
    [
        ('object_distractors', set()),
        ('texture', set()),
        ('lighting', set()),
        ('table_color', {'table_surface'}),
        ('scene', {'background'}),
    ],
)
def test_randomization_profiles_keep_position_randomization_and_isolate_visual_domain(
    profile,
    appearance_groups,
):
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )

    randomized, result = apply_domain_randomization(
        recipe,
        seed=17,
        enabled_override=True,
        profile=profile,
    )

    assert result['profile'] == profile
    assert set(result['groups']) == {'start_parts', 'assembly_base'}
    assert set(result['appearance_groups']) == appearance_groups
    if profile != 'object_distractors':
        assert not result['visual_distractors']
    else:
        assert 0 <= len(result['visual_distractors']) <= 8
    assert bool(result['table_texture']) is (profile == 'texture')
    assert bool(result['lighting']) is (profile == 'lighting')
    assert bool(result['scene']) is (profile == 'scene')
    if profile == 'texture':
        texture_objects = {
            item['name']: item
            for item in randomized['objects']
            if item['name'] in {'factory_tabletop_visual', 'factory_background_visual', 'factory_floor_visual'}
        }
        assert set(result['table_texture']['surface_map']) == {
            'factory_tabletop_visual',
            'factory_background_visual',
            'factory_floor_visual',
        }
        for surface in result['table_texture']['surfaces']:
            assert Path(surface['path']).is_file()
            assert texture_objects[surface['object']]['texture_path'] == surface['path']
    if profile == 'lighting':
        assert 3 <= len(randomized['scene_lights']) <= 5
        assert all(0.7 <= item['domain_randomization_intensity_multiplier'] <= 1.3 for item in result['lighting'])
        assert result['lighting'][0]['name'] == 'warehouse_dome_fill'
        area_lights = result['lighting'][1:]
        assert 2 <= len(area_lights) <= 4
        assert all(abs(value) <= 1.0 for item in area_lights for value in item['position'][:2])
        assert all(3.2 <= item['position'][2] <= 3.8 for item in area_lights)
    if profile == 'scene':
        assert Path(result['scene']['asset_path']).name in {
            'warehouse_with_forklifts.usd',
            'warehouse.usd',
            'warehouse_multiple_shelves.usd',
            'full_warehouse.usd',
        }
        assert result['scene']['variant'] in set(result['scene']['available_variants'])
        assert abs(result['scene']['position'][0]) <= 0.08
        assert abs(result['scene']['position'][1]) <= 0.08
        assert result['scene']['position'][2] == 0.0


def test_table_color_profile_varies_only_the_table_surface_color():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    original_table = next(item for item in recipe['objects'] if item['name'] == 'factory_tabletop_visual')
    sampled_colors = set()

    for seed in range(16):
        randomized, result = apply_domain_randomization(
            recipe,
            seed=seed,
            enabled_override=True,
            profile='table_color',
        )
        randomized_table = next(
            item for item in randomized['objects'] if item['name'] == 'factory_tabletop_visual'
        )
        sampled_colors.add(tuple(result['appearance_groups']['table_surface']['color']))
        np.testing.assert_allclose(randomized_table['position'], original_table['position'])
        np.testing.assert_allclose(randomized_table['scale'], original_table['scale'])

    assert len(sampled_colors) > 1


def test_object_distractor_profile_samples_zero_to_eight_objects():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    counts = []
    for seed in range(32):
        _, result = apply_domain_randomization(
            recipe,
            seed=seed,
            enabled_override=True,
            profile='object_distractors',
        )
        counts.append(len(result['visual_distractors']))

    assert min(counts) == 0
    assert max(counts) <= 8
    assert len(set(counts)) > 1


def test_fixed_object_cannot_be_added_to_a_position_randomization_group():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    recipe['domain_randomization']['groups']['assembly_base']['objects'] = ['optical_board']

    try:
        apply_domain_randomization(recipe, seed=17, enabled_override=True)
    except ValueError as exc:
        assert "Fixed object 'optical_board'" in str(exc)
    else:
        raise AssertionError('Expected fixed optical board randomization to be rejected.')


def test_wide_showcase_layouts_move_assembly_on_fixed_board_over_fifteen_centimeter_span():
    recipe_path = (
        Path(__file__).resolve().parents[2]
        / 'roboassemblybench'
        / 'tasks'
        / 'fabrica_plumbers_block_ur5e_right_base_prepare'
        / 'recipe_wide_15cm_showcase.yaml'
    )
    recipe = load_task_recipe(
        str(recipe_path),
        scene_profile='taoyuan_grscenes_tabletop',
    )
    layout_seeds = recipe['collection']['layout_seeds']
    translations = np.asarray(
        [
            apply_domain_randomization(recipe, seed=seed, enabled_override=True)[1]['groups']['assembly_base'][
                'translation'
            ]
            for seed in layout_seeds
        ],
        dtype=float,
    )
    grid_axis = (-0.075, -0.025, 0.025, 0.075)
    expected_xy = np.asarray([(x, y) for y in grid_axis for x in grid_axis], dtype=float)

    assert len(layout_seeds) == len(set(layout_seeds)) == 16
    np.testing.assert_allclose(translations[:, :2], expected_xy, atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(translations[:, 2], 0.0)
    assert np.ptp(translations[:, 0]) >= 0.149
    assert np.ptp(translations[:, 1]) >= 0.149
    assert recipe['domain_randomization']['fixed_objects'] == ['optical_board']
    assert recipe['domain_randomization']['groups']['assembly_base']['objects'] == []
    assert recipe['max_steps'] >= 22000

    original_objects = {item['name']: item for item in recipe['objects']}
    original_targets = {item['name']: item for item in recipe['targets']}
    for layout_seed in layout_seeds:
        randomized, result = apply_domain_randomization(
            recipe,
            seed=layout_seed,
            enabled_override=True,
        )
        randomized_objects = {item['name']: item for item in randomized['objects']}
        randomized_targets = {item['name']: item for item in randomized['targets']}
        np.testing.assert_allclose(
            randomized_objects['optical_board']['position'],
            original_objects['optical_board']['position'],
        )
        assert 'domain_randomization_group' not in randomized_objects['optical_board']
        np.testing.assert_allclose(
            np.asarray(randomized_targets['part_2_ur5e_table_staging']['position'])
            - np.asarray(original_targets['part_2_ur5e_table_staging']['position']),
            result['groups']['assembly_base']['translation'],
        )


def test_wide_30cm_showcase_moves_pickup_and_assembly_while_optical_board_stays_fixed():
    recipe_path = (
        Path(__file__).resolve().parents[2]
        / 'roboassemblybench'
        / 'tasks'
        / 'fabrica_plumbers_block_ur5e_right_base_prepare'
        / 'recipe_wide_30cm_showcase.yaml'
    )
    recipe = load_task_recipe(
        str(recipe_path),
        scene_profile='taoyuan_grscenes_tabletop',
    )
    layout_seeds = recipe['collection']['layout_seeds']
    pickup_translations = np.asarray(
        [
            apply_domain_randomization(recipe, seed=seed, enabled_override=True)[1]['groups']['start_parts'][
                'translation'
            ]
            for seed in layout_seeds
        ],
        dtype=float,
    )
    assembly_translations = np.asarray(
        [
            apply_domain_randomization(recipe, seed=seed, enabled_override=True)[1]['groups']['assembly_base'][
                'translation'
            ]
            for seed in layout_seeds
        ],
        dtype=float,
    )
    expected_xy = np.asarray(
        [
            (-0.15, 0.07),
            (-7.0 / 60.0, 0.07),
            (-1.0 / 12.0, 0.05),
            (-0.05, 0.05),
            *(
                (x, y)
                for y in (1.0 / 12.0, 7.0 / 60.0, 0.15)
                for x in (
                    -0.15,
                    -7.0 / 60.0,
                    -1.0 / 12.0,
                    -0.05,
                )
            ),
        ],
        dtype=float,
    )

    assert len(layout_seeds) == len(set(layout_seeds)) == 16
    np.testing.assert_allclose(pickup_translations[:, :2], expected_xy, atol=2.0e-3, rtol=0.0)
    np.testing.assert_allclose(pickup_translations[:, 2], 0.0)
    np.testing.assert_allclose(assembly_translations[:, 2], 0.0)
    assert np.ptp(pickup_translations[:, 0]) >= 0.099
    assert np.ptp(pickup_translations[:, 1]) >= 0.099
    assert np.ptp(assembly_translations[:, 0]) >= 0.14
    assert np.ptp(assembly_translations[:, 1]) >= 0.13
    assert np.all(pickup_translations[:, 0] >= -0.151)
    assert np.all(pickup_translations[:, 0] <= -0.049)
    assert np.all(pickup_translations[:, 1] >= 0.049)
    assert np.all(pickup_translations[:, 1] <= 0.151)
    assert np.all(np.abs(assembly_translations[:, :2]) <= 0.0751)

    original_objects = {item['name']: item for item in recipe['objects']}
    original_targets = {item['name']: item for item in recipe['targets']}
    for layout_seed in layout_seeds:
        randomized, result = apply_domain_randomization(
            recipe,
            seed=layout_seed,
            enabled_override=True,
        )
        randomized_objects = {item['name']: item for item in randomized['objects']}
        randomized_targets = {item['name']: item for item in randomized['targets']}
        np.testing.assert_allclose(
            randomized_objects['optical_board']['position'],
            original_objects['optical_board']['position'],
        )
        assert 'domain_randomization_group' not in randomized_objects['optical_board']
        np.testing.assert_allclose(
            np.asarray(randomized_targets['part_2_table_hover']['position'])
            - np.asarray(original_targets['part_2_table_hover']['position']),
            result['groups']['start_parts']['translation'],
        )
        for target_name in recipe['domain_randomization']['groups']['assembly_base']['targets']:
            np.testing.assert_allclose(
                np.asarray(randomized_targets[target_name]['position'])
                - np.asarray(original_targets[target_name]['position']),
                result['groups']['assembly_base']['translation'],
            )

    assert 'part_2_table_hover' in recipe['domain_randomization']['groups']['start_parts']['targets']
    assert recipe['max_steps'] >= 26000


def test_scene_builder_preserves_nominal_recipe_and_records_randomization():
    nominal = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        seed=5,
        scene_profile='taoyuan_grscenes_tabletop',
    )
    randomized = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        seed=5,
        scene_profile='taoyuan_grscenes_tabletop',
        domain_randomization_enabled=True,
    )

    assert nominal.domain_randomization['enabled'] is False
    assert randomized.domain_randomization['enabled'] is True
    assert len(nominal.recipe_fingerprint) == 64
    assert randomized.recipe_fingerprint == nominal.recipe_fingerprint
    start_delta = np.asarray(randomized.domain_randomization['groups']['start_parts']['translation'])
    nominal_objects = {item['name']: item for item in nominal.object_metadata}
    randomized_objects = {item['name']: item for item in randomized.object_metadata}
    np.testing.assert_allclose(
        np.asarray(randomized_objects['fabrica_plumbers_block_0']['sampled_position'])
        - np.asarray(nominal_objects['fabrica_plumbers_block_0']['sampled_position']),
        start_delta,
    )
    assert randomized_objects['fabrica_plumbers_block_0']['domain_randomization_group'] == 'start_parts'
    assert randomized_objects['fabrica_plumbers_block_3']['angular_damping'] == 2.0
    assert randomized_objects['fabrica_plumbers_block_3']['solver_position_iteration_count'] == 16
    assert randomized.target_annotations['part_1_right_hole']['domain_randomization_group'] == 'assembly_base'

    evaluation = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        seed=10001,
        scene_profile='taoyuan_grscenes_tabletop',
        domain_randomization_enabled=True,
        policy_evaluation_mode=True,
    )
    assert evaluation.policy_evaluation_mode is True
    assert evaluation.policy_success_stable_steps == 24


def test_scene_builder_decouples_episode_identity_from_layout_seed():
    first = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        seed=100,
        layout_seed=4906,
        scene_profile='taoyuan_grscenes_tabletop',
        domain_randomization_enabled=True,
    )
    second = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_ur5e_right_base_prepare',
        seed=101,
        layout_seed=4906,
        scene_profile='taoyuan_grscenes_tabletop',
        domain_randomization_enabled=True,
    )

    assert first.seed == 100
    assert second.seed == 101
    assert first.layout_seed == second.layout_seed == 4906
    assert first.domain_randomization == second.domain_randomization


def test_ur5e_atomic_skill_api_compiles_runtime_contract_and_keeps_old_import():
    assert UR5ePlumbersBlockAtomicSkillAdapter is UR5eAssemblyAtomicSkillAdapter
    phases = UR5eAssemblySkillAPI.compile_plan(
        [
            {
                'skill': 'move_above_part',
                'robot': 'franka_left',
                'object': 'part_0',
                'offset': [0.0, 0.0, 0.12],
            },
            {
                'skill': 'preshape_gripper',
                'robot': 'franka_left',
                'object': 'part_0',
                'gripper_openness': 0.65,
            },
            {
                'skill': 'move_part_to_target',
                'robot': 'franka_left',
                'object': 'part_0',
                'target_object_target': 'part_0_target',
                'timeout_steps': 720,
            },
            {
                'skill': 'move_arm_to_joint_positions',
                'robot': 'franka_left',
                'joint_positions': [0.0, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7],
            },
        ]
    )

    assert [phase['local_skill']['name'] for phase in phases] == [
        'ur5e_move_above_part',
        'ur5e_preshape_gripper',
        'ur5e_move_part_to_staging',
        'move_arm_to_joint_positions',
    ]
    assert phases[2]['timeout_steps'] == 720
    assert phases[2]['advance']['type'] == 'local_skill_complete'
    assert phases[0]['local_skill']['cartesian_servo'] is True
    assert phases[0]['local_skill']['guard_ik_branch_jump'] is True
    assert phases[0]['local_skill']['ik_branch_jump_limit'] == 0.45
    assert phases[0]['local_skill']['ik_branch_jump_reference_mode'] == 'reference'
    assert phases[0]['local_skill']['allow_initial_ik_branch_jump'] is False
    assert phases[0]['local_skill']['max_command_tracking_error'] == 0.18
    assert phases[0]['local_skill']['cartesian_position_step'] == 0.015
    assert phases[0]['local_skill']['cartesian_orientation_step'] == 0.030
    assert phases[0]['local_skill']['default_max_command_joint_step'] == 0.060
    assert phases[0]['local_skill']['default_max_command_wrist_joint_step'] == 0.040
    assert phases[1]['local_skill']['gripper_openness'] == 0.65
    assert phases[2]['local_skill']['default_max_command_joint_step'] == 0.060
    assert phases[2]['local_skill']['default_max_command_wrist_joint_step'] == 0.040
    assert phases[2]['local_skill']['servo_target_object_pose'] is True
    assert phases[3]['local_skill']['cartesian_servo'] is False
    assert phases[3]['local_skill']['joint_target_stable_steps'] == 8


def test_cartesian_skill_uses_kinematics_pose_instead_of_physical_gripper_frame():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(
        _get_robot_kinematics_pose=lambda _name: (
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
    )
    tracked_robots = {
        'franka_right': {
            'position': [0.1, 0.2, 0.3],
            'orientation': [0.0, 1.0, 0.0, 0.0],
        }
    }

    pose = adapter._current_robot_pose(
        task=task,
        robot_name='franka_right',
        tracked_robots=tracked_robots,
    )

    np.testing.assert_allclose(pose['position'], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose['orientation'], [1.0, 0.0, 0.0, 0.0])


def test_preshape_gripper_requires_both_joint_and_object_stability(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(
        phase_index=1,
        phase_entry_step=0,
        phase_step_counter=0,
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    monkeypatch.setattr(adapter, '_hold_joint_action', lambda **_kwargs: {})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_kwargs: 0.28)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_kwargs: (0.0, 0.8))
    spec = {
        'name': 'ur5e_preshape_gripper',
        'object': 'part_3',
        'gripper_openness': 0.65,
        'stable_steps': 2,
        'max_object_linear_speed': 0.01,
        'max_object_angular_speed': 0.1,
    }

    for step in range(2):
        task.phase_step_counter = step
        action = adapter.act(
            task=task,
            robot_name='franka_left',
            phase_spec={},
            skill_spec=spec,
            tracked_robots={},
            tracked_objects={'part_3': {'linear_speed': 0.0, 'angular_speed': 0.0}},
        )

    assert action['gripper_controller'] == [0.65]
    assert completed[-1]['skill_name'] == 'ur5e_preshape_gripper'


def test_preshape_accepts_a_static_safe_aperture_when_the_target_stalls(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(
        phase_index=1,
        phase_entry_step=0,
        phase_step_counter=0,
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    monkeypatch.setattr(adapter, '_hold_joint_action', lambda **_kwargs: {})
    # Requested openness is 0.564 (q=0.349), but a static q=0.394 still
    # leaves openness 0.508, above the grasp-safe aperture 0.444.
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_kwargs: 0.394)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_kwargs: (0.0, 0.8))

    adapter.act(
        task=task,
        robot_name='franka_left',
        phase_spec={},
        skill_spec={
            'name': 'ur5e_preshape_gripper',
            'object': 'part_6',
            'gripper_openness': 0.564,
            'minimum_gripper_openness': 0.444,
            'stable_steps': 1,
        },
        tracked_robots={},
        tracked_objects={'part_6': {'linear_speed': 0.0, 'angular_speed': 0.0}},
    )

    assert completed[-1]['skill_name'] == 'ur5e_preshape_gripper'
    assert completed[-1]['detail']['target_gripper_ready'] is False
    assert completed[-1]['detail']['minimum_opening_ready'] is True


def test_object_relative_hover_retracts_along_the_grasp_axis():
    adapter = UR5eAssemblyAtomicSkillAdapter({})

    target = adapter._target_pose(
        phase_key=('test',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'grasp_relative_position': [0.0, 0.0, 0.1],
            'grasp_relative_orientation': [1.0, 0.0, 0.0, 0.0],
            'approach_clearance': 0.05,
        },
        tracked_robots={},
        tracked_objects={
            'part': {
                'position': [0.0, 0.0, 0.0],
                'orientation': [1.0, 0.0, 0.0, 0.0],
            }
        },
    )

    np.testing.assert_allclose(target['position'], [0.0, 0.0, -0.15])


def test_preshape_accepts_pose_stability_when_physx_velocity_is_noisy(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(
        phase_index=1,
        phase_entry_step=0,
        phase_step_counter=0,
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    monkeypatch.setattr(adapter, '_hold_joint_action', lambda **_kwargs: {})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_kwargs: 0.28)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_kwargs: (0.0, 0.8))

    adapter.act(
        task=task,
        robot_name='franka_left',
        phase_spec={},
        skill_spec={
            'name': 'ur5e_preshape_gripper',
            'object': 'part',
            'gripper_openness': 0.65,
            'stable_steps': 1,
        },
        tracked_robots={},
        tracked_objects={
            'part': {
                'linear_speed': 0.04,
                'angular_speed': 4.0,
                'pose_stable_override': True,
            }
        },
    )

    assert completed[-1]['detail']['motion_ready'] is True


def test_measured_joint_tracking_limit_bounds_arm_and_wrist_commands():
    current_q = np.zeros(6, dtype=float)
    command_q = np.ones(6, dtype=float)

    limited = UR5eAssemblyAtomicSkillAdapter._limit_command_to_measured_state(
        current_q=current_q,
        command_q=command_q,
        spec={
            'max_command_tracking_error': 0.18,
            'max_wrist_command_tracking_error': 0.12,
        },
    )

    assert np.max(np.abs(limited[:3] - current_q[:3])) <= 0.18
    assert np.max(np.abs(limited[3:] - current_q[3:])) <= 0.12
    np.testing.assert_allclose(limited, 0.12)


def test_cartesian_completion_only_relaxes_position_tolerance_after_delay():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(
        phase_step_counter=599,
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    target_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    current_pose = {
        'position': np.asarray([0.010, 0.0, 0.0], dtype=float),
        'orientation': target_pose['orientation'].copy(),
    }
    kwargs = {
        'task': task,
        'robot_name': 'franka_left',
        'skill_name': 'ur5e_descend_to_grasp',
        'spec': {
            'position_tolerance': 0.007,
            'relaxed_position_tolerance': 0.018,
            'relaxed_position_tolerance_after_steps': 600,
            'orientation_tolerance': 0.10,
        },
        'target_pose': target_pose,
        'ik_target_pose': target_pose,
        'current_pose': current_pose,
        'tracked_objects': {},
        'current_q': None,
        'target_q': None,
    }

    adapter._maybe_mark_complete(**kwargs)
    assert completed == []

    task.phase_step_counter = 600
    adapter._maybe_mark_complete(**kwargs)
    assert completed[-1]['detail']['relaxed_position_tolerance_active'] is True
    assert completed[-1]['detail']['position_tolerance'] == 0.018


def test_cartesian_completion_relaxes_carried_object_tolerance_after_delay():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    target_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    task = SimpleNamespace(
        phase_step_counter=1199,
        target_poses={'insert_target': target_pose},
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    kwargs = {
        'task': task,
        'robot_name': 'franka_left',
        'skill_name': 'ur5e_move_part_to_staging',
        'spec': {
            'object': 'part',
            'target_object_target': 'insert_target',
            'position_tolerance': 0.006,
            'relaxed_position_tolerance': 0.012,
            'relaxed_target_object_position_tolerance': 0.012,
            'relaxed_position_tolerance_after_steps': 1200,
            'orientation_tolerance': 0.08,
            'require_target_object_pose_convergence': True,
            'target_object_position_tolerance': 0.008,
            'target_object_orientation_tolerance': 0.10,
        },
        'target_pose': target_pose,
        'ik_target_pose': target_pose,
        'current_pose': {
            'position': np.asarray([0.010, 0.0, 0.0], dtype=float),
            'orientation': target_pose['orientation'].copy(),
        },
        'tracked_objects': {
            'part': {
                'position': [0.010, 0.0, 0.0],
                'orientation': target_pose['orientation'].tolist(),
            }
        },
        'current_q': None,
        'target_q': None,
    }

    adapter._maybe_mark_complete(**kwargs)
    assert completed == []

    task.phase_step_counter = 1200
    adapter._maybe_mark_complete(**kwargs)
    detail = completed[-1]['detail']
    assert detail['position_tolerance'] == 0.012
    assert detail['target_object_position_tolerance'] == 0.012
    assert detail['target_object_pose_complete'] is True


def test_cartesian_completion_accepts_consistent_pose_stability_with_noisy_velocity():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    target_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    task = SimpleNamespace(
        phase_index=4,
        phase_entry_step=100,
        phase_step_counter=20,
        target_poses={'insert_target': target_pose},
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    object_state = {
        'position': target_pose['position'].tolist(),
        'orientation': target_pose['orientation'].tolist(),
        'linear_speed': 0.04,
        'angular_speed': 0.1,
        'is_static': False,
        'pose_stable_override': True,
    }
    kwargs = {
        'task': task,
        'robot_name': 'franka_left',
        'skill_name': 'ur5e_move_part_to_staging',
        'spec': {
            'object': 'part',
            'target_object_target': 'insert_target',
            'position_tolerance': 0.006,
            'orientation_tolerance': 0.08,
            'require_target_object_pose_convergence': True,
            'target_object_position_tolerance': 0.008,
            'target_object_orientation_tolerance': 0.10,
            'require_target_object_static': True,
            'target_object_max_linear_speed': 0.03,
            'target_object_max_angular_speed': 2.0,
            'target_object_stable_steps': 8,
        },
        'target_pose': target_pose,
        'ik_target_pose': target_pose,
        'current_pose': target_pose,
        'tracked_objects': {'part': object_state},
        'current_q': None,
        'target_q': None,
    }

    adapter._maybe_mark_complete(**kwargs)
    assert completed == []

    object_state['is_static'] = True
    for _ in range(7):
        adapter._maybe_mark_complete(**kwargs)
    assert completed == []

    adapter._maybe_mark_complete(**kwargs)
    detail = completed[-1]['detail']
    assert detail['target_object_motion_ready'] is True
    assert detail['target_object_pose_stable_override_used'] is True
    assert detail['target_object_stable_steps'] == 8


def test_cartesian_completion_strictly_captures_final_target_at_phase_entry():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    target_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    task = SimpleNamespace(
        phase_index=4,
        phase_entry_step=100,
        phase_step_counter=1,
        target_poses={'final_target': target_pose},
        mark_local_skill_complete=lambda **payload: completed.append(payload),
    )
    adapter._maybe_mark_complete(
        phase_key=('final-entry',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec={
            'object': 'part',
            'target_object_target': 'final_target',
            'position_tolerance': 0.006,
            'orientation_tolerance': 0.08,
            'require_target_object_pose_convergence': True,
            'target_object_position_tolerance': 0.008,
            'target_object_orientation_tolerance': 0.10,
            'require_target_object_static': True,
            'target_object_max_linear_speed': 0.03,
            'target_object_max_angular_speed': 2.0,
            'target_object_stable_steps': 8,
            'target_object_entry_capture_max_steps': 4,
        },
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=target_pose,
        tracked_objects={
            'part': {
                'position': target_pose['position'].tolist(),
                'orientation': target_pose['orientation'].tolist(),
                'linear_speed': 0.01,
                'angular_speed': 0.2,
                'is_static': False,
                'pose_stable_override': False,
            }
        },
        current_q=None,
        target_q=None,
    )

    detail = completed[-1]['detail']
    assert detail['target_object_entry_capture_active'] is True
    assert detail['configured_target_object_stable_steps'] == 8
    assert detail['required_target_object_stable_steps'] == 1


def test_cartesian_skill_retries_ik_from_measured_joints(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_index=1, phase_entry_step=0, phase_step_counter=0)
    pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    measured_q = np.zeros(6, dtype=float)
    command_q = np.full(6, 0.1, dtype=float)
    warm_starts = []

    monkeypatch.setattr(adapter, '_target_pose', lambda **_: pose)
    monkeypatch.setattr(adapter, '_locked_target_pose', lambda **kwargs: kwargs['target_pose'])
    monkeypatch.setattr(adapter, '_prealign_action', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_robot_pose', lambda **_: pose)
    monkeypatch.setattr(adapter, '_current_tcp_pose', lambda **kwargs: kwargs['current_pose'])
    monkeypatch.setattr(adapter, '_object_tcp_slip_failure', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_: measured_q)
    monkeypatch.setattr(
        adapter,
        '_command_reference_q',
        lambda **_: np.full(6, 0.2, dtype=float),
    )

    def solve_ik(**kwargs):
        warm_start = np.asarray(kwargs['warm_start'], dtype=float)
        warm_starts.append(warm_start.copy())
        return command_q if np.array_equal(warm_start, measured_q) else None

    monkeypatch.setattr(adapter, '_solve_ik', solve_ik)
    monkeypatch.setattr(adapter, '_continuous_command_q', lambda **kwargs: kwargs['command_q'])
    monkeypatch.setattr(adapter, '_remember_arm_command', lambda *_: None)
    monkeypatch.setattr(adapter, '_maybe_mark_complete', lambda **_: None)

    action = adapter.act(
        task=task,
        robot_name='franka_left',
        phase_spec={},
        skill_spec={
            'name': 'ur5e_descend_to_grasp',
            'cartesian_servo': True,
            'guard_ik_branch_jump': False,
            'max_joint_step': 1.0,
        },
        tracked_robots={},
        tracked_objects={},
    )

    np.testing.assert_allclose(warm_starts[0], 0.2)
    np.testing.assert_allclose(warm_starts[1], measured_q)
    np.testing.assert_allclose(action['arm_joint_controller'][0], 0.12)


def test_cartesian_ik_tolerances_are_smaller_than_servo_steps():
    position_tolerance, orientation_tolerance = UR5eAssemblyAtomicSkillAdapter._ik_solver_tolerances(
        {
            'cartesian_servo': True,
            'cartesian_position_step': 0.0025,
            'cartesian_orientation_step': 0.012,
        }
    )

    assert 0.0 < position_tolerance < 0.0025
    assert 0.0 < orientation_tolerance < 0.012
    assert np.isclose(position_tolerance, 0.000125)
    assert np.isclose(orientation_tolerance, 0.0012)

    assert UR5eAssemblyAtomicSkillAdapter._ik_solver_tolerances(
        {
            'cartesian_servo': True,
            'cartesian_position_step': 0.0025,
            'cartesian_orientation_step': 0.012,
            'ik_position_tolerance': 0.0001,
            'ik_orientation_tolerance': 0.0003,
        }
    ) == (0.0001, 0.0003)
    assert UR5eAssemblyAtomicSkillAdapter._ik_solver_tolerances({}) == (None, None)


def test_hybrid_ik_reference_falls_back_when_command_tracking_lags():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(
        _ur5e_plumbers_last_arm_command_q={
            'franka_left': np.full(6, 0.20, dtype=float),
        }
    )
    measured_q = np.zeros(6, dtype=float)

    bounded = adapter._command_reference_q(
        task=task,
        robot_name='franka_left',
        current_q=measured_q,
        spec={
            'ik_reference_mode': 'hybrid',
            'ik_reference_command_max_tracking_error': 0.12,
        },
    )
    continuous = adapter._command_reference_q(
        task=task,
        robot_name='franka_left',
        current_q=measured_q,
        spec={
            'ik_reference_mode': 'hybrid',
            'ik_reference_command_max_tracking_error': 0.24,
        },
    )

    np.testing.assert_allclose(bounded, measured_q)
    np.testing.assert_allclose(continuous, 0.20)


def test_cartesian_orientation_command_warm_start_accumulates_with_bounded_lookahead():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = (1, 2, 3, 'franka_left', 'ur5e_move_above_part')
    current_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    target_pose = {
        'position': np.asarray([1.0, 0.0, 0.0], dtype=float),
        'orientation': np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float),
    }
    spec = {
        'cartesian_orientation_command_warm_start': True,
        'cartesian_orientation_command_lookahead': 0.36,
    }

    first = adapter._cartesian_command_servo_target_pose(
        phase_key=phase_key,
        current_pose=current_pose,
        target_pose=target_pose,
        max_position_step=0.01,
        max_orientation_step=0.03,
        spec=spec,
    )
    adapter._remember_cartesian_command_orientation(
        phase_key=phase_key,
        command_target_pose=first,
    )
    second = adapter._cartesian_command_servo_target_pose(
        phase_key=phase_key,
        current_pose=current_pose,
        target_pose=target_pose,
        max_position_step=0.01,
        max_orientation_step=0.03,
        spec=spec,
    )

    first_angle = 2.0 * np.arccos(np.clip(abs(first['orientation'][0]), 0.0, 1.0))
    second_angle = 2.0 * np.arccos(np.clip(abs(second['orientation'][0]), 0.0, 1.0))
    assert np.isclose(first_angle, 0.03)
    assert np.isclose(second_angle, 0.06)

    adapter._cartesian_command_orientations[phase_key] = target_pose['orientation'].copy()
    bounded = adapter._cartesian_command_servo_target_pose(
        phase_key=phase_key,
        current_pose=current_pose,
        target_pose=target_pose,
        max_position_step=0.01,
        max_orientation_step=0.03,
        spec=spec,
    )
    bounded_angle = 2.0 * np.arccos(np.clip(abs(bounded['orientation'][0]), 0.0, 1.0))
    assert bounded_angle <= 0.36 + 1e-9


def test_cartesian_skill_tolerates_short_ik_failure_streak(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_index=1, phase_entry_step=0, phase_step_counter=0)
    pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }

    monkeypatch.setattr(adapter, '_target_pose', lambda **_: pose)
    monkeypatch.setattr(adapter, '_locked_target_pose', lambda **kwargs: kwargs['target_pose'])
    monkeypatch.setattr(adapter, '_prealign_action', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_robot_pose', lambda **_: pose)
    monkeypatch.setattr(adapter, '_current_tcp_pose', lambda **kwargs: kwargs['current_pose'])
    monkeypatch.setattr(adapter, '_object_tcp_slip_failure', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_: np.zeros(6, dtype=float))
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_: np.zeros(6, dtype=float))
    monkeypatch.setattr(adapter, '_solve_ik', lambda **_: None)
    monkeypatch.setattr(
        adapter,
        '_hold_joint_action',
        lambda **_: {'arm_joint_controller': [[0.0] * 6]},
    )

    kwargs = {
        'task': task,
        'robot_name': 'franka_left',
        'phase_spec': {},
        'skill_spec': {
            'name': 'ur5e_descend_to_grasp',
            'cartesian_servo': True,
            'require_success': True,
            'ik_failure_tolerance_steps': 1,
        },
        'tracked_robots': {},
        'tracked_objects': {},
    }
    first = adapter.act(**kwargs)
    second = adapter.act(**kwargs)

    assert '__local_skill_failure__' not in first
    assert second['__local_skill_failure__'] is True
    assert second['diagnostics']['consecutive_failures'] == 2


def test_cartesian_skill_backtracks_an_unresolved_ik_step(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_index=1, phase_entry_step=0, phase_step_counter=0)
    current_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    target_pose = {
        'position': np.asarray([1.0, 0.0, 0.0], dtype=float),
        'orientation': current_pose['orientation'],
    }
    solved_positions = []

    monkeypatch.setattr(adapter, '_target_pose', lambda **_: target_pose)
    monkeypatch.setattr(adapter, '_locked_target_pose', lambda **kwargs: kwargs['target_pose'])
    monkeypatch.setattr(adapter, '_prealign_action', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_robot_pose', lambda **_: current_pose)
    monkeypatch.setattr(adapter, '_current_tcp_pose', lambda **kwargs: kwargs['current_pose'])
    monkeypatch.setattr(adapter, '_object_tcp_slip_failure', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_: np.zeros(6, dtype=float))
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_: np.zeros(6, dtype=float))

    def solve_ik(**kwargs):
        position = np.asarray(kwargs['target_pose']['position'], dtype=float)
        solved_positions.append(position.copy())
        return np.full(6, 0.01, dtype=float) if position[0] < 0.004 else None

    monkeypatch.setattr(adapter, '_solve_ik', solve_ik)
    monkeypatch.setattr(adapter, '_continuous_command_q', lambda **kwargs: kwargs['command_q'])
    monkeypatch.setattr(adapter, '_remember_arm_command', lambda *_: None)
    monkeypatch.setattr(adapter, '_maybe_mark_complete', lambda **_: None)

    action = adapter.act(
        task=task,
        robot_name='franka_left',
        phase_spec={},
        skill_spec={
            'name': 'ur5e_descend_to_grasp',
            'cartesian_servo': True,
            'cartesian_position_step': 0.004,
            'guard_ik_branch_jump': False,
        },
        tracked_robots={},
        tracked_objects={},
    )

    np.testing.assert_allclose(solved_positions[0], [0.004, 0.0, 0.0])
    np.testing.assert_allclose(solved_positions[1], [0.002, 0.0, 0.0])
    assert 'arm_joint_controller' in action


def test_move_above_recovers_branch_jump_with_translation_only_step(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_index=1, phase_entry_step=0, phase_step_counter=0)
    current_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    target_pose = {
        'position': np.asarray([1.0, 0.0, 0.0], dtype=float),
        'orientation': np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
    }
    solved_orientations = []

    monkeypatch.setattr(adapter, '_target_pose', lambda **_: target_pose)
    monkeypatch.setattr(adapter, '_locked_target_pose', lambda **kwargs: kwargs['target_pose'])
    monkeypatch.setattr(adapter, '_prealign_action', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_robot_pose', lambda **_: current_pose)
    monkeypatch.setattr(adapter, '_current_tcp_pose', lambda **kwargs: kwargs['current_pose'])
    monkeypatch.setattr(adapter, '_object_tcp_slip_failure', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_: np.zeros(6, dtype=float))
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_: np.zeros(6, dtype=float))

    def solve_ik(**kwargs):
        orientation = np.asarray(kwargs['target_pose']['orientation'], dtype=float)
        solved_orientations.append(orientation.copy())
        if np.allclose(orientation, current_pose['orientation']):
            return np.full(6, 0.05, dtype=float)
        return np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    monkeypatch.setattr(adapter, '_solve_ik', solve_ik)
    monkeypatch.setattr(adapter, '_continuous_command_q', lambda **kwargs: kwargs['command_q'])
    monkeypatch.setattr(adapter, '_remember_arm_command', lambda *_: None)
    monkeypatch.setattr(adapter, '_maybe_mark_complete', lambda **_: None)

    action = adapter.act(
        task=task,
        robot_name='franka_left',
        phase_spec={},
        skill_spec={
            'name': 'ur5e_move_above_part',
            'cartesian_servo': True,
            'cartesian_position_step': 0.1,
            'cartesian_orientation_step': 0.1,
            'guard_ik_branch_jump': True,
            'ik_branch_jump_limit': 0.3,
            'max_joint_step': 1.0,
        },
        tracked_robots={},
        tracked_objects={},
    )

    assert np.allclose(solved_orientations[-1], current_pose['orientation'])
    np.testing.assert_allclose(action['arm_joint_controller'][0], 0.025)
    assert np.all(np.asarray(action['arm_joint_controller'][0]) > 0.0)
    assert (
        adapter._last_targets[next(iter(adapter._last_targets))]['ik_branch_recovery_mode']
        == 'translation_only_command_warm_start'
    )


def test_persistent_branch_jump_fails_after_tolerance(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_index=1, phase_entry_step=0, phase_step_counter=0)
    pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }

    monkeypatch.setattr(adapter, '_target_pose', lambda **_: pose)
    monkeypatch.setattr(adapter, '_locked_target_pose', lambda **kwargs: kwargs['target_pose'])
    monkeypatch.setattr(adapter, '_prealign_action', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_robot_pose', lambda **_: pose)
    monkeypatch.setattr(adapter, '_current_tcp_pose', lambda **kwargs: kwargs['current_pose'])
    monkeypatch.setattr(adapter, '_object_tcp_slip_failure', lambda **_: None)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_: np.zeros(6, dtype=float))
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_: np.zeros(6, dtype=float))
    monkeypatch.setattr(
        adapter,
        '_solve_ik',
        lambda **_: np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(
        adapter,
        '_hold_joint_action',
        lambda **_: {'arm_joint_controller': [[0.0] * 6]},
    )

    kwargs = {
        'task': task,
        'robot_name': 'franka_left',
        'phase_spec': {},
        'skill_spec': {
            'name': 'ur5e_descend_to_grasp',
            'cartesian_servo': True,
            'guard_ik_branch_jump': True,
            'ik_branch_jump_limit': 0.3,
            'ik_branch_jump_tolerance_steps': 1,
            'require_success': True,
        },
        'tracked_robots': {},
        'tracked_objects': {},
    }
    first = adapter.act(**kwargs)
    second = adapter.act(**kwargs)

    assert '__local_skill_failure__' not in first
    assert second['__local_skill_failure__'] is True
    assert second['reason'] == 'ik_branch_jump_guard'
    assert second['diagnostics']['consecutive_failures'] == 2


def test_previous_ik_target_keeps_wrist_continuous_across_pi_boundary():
    previous_target_q = np.asarray([1.8, -1.6, 1.7, -1.7, -1.6, 2.99])
    wrapped_ik_q = np.asarray([1.75, -1.59, 1.72, -1.73, -1.63, -2.91])
    spec = {
        'unwrap_revolute_joints': True,
        'ik_branch_jump_limit': 0.45,
    }

    continuous_q = UR5eAssemblyAtomicSkillAdapter._joint_target_near_reference(
        target_q=wrapped_ik_q,
        reference_q=previous_target_q,
        spec=spec,
        preferred_abs_limit=3.05,
        hard_preferred_abs_limit=True,
    )

    assert continuous_q[-1] > np.pi
    assert abs(continuous_q[-1] - previous_target_q[-1]) < 0.45
    assert not UR5eAssemblyAtomicSkillAdapter._ik_branch_jump_detected(
        reference_q=previous_target_q,
        target_q=continuous_q,
        spec=spec,
    )


def test_attached_hold_defers_static_check_until_release():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    phases = {phase['name']: phase for phase in recipe['phases']}

    hold_advance = phases['right_hold_block_2_end_for_left_insert']['advance']
    assert hold_advance == {
        'type': 'local_skill_complete',
        'robot': 'franka_right',
        'skill': 'ur5e_hold_part_end',
        'min_steps': 120,
    }
    release_advance = phases['right_release_and_lock_block_2_on_table']['advance']
    assert release_advance['type'] == 'all_of'
    assert release_advance['conditions'][0]['type'] == 'objects_static'


def test_light_insert_parts_allow_pose_stability_to_filter_noisy_physx_velocity():
    recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    phases = {phase['name']: phase for phase in recipe['phases']}

    for part_index in (3, 4, 1):
        close_phase = phases[f'left_close_gripper_on_block_{part_index}']
        close_skill = close_phase['local_skill']
        assert close_skill['require_strict_physical_contact'] is True
        assert close_skill['require_dual_finger_contact'] is True
        assert close_skill['close_contact_stable_steps'] == 8
        assert close_skill['close_contact_motion_stable_steps'] == 8
        assert close_skill['close_contact_allow_pose_stable_override'] is True
        assert close_skill['physical_attach_surface_gap'] <= close_skill['finger_contact_distance']
        for attach_spec in close_phase['attach']:
            assert attach_spec['physical_attach_surface_gap'] <= attach_spec['finger_contact_distance']

    part_3_approach_phase = phases['left_move_above_block_3']
    part_3_preshape_phase = phases['left_preshape_gripper_for_block_3']
    part_3_descend_phase = phases['left_descend_to_block_3_grasp']
    part_3_descend = part_3_descend_phase['local_skill']
    part_3_close = phases['left_close_gripper_on_block_3']['local_skill']
    grasp_openness = part_3_preshape_phase['local_skill']['gripper_openness']
    assert part_3_approach_phase['gripper_commands']['franka_left'] == 0.58
    assert part_3_approach_phase['local_skill']['gripper_command'] == 0.58
    assert part_3_approach_phase['local_skill']['gripper_command'] < grasp_openness
    assert part_3_preshape_phase['gripper_commands']['franka_left'] == grasp_openness
    assert part_3_descend_phase['gripper_commands']['franka_left'] == grasp_openness
    assert part_3_descend['gripper_command'] == grasp_openness
    assert part_3_close['preclose_openness'] == grasp_openness
    assert 'target_pose_target' not in part_3_close
    assert part_3_close['grasp_relative_position'] == part_3_descend['grasp_relative_position']
    assert part_3_close['grasp_relative_orientation'] == part_3_descend['grasp_relative_orientation']
    assert part_3_approach_phase['local_skill']['approach_clearance'] == 0.10
    assert part_3_approach_phase['local_skill']['approach_clearance'] > part_3_descend['approach_clearance']
    assert part_3_descend['approach_clearance'] == 0.027
    assert part_3_close['approach_clearance'] == part_3_descend['approach_clearance']

    for part_index in (4, 1):
        approach = phases[f'left_move_above_block_{part_index}']['local_skill']
        descend = phases[f'left_descend_to_block_{part_index}_grasp']['local_skill']
        assert approach['offset_frame'] == 'world'
        assert approach['offset'][2] == 0.10
        assert descend['offset_frame'] == 'world'
        assert descend['offset'][2] == 0.0


def test_close_pose_gate_uses_local_cartesian_target_and_measured_joint_limits():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_step_counter=0)
    current_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    target_pose = {
        'position': np.asarray([1.0, 0.0, 0.0], dtype=float),
        'orientation': np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float),
    }
    solved_targets = []

    adapter._target_pose = lambda **_: target_pose
    adapter._locked_target_pose = lambda **kwargs: kwargs['target_pose']
    adapter._current_robot_pose = lambda **_: current_pose
    adapter._current_tcp_pose = lambda **kwargs: kwargs['current_pose']
    adapter._current_arm_q = lambda *_: np.zeros(6, dtype=float)
    adapter._command_reference_q = lambda **_: np.zeros(6, dtype=float)
    adapter._ik_target_pose = lambda **kwargs: kwargs['target_pose']

    def solve_ik(**kwargs):
        solved_targets.append(kwargs['target_pose'])
        return np.ones(6, dtype=float)

    adapter._solve_ik = solve_ik
    adapter._unwrap_to_reference = lambda **kwargs: kwargs['target_q']
    adapter._continuous_command_q = lambda **kwargs: kwargs['command_q']
    adapter._remember_arm_command = lambda *_: None

    ready, action, detail = adapter._close_pose_gate_action(
        phase_key=('close',),
        task=task,
        robot_name='franka_right',
        spec={
            'close_gate_guard_ik_branch_jump': False,
            'close_gate_cartesian_position_step': 0.004,
            'close_gate_cartesian_orientation_step': 0.015,
            'close_gate_max_joint_step': 1.0,
            'max_command_tracking_error': 0.18,
            'max_wrist_command_tracking_error': 0.12,
        },
        tracked_robots={},
        tracked_objects={},
    )

    assert ready is False
    np.testing.assert_allclose(solved_targets[0]['position'], [0.004, 0.0, 0.0])
    command_orientation = np.asarray(solved_targets[0]['orientation'], dtype=float)
    orientation_step = 2.0 * np.arccos(np.clip(abs(command_orientation[0]), 0.0, 1.0))
    assert orientation_step <= 0.015 + 1e-9
    np.testing.assert_allclose(action['arm_joint_controller'][0], 0.12)
    np.testing.assert_allclose(detail['target_position'], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(detail['command_target_position'], [0.004, 0.0, 0.0])


def test_close_pose_gate_holds_final_bounded_servo_command_while_closing():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_step_counter=7)
    pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    phase_key = ('close',)

    adapter._target_pose = lambda **_: pose
    adapter._locked_target_pose = lambda **kwargs: kwargs['target_pose']
    adapter._current_robot_pose = lambda **_: pose
    adapter._current_tcp_pose = lambda **kwargs: kwargs['current_pose']
    adapter._current_arm_q = lambda *_: np.zeros(6, dtype=float)
    adapter._command_reference_q = lambda **_: np.zeros(6, dtype=float)
    adapter._ik_target_pose = lambda **kwargs: kwargs['target_pose']
    adapter._solve_ik = lambda **_: np.ones(6, dtype=float)
    adapter._unwrap_to_reference = lambda **kwargs: kwargs['target_q']
    adapter._continuous_command_q = lambda **kwargs: kwargs['command_q']
    adapter._remember_arm_command = lambda *_: None

    ready, action, _ = adapter._close_pose_gate_action(
        phase_key=phase_key,
        task=task,
        robot_name='franka_left',
        spec={
            'close_ready_stable_steps': 1,
            'close_gate_guard_ik_branch_jump': False,
            'close_gate_hold_refined_command': True,
            'close_gate_max_joint_step': 1.0,
            'max_command_tracking_error': 0.18,
            'max_wrist_command_tracking_error': 0.12,
        },
        tracked_robots={},
        tracked_objects={},
    )

    assert ready is True
    command_q = np.asarray(action['arm_joint_controller'][0], dtype=float)
    np.testing.assert_allclose(adapter._close_gate_state[phase_key]['hold_q'], command_q)
    assert np.any(command_q != 0.0)


def test_close_pose_gate_tracks_a_part_that_moves_while_closing():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_step_counter=7)
    pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    phase_key = ('moving-close',)
    solved_q = [np.ones(6, dtype=float), np.full(6, 2.0, dtype=float)]

    adapter._target_pose = lambda **_: pose
    adapter._locked_target_pose = lambda **kwargs: kwargs['target_pose']
    adapter._current_robot_pose = lambda **_: pose
    adapter._current_tcp_pose = lambda **kwargs: kwargs['current_pose']
    adapter._current_arm_q = lambda *_: np.zeros(6, dtype=float)
    adapter._command_reference_q = lambda **_: np.zeros(6, dtype=float)
    adapter._ik_target_pose = lambda **kwargs: kwargs['target_pose']
    adapter._solve_ik = lambda **_: solved_q.pop(0)
    adapter._unwrap_to_reference = lambda **kwargs: kwargs['target_q']
    adapter._continuous_command_q = lambda **kwargs: kwargs['command_q']
    adapter._remember_arm_command = lambda *_: None

    spec = {
        'close_ready_stable_steps': 1,
        'close_gate_guard_ik_branch_jump': False,
        'close_gate_hold_refined_command': True,
        'close_gate_track_object_during_close': True,
        'close_gate_lock_terminal_ik_target': False,
        'close_gate_max_joint_step': 3.0,
        'limit_command_to_measured_state': False,
    }
    adapter._close_pose_gate_action(
        phase_key=phase_key,
        task=task,
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects={},
    )
    first_hold = adapter._close_gate_state[phase_key]['hold_q'].copy()
    task.phase_step_counter += 1
    adapter._close_pose_gate_action(
        phase_key=phase_key,
        task=task,
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects={},
    )

    np.testing.assert_allclose(first_hold, 1.0)
    np.testing.assert_allclose(adapter._close_gate_state[phase_key]['hold_q'], 2.0)


def test_close_gate_recenters_toward_a_single_contacting_finger():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    state = {}
    close_detail = {
        'contact_detail': {
            'contact_metrics': {
                'contact_box_orientation': [1.0, 0.0, 0.0, 0.0],
                'left_finger': {
                    'surface_gap': 0.0002,
                    'local_contact': {
                        'best_axis': 'x',
                        'best_surface_gap': 0.0002,
                        'local_point': [-0.0175, 0.0, 0.0],
                    },
                },
                'right_finger': {
                    'surface_gap': 0.025,
                    'local_contact': {
                        'best_axis': 'x',
                        'best_surface_gap': 0.025,
                        'local_point': [0.0425, 0.0, 0.0],
                    },
                },
            }
        }
    }
    spec = {
        'close_gate_recenter_single_finger_contact': True,
        'close_gate_recenter_stable_steps': 2,
        'close_gate_recenter_step': 0.00075,
        'close_gate_recenter_max_offset': 0.025,
        'finger_contact_distance': 0.008,
    }

    first = adapter._update_close_recenter_offset(
        state=state,
        close_detail=close_detail,
        spec=spec,
        close_ready=False,
    )
    second = adapter._update_close_recenter_offset(
        state=state,
        close_detail=close_detail,
        spec=spec,
        close_ready=False,
    )

    assert first['updated'] is False
    assert second['updated'] is True
    assert second['single_finger_side'] == 'left'
    assert second['axis'] == 'x'
    np.testing.assert_allclose(state['recenter_offset_world'], [-0.00075, 0.0, 0.0])
    third = adapter._update_close_recenter_offset(
        state=state,
        close_detail=close_detail,
        spec=spec,
        close_ready=False,
    )
    assert third['updated'] is False
    assert third['reason'] == 'previous_offset_servo_in_progress'
    np.testing.assert_allclose(state['recenter_offset_world'], [-0.00075, 0.0, 0.0])


def test_close_pose_gate_applies_accumulated_recenter_offset():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_step_counter=7)
    phase_key = ('recenter-close',)
    pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    current_pose = {
        'position': np.asarray([0.25, 0.0, 0.0], dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    adapter._close_gate_state[phase_key] = {
        'ready_steps': 0,
        'recenter_offset_world': np.asarray([-0.01, 0.002, 0.0], dtype=float),
    }
    adapter._phase_locks[phase_key] = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    adapter._target_pose = lambda **_: pose
    adapter._current_robot_pose = lambda **_: current_pose
    adapter._current_tcp_pose = lambda **kwargs: kwargs['current_pose']
    adapter._current_arm_q = lambda *_: np.zeros(6, dtype=float)
    adapter._command_reference_q = lambda **_: np.zeros(6, dtype=float)
    adapter._ik_target_pose = lambda **kwargs: kwargs['target_pose']
    adapter._solve_ik = lambda **_: np.zeros(6, dtype=float)
    adapter._unwrap_to_reference = lambda **kwargs: kwargs['target_q']
    adapter._continuous_command_q = lambda **kwargs: kwargs['command_q']
    adapter._remember_arm_command = lambda *_: None

    _, _, detail = adapter._close_pose_gate_action(
        phase_key=phase_key,
        task=task,
        robot_name='franka_left',
        spec={
            'close_gate_guard_ik_branch_jump': False,
            'close_gate_recenter_single_finger_contact': True,
            'lock_target_position': True,
            'lock_target_orientation': True,
            'close_gate_cartesian_position_step': 0.02,
            'close_gate_max_joint_step': 1.0,
            'limit_command_to_measured_state': False,
        },
        tracked_robots={},
        tracked_objects={},
    )

    np.testing.assert_allclose(detail['target_position'], [0.24, 0.002, 0.0])
    np.testing.assert_allclose(detail['recenter_offset_world'], [-0.01, 0.002, 0.0])
    pose['position'] = np.asarray([1.0, 0.0, 0.0], dtype=float)
    _, _, second_detail = adapter._close_pose_gate_action(
        phase_key=phase_key,
        task=task,
        robot_name='franka_left',
        spec={
            'close_gate_guard_ik_branch_jump': False,
            'close_gate_cartesian_position_step': 0.02,
            'close_gate_max_joint_step': 1.0,
            'limit_command_to_measured_state': False,
            'close_gate_recenter_single_finger_contact': True,
            'lock_target_position': True,
            'lock_target_orientation': True,
        },
        tracked_robots={},
        tracked_objects={},
    )
    np.testing.assert_allclose(second_detail['target_position'], [0.24, 0.002, 0.0])


def test_transport_uses_observed_fixed_attachment_pose_and_invalidates_regrasp():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace()
    object_pose = {
        'position': np.asarray([1.0, 2.0, 3.0], dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    tracked_robots = {
        'franka_left': {
            'position': [0.0, 0.0, 0.0],
            'orientation': [1.0, 0.0, 0.0, 0.0],
        }
    }
    tracked_objects = {
        'part': {
            'position': object_pose['position'].tolist(),
            'orientation': object_pose['orientation'].tolist(),
            'attached_to': 'franka_left',
            'attachment': {
                'robot_name': 'franka_left',
                'mode': 'fixed_joint',
                'attach_step': 10,
                'joint_path': '/World/part/joint',
                'position': [0.1, 0.2, 0.3],
                'orientation': [1.0, 0.0, 0.0, 0.0],
            },
        }
    }

    first_position, first_orientation = adapter._object_tcp_relative_pose(
        phase_key=('transport',),
        task=task,
        robot_name='franka_left',
        object_name='part',
        spec={},
        tracked_robots=tracked_robots,
        object_pose=object_pose,
        tracked_objects=tracked_objects,
    )
    np.testing.assert_allclose(first_position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(first_orientation, [1.0, 0.0, 0.0, 0.0])

    tracked_objects['part']['attachment'].update(
        {
            'attach_step': 20,
            'position': [-0.3, -0.2, -0.1],
            'orientation': [0.0, 1.0, 0.0, 0.0],
        }
    )
    second_position, second_orientation = adapter._object_tcp_relative_pose(
        phase_key=('transport',),
        task=task,
        robot_name='franka_left',
        object_name='part',
        spec={},
        tracked_robots=tracked_robots,
        object_pose=object_pose,
        tracked_objects=tracked_objects,
    )
    np.testing.assert_allclose(second_position, [-0.3, -0.2, -0.1])
    np.testing.assert_allclose(second_orientation, [0.0, 1.0, 0.0, 0.0])


def test_transport_converts_physical_attachment_pose_to_control_frame():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    conversions = []
    task = SimpleNamespace(
        _physical_relative_pose_to_control_frame=lambda robot, position, orientation: (
            conversions.append((robot, np.asarray(position), np.asarray(orientation)))
            or (
                np.asarray([-0.1, -0.2, 0.3], dtype=float),
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float),
            )
        )
    )
    attachment = {
        'robot_name': 'franka_right',
        'mode': 'fixed_joint',
        'attach_step': 10,
        'joint_path': '/World/part/joint',
        'position': [0.1, 0.2, 0.3],
        'orientation': [1.0, 0.0, 0.0, 0.0],
    }

    position, orientation = adapter._object_tcp_relative_pose(
        phase_key=('transport',),
        task=task,
        robot_name='franka_right',
        object_name='part',
        spec={},
        tracked_robots={},
        object_pose={
            'position': np.asarray([1.0, 2.0, 3.0], dtype=float),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        },
        tracked_objects={
            'part': {
                'attached_to': 'franka_right',
                'attachment': attachment,
            }
        },
    )

    assert conversions[0][0] == 'franka_right'
    np.testing.assert_allclose(conversions[0][1], attachment['position'])
    np.testing.assert_allclose(position, [-0.1, -0.2, 0.3])
    np.testing.assert_allclose(orientation, [0.0, 0.0, 0.0, 1.0])


def test_close_pose_gate_timeout_allows_in_progress_stability_window(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_step_counter=240)
    gate_action = {'arm_joint_controller': [[0.0] * 6]}
    monkeypatch.setattr(adapter, '_hold_joint_action', lambda **_: {})
    monkeypatch.setattr(
        adapter,
        '_close_pose_gate_action',
        lambda **_: (
            False,
            gate_action,
            {'ready_steps': 3, 'required_ready_steps': 4},
        ),
    )
    monkeypatch.setattr(
        adapter,
        '_failure_or_hold',
        lambda *_, **__: {'failed': True},
    )
    kwargs = {
        'task': task,
        'robot_name': 'franka_left',
        'phase_spec': {},
        'skill_spec': {
            'name': 'ur5e_close_gripper',
            'require_close_pose_gate': True,
            'close_pose_gate_timeout_steps': 240,
        },
        'tracked_robots': {},
        'tracked_objects': {},
    }

    assert adapter.act(**kwargs) is gate_action
    task.phase_step_counter = 244
    assert adapter.act(**kwargs) == {'failed': True}


def test_target_object_servo_pivots_tcp_around_held_object():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current_pose = {
        'position': np.asarray([0.0, 0.0, 0.0], dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    target_orientation = np.asarray([2**-0.5, 0.0, 2**-0.5, 0.0], dtype=float)
    target_pose = {
        'position': np.asarray([-1.0, 0.0, 1.0], dtype=float),
        'orientation': target_orientation,
    }
    object_pose = {
        'position': np.asarray([0.0, 0.0, 1.0], dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    adapter._object_name_from_spec = lambda _: 'part'
    adapter._object_pose = lambda **_: object_pose
    adapter._target_object_pose = lambda **_: object_pose

    command_pose = adapter._target_object_servo_pose(
        phase_key=('transport',),
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'cartesian_position_step': 0.1,
            'cartesian_orientation_step': 2.0,
            'servo_target_object_pose': True,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    reconstructed_object_position = np.asarray(command_pose['position']) + np.asarray([1.0, 0.0, 0.0])
    np.testing.assert_allclose(command_pose['orientation'], target_orientation)
    np.testing.assert_allclose(reconstructed_object_position, object_pose['position'], atol=1e-7)


def _fake_task():
    camera_metadata = [
        {'name': 'front', 'owner': 'franka_left', 'view_type': 'front'},
        {'name': 'left_wrist', 'owner': 'franka_left', 'robot': 'franka_left', 'view_type': 'wrist'},
        {'name': 'right_wrist', 'owner': 'franka_right', 'robot': 'franka_right', 'view_type': 'wrist'},
    ]
    robot_config = SimpleNamespace(gripper_open_position=0.0, gripper_closed_position=0.8)
    articulation = SimpleNamespace(get_joint_positions=lambda: np.zeros(9, dtype=np.float32))
    return SimpleNamespace(
        step_counter=0,
        phase_index=0,
        phase_step_counter=0,
        phase='phase_0',
        robots={
            'franka_left': SimpleNamespace(config=robot_config, articulation=articulation),
            'franka_right': SimpleNamespace(config=robot_config, articulation=articulation),
        },
        config=SimpleNamespace(
            camera_metadata=camera_metadata,
            seed=3,
            recipe='test_recipe',
            recipe_fingerprint='test-fingerprint',
            task_description='test task',
            prompt='test task',
            scene_profile='taoyuan_grscenes_tabletop',
            scene_asset_path='/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd',
            scene_asset_fallback_path='/benchmark/scenes/usd/factory_cell.usda',
            scene_asset_source='primary',
            resolved_scene_family='isaac_simple_warehouse_tabletop',
            scene_profile_metadata={'scene_family': 'isaac_simple_warehouse_tabletop'},
            domain_randomization={'enabled': True},
        ),
    )


def _fake_observation(position_x: float):
    frame = np.full((12, 16, 4), 127, dtype=np.uint8)
    base = {
        'eef_position': np.asarray([position_x, 0.0, 1.0], dtype=np.float32),
        'eef_orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        'controllers': {'gripper_controller': {'gripper_pos': [0.0]}},
    }
    left = {
        **base,
        'sensors': {
            'front': {'rgba': frame},
            'left_wrist': {'rgba': frame},
        },
    }
    right = {
        **base,
        'sensors': {'right_wrist': {'rgba': frame}},
    }
    return {'franka_left': left, 'franka_right': right}


def test_compact_cartesian_recorder_aligns_next_sample_actions_and_three_videos(tmp_path):
    task = _fake_task()
    recorder = CompactCartesianEpisodeRecorder(
        output_dir=tmp_path,
        episode_idx=0,
        task=task,
        fps=30,
        frame_stride=2,
        output_resolution=(16, 12),
    )
    actions = {
        'franka_left': {'gripper_controller': [1.0]},
        'franka_right': {'gripper_controller': [1.0]},
    }
    assert recorder.record(task=task, obs=_fake_observation(0.1), actions=actions)
    task.step_counter = 1
    assert not recorder.record(task=task, obs=_fake_observation(0.2), actions=actions)
    task.step_counter = 2
    assert recorder.record(task=task, obs=_fake_observation(0.3), actions=actions)
    output = recorder.finalize(task=task, metrics={'success': True})

    trajectory = np.load(output['trajectory_path'])
    assert trajectory['observation_state'].shape == (2, 16)
    assert trajectory['action'].shape == (2, 16)
    assert trajectory['action'][0, 0] == trajectory['observation_state'][1, 0]
    assert set(output['videos']) == {
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.right_wrist',
    }
    metadata = json.loads(Path(output['metadata_path']).read_text(encoding='utf-8'))
    assert set(tuple(shape) for shape in metadata['video_shapes'].values()) == {(12, 16, 3)}
    assert metadata['timing']['rendering_interval'] == 1
    assert metadata['timing']['camera_render_period_steps'] == 2
    assert metadata['timing']['camera_state_action_aligned']
    assert metadata['scene_profile'] == 'taoyuan_grscenes_tabletop'
    assert metadata['scene_asset_path'].endswith('/warehouse_with_forklifts.usd')
    assert metadata['scene_asset_fallback_path'].endswith('/factory_cell.usda')
    assert metadata['scene_asset_source'] == 'primary'
    assert metadata['scene_family'] == 'isaac_simple_warehouse_tabletop'
    for video_path in output['videos'].values():
        capture = cv2.VideoCapture(video_path)
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        capture.release()


def test_policy_evaluation_success_detector_requires_consecutive_stable_steps():
    task = object.__new__(FactoryDualFrankaAssemblyTask)
    task.step_counter = 0
    task.max_steps = 100
    task.success = False
    task.failed = False
    task.policy_evaluation_mode = True
    task._policy_success_stable_count = 0
    task._policy_success_stable_steps = 3
    task._check_success = lambda: True

    def set_terminal_state(_phase, *, reason, status, detail):
        task.success = status == 'success'
        task.failed = status == 'failed'
        task.terminal_reason = reason
        task.terminal_detail = detail

    task._set_terminal_state = set_terminal_state
    assert not FactoryDualFrankaAssemblyTask.is_done(task)
    assert not FactoryDualFrankaAssemblyTask.is_done(task)
    assert FactoryDualFrankaAssemblyTask.is_done(task)
    assert task.success is True
    assert task.terminal_reason == 'policy-success-detector-stable'


def test_policy_evaluation_metrics_do_not_bypass_stable_success_gate():
    task = object.__new__(FactoryDualFrankaAssemblyTask)
    task.policy_evaluation_mode = True
    task.success = False
    task.failed = True
    task.config = type(
        'TaskConfig',
        (),
        {'recipe': 'recipe', 'seed': 1, 'layout_seed': 12, 'episode_idx': 0},
    )()
    task.phase_status = 'failed'
    task.phase_history = []
    task.phase_transition_history = []
    task.phase_attempts = {}
    task.step_counter = 100
    task.phase_timeout_count = 0
    task.phase_recovery_count = 0
    task._handoff_history = []
    task._recovery_history = []
    task.terminal_reason = 'policy-evaluation-max-steps'
    task._policy_success_stable_count = 1
    task._policy_success_stable_steps = 24
    task._policy_interaction_history = []
    task._check_success = lambda: True
    task.get_tracked_object_states = lambda: {}
    task.get_tracked_robot_states = lambda: {}
    task._success_diagnostics = lambda: []

    metrics = FactoryDualFrankaAssemblyTask.calculate_metrics(task)

    assert metrics['success'] is False
    assert metrics['success_detector']['passed'] is False
    assert metrics['success_detector']['criteria_passed'] is True
