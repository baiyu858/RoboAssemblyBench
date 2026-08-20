from pathlib import Path

import numpy as np
import pytest

from roboassemblybench.core.domain_randomization import apply_domain_randomization
from roboassemblybench.core.fabrica_canonical import (
    compile_fabrica_canonical_recipe,
    load_fabrica_canonical_metadata,
)
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe

TASKS = {
    'beam': 5,
    'car': 6,
    'cooling_manifold': 7,
    'duct': 8,
    'gamepad': 6,
    'plumbers_block': 5,
    'stool_circular': 9,
}
STAGED_TASKS = tuple(TASKS)


def _object_map(recipe):
    return {entry['name']: entry for entry in recipe['objects']}


def _target_map(recipe):
    return {entry['name']: entry for entry in recipe['targets']}


def _quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_rotate(quaternion, vector):
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    conjugate = quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
    rotated = _quat_multiply(
        _quat_multiply(quaternion, [0.0, *vector]),
        conjugate,
    )
    return rotated[1:]


def _target_tcp_position(target, grasp):
    relative_orientation = np.asarray(grasp['object_in_tcp_orientation'], dtype=float)
    relative_orientation = relative_orientation / np.linalg.norm(relative_orientation)
    relative_conjugate = relative_orientation * np.asarray([1.0, -1.0, -1.0, -1.0])
    tcp_orientation = _quat_multiply(target['orientation'], relative_conjugate)
    return np.asarray(target['position']) - _quat_rotate(
        tcp_orientation,
        grasp['object_in_tcp_position'],
    )


def test_canonical_metadata_covers_all_bundles_and_uses_runtime_safe_assets():
    metadata = load_fabrica_canonical_metadata()

    assert set(metadata['tasks']) == set(TASKS)
    for task_name, part_count in TASKS.items():
        task = metadata['tasks'][task_name]
        assert len(task['parts']) == part_count
        assert len(task['assembly_steps']) == part_count - 1
        assert task['base_part'] not in {step['move_part'] for step in task['assembly_steps']}
        assert [step['move_part'] for step in task['assembly_steps']] == [
            step['move_part'] for step in reversed(task['disassembly_steps'])
        ]
        base_grasp_candidates = task['base_grasp_candidates']
        assert base_grasp_candidates
        assert [item['grasp_id'] for item in base_grasp_candidates] == sorted(
            item['grasp_id'] for item in base_grasp_candidates
        )
        planner_base_grasp_id = base_grasp_candidates[0]['planner_grasp_id']
        assert any(item['grasp_id'] == planner_base_grasp_id for item in base_grasp_candidates)
        for base_grasp in base_grasp_candidates:
            assert base_grasp['panda_compatible'] is True
            assert isinstance(base_grasp['robotiq_compatible'], bool)
            if base_grasp['robotiq_compatible']:
                assert base_grasp['target_gripper'] == 'robotiq-85'
                assert base_grasp['target_gripper_asset'] == 'isaac_official_robotiq_2f85'
                assert base_grasp['gripper_frame_conversion'] == ('fabrica_minus_x_to_isaac_plus_y')
                assert np.allclose(
                    base_grasp['gripper_frame_rotation_wxyz'],
                    [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                )
            assert base_grasp['selection_method'] == ('compiler_joint_pickup_yaw_base_grasp_selection')
            assert base_grasp['interior_clearance_minimum'] == 0.20
            assert (
                base_grasp['interior_clearance_score'] >= 0.20
                or base_grasp['is_planner_grasp']
            )
            assert base_grasp['interior_clearance_planner_exemption'] == (
                base_grasp['is_planner_grasp']
                and base_grasp['interior_clearance_score'] < 0.20
            )
            assert base_grasp['valid_candidate_count'] == len(base_grasp_candidates)
            assert len(base_grasp['assembly_approach_direction']) == 3
        for step in task['assembly_steps']:
            move_grasp_candidates = step['move_grasp_candidates']
            assert move_grasp_candidates
            assert [item['grasp_id'] for item in move_grasp_candidates] == sorted(
                item['grasp_id'] for item in move_grasp_candidates
            )
            assert sum(item['is_planner_grasp'] for item in move_grasp_candidates) == 1
            planner_grasp = next(item for item in move_grasp_candidates if item['is_planner_grasp'])
            assert planner_grasp['grasp_id'] == step['move_grasp']['grasp_id']
            for candidate in move_grasp_candidates:
                assert candidate['selection_method'] == ('compiler_move_grasp_candidate_conversion')
                assert candidate['panda_compatible'] is True
                assert isinstance(candidate['robotiq_compatible'], bool)
                if candidate['robotiq_compatible']:
                    assert candidate['target_gripper'] == 'robotiq-85'
                assert candidate['valid_candidate_count'] == len(move_grasp_candidates)
                assert candidate['grasp_lever_arm_m'] > 0.0
                assert candidate['source_collision_count'] >= 0
                assert len(candidate['grasp_center_m']) == 3
        for asset_path in [
            task['fixture']['usd_path'],
            task['optical_board']['usd_path'],
            *(part['usd_path'] for part in task['parts']),
        ]:
            path = Path(__file__).resolve().parents[2] / asset_path
            assert path.is_file(), asset_path


def test_duct_uses_physical_socket_relations_instead_of_optimizer_hold_relations():
    task = load_fabrica_canonical_metadata()['tasks']['duct']
    steps = {step['move_part']: step for step in task['assembly_steps']}

    assert steps['7']['socket_part'] == '1'
    assert steps['7']['optimizer_hold_part'] == '0'
    assert steps['2']['socket_part'] == '1'
    assert steps['2']['optimizer_hold_part'] == '0'


def test_move_grasp_selection_rejects_an_ill_conditioned_full_pose_ik_path():
    recipe = load_task_recipe(
        'fabrica_car_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    diagnostics = recipe['fabrica_canonical_resolved']['move_grasp_selection']['3']

    assert diagnostics['selected']['ik_feasible'] is True
    assert diagnostics['selected']['ik_maximum_position_error'] <= 0.01
    assert diagnostics['selected']['ik_maximum_orientation_error'] <= 0.03
    assert diagnostics['selected']['ik_minimum_path_manipulability'] >= 0.08
    assert diagnostics['selected']['pickup_orientation_continuity'] >= diagnostics['required_orientation_continuity']


def test_canonical_compiler_rejects_lateral_step_larger_than_final_tolerance():
    recipe = load_task_recipe(
        'fabrica_beam_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    recipe['fabrica_canonical']['insertion_lateral_alignment_cartesian_position_step'] = 0.002

    with pytest.raises(
        ValueError,
        match='no larger than the final insertion lateral tolerance',
    ):
        compile_fabrica_canonical_recipe(recipe)


def test_staged_recipes_compile_complete_contact_gated_skill_sequences():
    reference_recipe = load_task_recipe(
        'fabrica_plumbers_block_ur5e_right_base_prepare',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    reference_robots = {entry['name']: entry for entry in reference_recipe['robots']}
    reference_cameras = {entry['name']: entry for entry in reference_recipe['camera_specs']}
    reference_workcell_objects = {
        entry['name']: entry
        for entry in reference_recipe['objects']
        if entry['name'].startswith(('factory_', 'taoyuan_'))
    }
    for task_name in STAGED_TASKS:
        recipe = load_task_recipe(
            f'fabrica_{task_name}_ur5e_staged',
            scene_profile='taoyuan_grscenes_tabletop',
        )
        resolved = recipe['fabrica_canonical_resolved']
        objects = _object_map(recipe)
        targets = _target_map(recipe)
        robots = {entry['name']: entry for entry in recipe['robots']}
        part_names = {f'fabrica_{task_name}_{part_id}' for part_id in range(20)}
        actual_parts = set(objects).intersection(part_names)

        assert recipe['scene_asset_path'] == reference_recipe['scene_asset_path']
        assert recipe['scene_asset_path'].endswith('/warehouse_with_forklifts.usd')
        assert recipe['scene_asset_fallback_path'] == reference_recipe['scene_asset_fallback_path']
        assert recipe['metadata']['scene_family'] == 'isaac_simple_warehouse_tabletop'
        assert recipe['domain_randomization']['appearance']['allowed_objects'] == [
            'factory_tabletop_visual',
            'factory_background_visual',
            'factory_floor_visual',
        ]
        assert recipe['domain_randomization']['appearance']['allowed_lights'] == ['warehouse_dome_fill']
        assert recipe['domain_randomization']['visual_distractors']['count_range'] == [0, 8]
        assert {
            entry['name']: entry for entry in recipe['objects'] if entry['name'].startswith(('factory_', 'taoyuan_'))
        } == reference_workcell_objects
        assert not any(entry['name'] == 'factory_backdrop_visual' for entry in recipe['objects'])
        for robot_name, robot in robots.items():
            reference_robot = reference_robots[robot_name]
            for field in (
                'type',
                'usd_path',
                'prim_path',
                'position',
                'orientation',
                'end_effector_prim_name',
                'gripper_xform_orient',
                'author_gripper_collision_pads',
                'initial_joint_positions',
            ):
                assert robot[field] == reference_robot[field]
        cameras = {entry['name']: entry for entry in recipe['camera_specs']}
        for camera_name, camera in cameras.items():
            reference_camera = reference_cameras[camera_name]
            for field in (
                'prim_path',
                'translation',
                'orientation_euler',
                'resolution',
            ):
                assert camera[field] == reference_camera[field]

        assert resolved['optical_board_position_randomized'] is False
        assert np.isclose(
            resolved['insertion_cartesian_position_step'],
            0.00025,
        )
        assert np.isclose(
            resolved['insertion_lateral_alignment_cartesian_position_step'],
            0.00025,
        )
        assert np.isclose(
            resolved['insertion_lateral_tolerance_object_extent_scale'],
            0.04,
        )
        assert np.isclose(
            resolved['insertion_lateral_alignment_entry_clearance'],
            0.01,
        )
        assert np.isclose(
            resolved['insertion_lateral_alignment_clearance_object_extent_scale'],
            1.0,
        )
        assert np.isclose(
            resolved['insertion_axial_recovery_cartesian_position_step'],
            0.001,
        )
        assert np.isclose(resolved['insertion_axial_recovery_deadband'], 0.0005)
        assert np.isclose(
            resolved['insertion_compliance_capture_max_linear_speed'],
            0.10,
        )
        assert np.isclose(
            resolved['insertion_compliance_capture_max_angular_speed'],
            2.0,
        )
        assert resolved['insertion_compliance_capture_stable_steps'] == 8
        assert resolved['insertion_compliance_geometric_capture_after_steps'] == 1200
        assert np.isclose(
            resolved['insertion_compliance_minimum_gravity_alignment'],
            0.70,
        )
        assert np.isclose(
            resolved['insertion_compliant_alignment_retraction_limit'],
            0.006,
        )
        assert resolved['insertion_compliant_track_object_orientation'] is True
        assert np.isclose(
            resolved['intermediate_insertion_lateral_position_tolerance'],
            0.002,
        )
        assert np.isclose(
            resolved['intermediate_insertion_lateral_alignment_cartesian_position_step'],
            0.001,
        )
        assert np.isclose(resolved['base_support_release_position_tolerance'], 0.012)
        assert np.isclose(resolved['base_support_lateral_position_tolerance'], 0.015)
        assert np.isclose(
            resolved['base_support_lateral_alignment_enter_tolerance'],
            0.002,
        )
        assert np.isclose(
            resolved['base_support_lateral_alignment_exit_tolerance'],
            0.004,
        )
        assert np.isclose(
            resolved['base_support_lateral_alignment_cartesian_position_step'],
            0.002,
        )
        assert np.isclose(resolved['release_retreat_distance'], 0.06)
        assert np.isclose(resolved['post_release_park_distance'], 0.35)
        assert np.isclose(resolved['post_release_park_vertical_offset'], 0.02)
        assert np.isclose(
            resolved['post_release_park_minimum_planar_radius'],
            0.28,
        )
        assert resolved['base_place_timeout_steps'] == 4800
        selection = resolved['pickup_layout_selection']
        assert np.isclose(selection['move_grasp_ik_position_tolerance'], 0.01)
        assert np.isclose(selection['move_grasp_ik_orientation_tolerance'], 0.03)
        assert selection['move_grasp_ik_max_iterations'] == 200
        assert np.isclose(selection['move_grasp_ik_minimum_manipulability'], 0.08)
        assert np.isclose(
            selection['move_grasp_minimum_relative_orientation_continuity'],
            0.90,
        )
        assert np.isclose(selection['move_grasp_fixture_footprint_margin'], 0.035)
        assert np.isclose(selection['move_grasp_minimum_fixture_clearance'], 0.001)
        selected_base_grasp = resolved['selected_base_grasp']
        assert selection['pickup_yaw_degrees'] in {0.0, 90.0, -90.0, 180.0}
        assert selection['base_grasp_id'] == selected_base_grasp['grasp_id']
        assert selection['vertical_clearance_score'] >= 0.70
        assert selection['interior_clearance_score'] >= 0.20
        assert selection['support_clearance_ratio'] >= 0.70
        assert np.isclose(
            selected_base_grasp['support_clearance_ratio'],
            selection['support_clearance_ratio'],
        )
        assert selection['orientation_continuity_score'] >= 0.50
        assert selection['maximum_pickup_tcp_reach'] <= 0.82
        assert selection['fixture_on_optical_board'] is True
        assert len(selection['orientation_continuity_by_part']) == TASKS[task_name]
        assert len(selection['pickup_tcp_reach_by_part']) == TASKS[task_name]
        selected_move_grasps = resolved['selected_move_grasps']
        move_grasp_selection = resolved['move_grasp_selection']
        assert len(selected_move_grasps) == TASKS[task_name] - 1
        assert set(selected_move_grasps) == set(move_grasp_selection)
        for part_id, selected_move_grasp in selected_move_grasps.items():
            diagnostics = move_grasp_selection[part_id]
            assert selected_move_grasp['selection_method'] == ('runtime_physical_move_grasp_selection')
            assert diagnostics['selected_grasp_id'] == selected_move_grasp['grasp_id']
            assert diagnostics['feasible_candidate_count'] > 0
            assert diagnostics['source_collision_feasible_candidate_count'] > 0
            assert diagnostics['fixture_feasible_candidate_count'] > 0
            assert diagnostics['ik_evaluated_candidate_count'] > 0
            assert diagnostics['ik_feasible_candidate_count'] > 0
            assert selected_move_grasp['ik_feasible'] is True
            assert selected_move_grasp['ik_maximum_position_error'] <= 0.01
            assert selected_move_grasp['ik_maximum_orientation_error'] <= 0.03
            assert selected_move_grasp['ik_minimum_path_manipulability'] >= 0.08
            assert (
                selected_move_grasp['pickup_orientation_continuity'] >= diagnostics['required_orientation_continuity']
                or selected_move_grasp['is_planner_grasp']
            )
            assert set(selected_move_grasp['ik_errors_by_target']) >= {
                'pickup_approach',
                'pickup',
                'lift',
                'assembly_clearance',
                'insertion_final',
            }
            assert selected_move_grasp['pickup_orientation_continuity'] >= 0.50
            assert selected_move_grasp['maximum_tcp_reach'] <= 0.82
            assert selected_move_grasp['source_collision_count'] == diagnostics['selected']['source_collision_count']
            assert selected_move_grasp['source_collision_count'] >= diagnostics['minimum_source_collision_count']
            assert selected_move_grasp['pickup_fixture_body_clearance'] >= diagnostics['required_fixture_clearance']
            assert selected_move_grasp['insertion_body_clearance'] >= 0.0
            assert (
                selected_move_grasp['interior_clearance_score'] >= diagnostics['required_interior_clearance']
                or selected_move_grasp['is_planner_grasp']
            )
        if task_name == 'car':
            assert selected_move_grasps['0']['is_planner_grasp'] is True
            assert selected_move_grasps['3']['is_planner_grasp'] is True
            assert move_grasp_selection['0']['selected']['pickup_fixture_body_clearance'] > 0.035
            assert move_grasp_selection['3']['selected']['pickup_fixture_body_clearance'] > 0.035
        assert selected_base_grasp['selection_method'] == ('joint_pickup_yaw_base_grasp_runtime_selection')
        np.testing.assert_allclose(
            objects['fabrica_fixture']['orientation'],
            resolved['selected_pickup_orientation'],
        )
        np.testing.assert_allclose(
            objects['fabrica_fixture']['position'],
            resolved['selected_pickup_origin'],
        )
        configured_pickup_pivot = np.asarray(resolved['configured_pickup_origin'], dtype=float) + _quat_rotate(
            resolved['configured_pickup_orientation'],
            selection['pickup_rotation_pivot'],
        )
        selected_pickup_pivot = np.asarray(resolved['selected_pickup_origin'], dtype=float) + _quat_rotate(
            resolved['selected_pickup_orientation'],
            selection['pickup_rotation_pivot'],
        )
        np.testing.assert_allclose(selected_pickup_pivot, configured_pickup_pivot)
        np.testing.assert_allclose(selection['pickup_layout_offset'], [0.0, 0.0, 0.0])
        assert len(actual_parts) == TASKS[task_name]
        assert set(recipe['domain_randomization']['fixed_objects']) >= {'optical_board'}
        assert all(
            'optical_board' not in (group.get('objects') or [])
            for group in recipe['domain_randomization']['groups'].values()
        )
        assert len(recipe['success']) == TASKS[task_name]
        assert recipe['max_steps'] >= len(recipe['phases']) * 360
        assert recipe['max_steps'] >= TASKS[task_name] * 7000
        base_part_id = recipe['fabrica_canonical_resolved']['base_part']
        base_release = next(
            phase for phase in recipe['phases'] if phase['name'] == f'base_{base_part_id}_release_and_lock'
        )
        assert base_release['detach'] == [
            {
                'object': f'fabrica_{recipe["fabrica_canonical_resolved"]["assembly"]}_{base_part_id}',
                'release_min_steps': 0,
            }
        ]
        rebased_targets = set(base_release['lock'][0]['rebase_targets'])
        assert {criterion['target'] for criterion in recipe['success']} <= rebased_targets
        assert rebased_targets == set(recipe['domain_randomization']['groups']['assembly_base']['targets'])
        base_retreat = next(phase for phase in recipe['phases'] if phase['name'] == f'base_{base_part_id}_retreat')
        base_park = next(phase for phase in recipe['phases'] if phase['name'] == f'base_{base_part_id}_park')
        np.testing.assert_allclose(
            base_retreat['local_skill']['offset'],
            [0.0, 0.0, 0.06],
        )
        assert base_retreat['local_skill']['lock_target_position'] is True
        assert base_retreat['local_skill']['lock_target_orientation'] is True
        park_offset = np.asarray(base_park['local_skill']['offset'], dtype=float)
        assert base_park['local_skill']['lock_target_position'] is True
        assert base_park['local_skill']['lock_target_orientation'] is False
        base_robot_name = recipe['fabrica_canonical']['base_robot']
        robot_to_assembly = (
            np.asarray(robots[base_robot_name]['position'], dtype=float)[:2]
            - np.asarray(recipe['fabrica_canonical']['assembly_origin'], dtype=float)[:2]
        )
        assert np.dot(park_offset[:2], robot_to_assembly) > 0.0
        assert np.isclose(np.linalg.norm(park_offset[:2]), 0.35)
        assert np.isclose(park_offset[2], 0.02)
        np.testing.assert_allclose(
            base_park['local_skill']['workspace_center'],
            robots[base_robot_name]['position'],
        )
        assert np.isclose(
            base_park['local_skill']['workspace_minimum_planar_radius'],
            0.28,
        )
        assert all(
            'rebase_targets' not in lock_spec
            for phase in recipe['phases']
            if phase is not base_release
            for lock_spec in phase.get('lock', [])
        )
        assert all(Path(objects[name]['usd_path']).is_file() for name in actual_parts)
        assert Path(objects['optical_board']['usd_path']).is_file()
        assert Path(objects['fabrica_fixture']['usd_path']).is_file()
        assert np.isclose(
            robots['franka_left']['initial_joint_positions']['shoulder_pan_joint'],
            0.0,
        )
        assert np.isclose(
            robots['franka_right']['initial_joint_positions']['shoulder_pan_joint'],
            0.0,
        )

        close_phases = [
            phase for phase in recipe['phases'] if (phase.get('local_skill') or {}).get('name') == 'ur5e_close_gripper'
        ]
        assert len(close_phases) == TASKS[task_name]
        assert resolved['stabilize_fixture_parts'] is True
        initial_locks = recipe['phases'][0]['lock']
        base_object = f'fabrica_{task_name}_{base_part_id}'
        assert {lock_spec['object'] for lock_spec in initial_locks} == actual_parts
        assert {lock_spec['target'] for lock_spec in initial_locks} == {
            f'part_{object_name.rsplit("_", 1)[-1]}_fixture_pickup' for object_name in actual_parts
        }
        for lock_spec in initial_locks:
            assert lock_spec['snap_free_object'] is True
            assert lock_spec['free_snap_steps'] == 0
            assert lock_spec['position_tolerance'] == 0.03
            assert lock_spec['orientation_tolerance'] == 0.20
            assert lock_spec['disable_collision_on_lock'] is False
            assert lock_spec['target'] in recipe['domain_randomization']['groups']['start_parts']['targets']
        for phase in close_phases:
            local_skill = phase['local_skill']
            attach = phase['attach'][0]
            assert 'unlock' not in phase
            assert 'unlock_after_steps' not in phase
            if local_skill['object'] == base_object:
                assert phase['fixture_lock'][0]['target'] == f'part_{base_part_id}_fixture_pickup'
                assert phase['fixture_lock'][0]['disable_collision_on_lock'] is False
            else:
                assert 'fixture_lock' not in phase
            assert local_skill['close_until_contact'] is True
            assert local_skill['close_position_tolerance'] == 0.007
            assert local_skill['close_gate_hold_refined_command'] is True
            assert local_skill['close_gate_track_object_during_close'] is True
            assert local_skill['close_gate_recenter_single_finger_contact'] is True
            assert local_skill['close_gate_recenter_stable_steps'] == 2
            assert local_skill['close_gate_recenter_step'] == pytest.approx(0.00075)
            assert local_skill['close_gate_recenter_max_offset'] == pytest.approx(0.025)
            assert local_skill['close_gate_recenter_target_tolerance'] == pytest.approx(0.00035)
            assert local_skill['require_grasp_contact'] is True
            assert local_skill['require_strict_physical_contact'] is True
            assert local_skill['allow_cross_axis_dual_finger_contact'] is True
            assert attach['require_contact'] is True
            assert attach['require_physical_contact'] is True
            assert attach['require_dual_finger_contact'] is True
            assert attach['allow_cross_axis_dual_finger_contact'] is True
            assert attach['require_local_skill_complete_for_attach'] is True
            assert attach['allow_strict_contact_target_refinement'] is True
            assert attach['strict_contact_target_refinement_max_distance'] == pytest.approx(0.025)
            assert attach['strict_contact_target_refinement_tracking_tolerance'] == pytest.approx(0.00035)
            assert attach['measure_force_contact'] is True
            assert local_skill['measure_force_contact'] is True
            assert attach['min_attach_steps'] == 24
            assert phase['advance']['min_steps'] == 24
            assert attach['allow_noncontact_fixed_joint'] is False
            assert attach['position_tolerance'] == 0.007
            assert attach['orientation_tolerance'] == 0.10
            assert attach['filter_gripper_collisions_on_attach'] is False
            assert attach['compliant_hold_linear_limit'] == 0.006
            assert attach['compliant_hold_angular_limit_degrees'] == 6.0
            assert attach['compliant_hold_linear_max_force'] == 20.0
            assert attach['compliant_hold_linear_damping'] == 10.0
            assert attach['compliant_hold_linear_stiffness'] == 500.0
            assert attach['compliant_hold_angular_max_force'] == 2.0
            assert attach['compliant_hold_angular_damping'] == 0.2
            assert attach['compliant_hold_angular_stiffness'] == 5.0
            assert attach['compliant_hold_gravity_force_multiplier'] == 6.0
            assert attach['compliant_hold_drive_damping_ratio'] == 1.0
            assert attach['compliant_hold_torque_force_fraction'] == 0.5
            assert attach['compliant_hold_linear_force_cap'] == 120.0
            assert attach['compliant_hold_angular_force_cap'] == 12.0
            assert 0.0 < attach['gripper_closed_threshold'] <= 0.8

        for phase in recipe['phases']:
            local_skill = phase.get('local_skill') or {}
            if local_skill.get('name') == 'ur5e_move_above_part':
                assert local_skill.get('requires_held_object', False) is False
                assert phase['timeout_steps'] == 3600
                assert local_skill['orientation_first_before_translation'] is False
                assert local_skill['orientation_first_max_steps'] == 720
                assert local_skill['orientation_first_tolerance'] == 0.08
                assert local_skill['ik_reference_mode'] == 'hybrid'
                assert local_skill['ik_reference_command_max_tracking_error'] == 0.12
                assert local_skill['cartesian_orientation_command_warm_start'] is True
                assert local_skill['cartesian_orientation_command_lookahead'] == 0.36
                assert local_skill['max_command_joint_step'] == 0.06
                assert local_skill['max_wrist_command_tracking_error'] == 0.24
                assert local_skill['prealign_steps'] == 240
                assert len(local_skill['prealign_joint_positions']) == 6
                assert local_skill['prealign_max_joint_step'] == 0.035
                assert 'prealign_shoulder_pan' not in local_skill
            if local_skill.get('name') == 'ur5e_preshape_gripper':
                assert phase['timeout_steps'] == 744
                assert local_skill['preshape_timeout_steps'] == 720
                assert local_skill['gripper_position_tolerance'] == 0.025
            if local_skill.get('name') == 'ur5e_descend_to_grasp':
                assert local_skill['position_tolerance'] == 0.007
                assert local_skill['relaxed_position_tolerance'] == 0.018
                assert local_skill['relaxed_position_tolerance_after_steps'] == 600
            if local_skill.get('name') == 'ur5e_move_part_to_staging':
                assert local_skill['requires_held_object'] is True
                assert local_skill['derive_tcp_orientation_from_target_object'] is True
                assert local_skill['ik_reference_mode'] == 'hybrid'
                assert local_skill['ik_reference_command_max_tracking_error'] == 0.12
                assert local_skill['cartesian_orientation_command_warm_start'] is True
                assert local_skill['cartesian_orientation_command_lookahead'] == 0.36
                assert local_skill['target_object_use_measured_orientation_for_position_servo'] is True
                if phase['name'].endswith('_transport_hover'):
                    assert local_skill['max_command_joint_step'] == 0.06
                    assert local_skill['max_command_tracking_error'] == 0.24
                    assert local_skill['max_wrist_command_tracking_error'] == 0.24
                    assert local_skill['position_tolerance'] == 0.006
                    assert local_skill['target_object_position_tolerance'] == 0.008
                    assert local_skill['orientation_tolerance'] == 0.10
                    assert 'relaxed_position_tolerance' not in local_skill
                if '_insert_' in phase['name']:
                    assert local_skill['cartesian_position_step'] == 0.00025
                    assert local_skill['target_object_servo_position_command_warm_start'] is True
                    assert local_skill['target_object_servo_position_command_gate_overdrive'] is True
                    assert local_skill['target_object_servo_position_command_lookahead'] == 0.004
                    assert (
                        local_skill['target_object_servo_position_command_accumulation_step']
                        == local_skill['cartesian_position_step']
                    )
                    assert local_skill['target_object_axial_recovery_cartesian_position_step'] == 0.001
                    assert local_skill['target_object_axial_recovery_deadband'] == 0.0005
                    assert local_skill['target_object_lateral_alignment_axial_clearance'] >= 0.0
                    assert local_skill['target_object_insertion_path_depth'] >= 0.0
                    expected_lateral_alignment_step = (
                        0.00025 if local_skill['target_object_target'].endswith('_assembled') else 0.001
                    )
                    assert (
                        local_skill['target_object_lateral_alignment_cartesian_position_step']
                        == expected_lateral_alignment_step
                    )
                    assert local_skill['max_command_joint_step'] == 0.035
                    assert local_skill['max_command_tracking_error'] == 0.12
                    assert local_skill['max_wrist_command_tracking_error'] == 0.10
                    assert local_skill['position_tolerance'] == 0.006
                    assert local_skill['target_object_position_tolerance'] == 0.008
                    is_final_insertion = local_skill['target_object_target'].endswith('_assembled')
                    assert local_skill['require_target_object_static'] is is_final_insertion
                    assert local_skill['hold_for_target_object_settle'] is is_final_insertion
                    assert local_skill['target_object_max_linear_speed'] == 0.03
                    assert local_skill['target_object_max_angular_speed'] == 2.0
                    assert local_skill['target_object_allow_pose_stable_override'] is True
                    assert local_skill['target_object_stable_steps'] == (8 if is_final_insertion else 1)
                    compliance_keys = {
                        'target_object_final_target',
                        'relax_fixed_attachment_within_final_position_tolerance',
                        'relax_fixed_attachment_after_steps',
                        'relax_fixed_attachment_require_waypoint_proximity',
                        'relax_fixed_attachment_waypoint_position_tolerance',
                        'relax_fixed_attachment_waypoint_axial_position_tolerance',
                        'relax_fixed_attachment_waypoint_lateral_position_tolerance',
                        'relax_fixed_attachment_geometric_capture_after_steps',
                        'relax_fixed_attachment_minimum_gravity_alignment',
                        'relax_fixed_attachment_final_orientation_tolerance',
                        'relax_fixed_attachment_max_linear_speed',
                        'relax_fixed_attachment_max_angular_speed',
                        'relax_fixed_attachment_stable_steps',
                        'compliant_servo_max_alignment_retraction',
                        'compliant_servo_track_object_orientation',
                    }
                    assert local_skill['target_object_final_target'].endswith('_assembled')
                    assert local_skill['relax_fixed_attachment_after_steps'] == 0
                    assert local_skill['relax_fixed_attachment_require_waypoint_proximity'] is True
                    assert local_skill['relax_fixed_attachment_waypoint_position_tolerance'] == 0.010
                    assert local_skill['relax_fixed_attachment_waypoint_axial_position_tolerance'] == 0.010
                    compliance_lateral_tolerance = local_skill[
                        'relax_fixed_attachment_waypoint_lateral_position_tolerance'
                    ]
                    assert (
                        local_skill['target_object_lateral_position_tolerance'] <= compliance_lateral_tolerance <= 0.002
                    )
                    assert local_skill['relax_fixed_attachment_geometric_capture_after_steps'] == 1200
                    assert np.isclose(
                        local_skill['relax_fixed_attachment_minimum_gravity_alignment'],
                        0.70,
                    )
                    if is_final_insertion:
                        assert 0.015 <= local_skill['relax_fixed_attachment_within_final_position_tolerance'] <= 0.060
                    else:
                        assert local_skill['relax_fixed_attachment_within_final_position_tolerance'] >= 0.015
                    assert compliance_keys.issubset(local_skill)
                    assert local_skill['relax_fixed_attachment_final_orientation_tolerance'] == 0.15
                    assert local_skill['relax_fixed_attachment_max_linear_speed'] == 0.10
                    assert local_skill['relax_fixed_attachment_max_angular_speed'] == 2.0
                    assert local_skill['relax_fixed_attachment_stable_steps'] == 8
                    assert local_skill['compliant_servo_max_alignment_retraction'] == 0.006
                    assert local_skill['compliant_servo_track_object_orientation'] is True
                    assert local_skill['target_object_settle_hold_steps'] == 48
                    assert local_skill['target_object_settle_retry_servo_steps'] == 8
                    assert np.isclose(
                        np.linalg.norm(local_skill['target_object_convergence_axis']),
                        1.0,
                    )
                    lateral_tolerance = local_skill['target_object_lateral_position_tolerance']
                    assert 0.001 <= lateral_tolerance <= 0.002
                    if not (
                        phase['name'].endswith('_insert_00')
                        or phase['name'].endswith('_insert_08')
                        or local_skill['target_object_target'].endswith('_assembled')
                    ):
                        assert lateral_tolerance == 0.002
                    if local_skill['target_object_target'].endswith('_assembled'):
                        assert local_skill['target_object_entry_capture_max_steps'] == 12
                    else:
                        assert 'target_object_entry_capture_max_steps' not in local_skill
                    expected_relaxed_tolerance = 0.015
                    assert local_skill['relaxed_position_tolerance'] == expected_relaxed_tolerance
                    assert local_skill['relaxed_target_object_position_tolerance'] == expected_relaxed_tolerance
                    assert local_skill['relaxed_position_tolerance_after_steps'] == (
                        600 if is_final_insertion else 0
                    )
            if phase['name'].endswith('_release_and_lock'):
                for lock_spec in phase.get('lock', []):
                    assert 0.015 <= lock_spec['position_tolerance'] <= 0.062

        for success in recipe['success']:
            assert success['target'] in targets
            assert success['require_released'] is True
            assert success['require_static'] is True

        if task_name == 'car':
            car_cover_final = next(
                phase['local_skill'] for phase in recipe['phases'] if phase['name'] == 'assemble_00_part_1_insert_09'
            )
            assert np.isclose(
                car_cover_final['target_object_lateral_position_tolerance'],
                0.002,
            )
            car_horizontal_insert = next(
                phase['local_skill'] for phase in recipe['phases'] if phase['name'] == 'assemble_01_part_3_insert_00'
            )
            assert (
                abs(car_horizontal_insert['target_object_convergence_axis'][2])
                < car_horizontal_insert['relax_fixed_attachment_minimum_gravity_alignment']
            )
            assert np.isclose(
                car_horizontal_insert['target_object_lateral_position_tolerance'],
                0.002,
            )


def test_staged_transport_uses_layout_aware_high_clearance_paths():
    metadata = load_fabrica_canonical_metadata()
    for task_name in STAGED_TASKS:
        recipe = load_task_recipe(
            f'fabrica_{task_name}_ur5e_staged',
            scene_profile='taoyuan_grscenes_tabletop',
        )
        task = metadata['tasks'][task_name]
        targets = _target_map(recipe)
        robots = {entry['name']: entry for entry in recipe['robots']}
        phases = {phase['name']: phase for phase in recipe['phases']}
        phase_order = {phase['name']: index for index, phase in enumerate(recipe['phases'])}
        pickup_group = set(recipe['domain_randomization']['groups']['start_parts']['targets'])
        assembly_group = set(recipe['domain_randomization']['groups']['assembly_base']['targets'])
        transport_tcp_height = recipe['fabrica_canonical_resolved']['transport_tcp_height']
        assert np.isclose(transport_tcp_height, 0.3525)
        assert recipe['fabrica_canonical_resolved']['transport_timeout_steps'] == 4800
        assert recipe['fabrica_canonical_resolved']['insertion_timeout_steps'] == 3600
        grasp_by_part = {
            str(task['base_part']): recipe['fabrica_canonical_resolved']['selected_base_grasp'],
            **recipe['fabrica_canonical_resolved']['selected_move_grasps'],
        }
        preshape_by_part = {
            phase['local_skill']['object'].rsplit('_', 1)[-1]: phase['local_skill']
            for phase in recipe['phases']
            if (phase.get('local_skill') or {}).get('name') == 'ur5e_preshape_gripper'
        }

        for part_id, grasp in grasp_by_part.items():
            pickup_target = f'part_{part_id}_pickup_clearance'
            assembly_target = f'part_{part_id}_assembly_clearance'
            assert pickup_target in pickup_group
            assert pickup_target not in assembly_group
            assert assembly_target in assembly_group
            assert assembly_target not in pickup_group
            np.testing.assert_allclose(
                _target_tcp_position(targets[pickup_target], grasp)[2],
                transport_tcp_height,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                _target_tcp_position(targets[assembly_target], grasp)[2],
                transport_tcp_height,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                preshape_by_part[part_id]['gripper_openness'],
                min(1.0, float(grasp['robotiq_open_ratio']) + 0.20),
            )

        base_part = str(task['base_part'])
        base_prefix = f'base_{base_part}'
        assert phase_order[f'{base_prefix}_lift'] < phase_order[f'{base_prefix}_pickup_clearance']
        assert phases[f'{base_prefix}_lift']['local_skill']['lock_target_position'] is True
        assert phases[f'{base_prefix}_lift']['local_skill']['lock_target_orientation'] is True
        assert phase_order[f'{base_prefix}_pickup_clearance'] < phase_order[f'{base_prefix}_assembly_clearance']
        assert phase_order[f'{base_prefix}_assembly_clearance'] < phase_order[f'{base_prefix}_transport_hover']
        assert phases[f'{base_prefix}_assembly_clearance']['timeout_steps'] == 4800
        assert phases[f'{base_prefix}_place']['timeout_steps'] == 4800
        assert phases[f'{base_prefix}_place']['local_skill']['relaxed_position_tolerance'] == 0.012
        assert phases[f'{base_prefix}_place']['local_skill']['relaxed_target_object_position_tolerance'] == 0.012
        np.testing.assert_allclose(
            phases[f'{base_prefix}_place']['local_skill']['target_object_convergence_axis'],
            [0.0, 0.0, -1.0],
        )
        assert phases[f'{base_prefix}_place']['local_skill']['target_object_lateral_position_tolerance'] == 0.015
        assert phases[f'{base_prefix}_place']['local_skill']['target_object_lateral_alignment_enter_tolerance'] == 0.002
        assert phases[f'{base_prefix}_place']['local_skill']['target_object_lateral_alignment_exit_tolerance'] == 0.004
        assert (
            phases[f'{base_prefix}_place']['local_skill']['target_object_lateral_alignment_cartesian_position_step']
            == 0.002
        )
        assert (
            phases[f'{base_prefix}_pickup_clearance']['local_skill']['target_object_target']
            == f'part_{base_part}_pickup_clearance'
        )
        assert (
            phase_order[f'{base_prefix}_release_and_lock']
            < phase_order[f'{base_prefix}_retreat']
            < phase_order[f'{base_prefix}_park']
        )

        for step_index, step in enumerate(task['assembly_steps']):
            part_id = str(step['move_part'])
            prefix = f'assemble_{step_index:02d}_part_{part_id}'
            assert phase_order[f'{prefix}_lift'] < phase_order[f'{prefix}_pickup_clearance']
            assert phase_order[f'{prefix}_pickup_clearance'] < phase_order[f'{prefix}_assembly_clearance']
            assert phase_order[f'{prefix}_assembly_clearance'] < phase_order[f'{prefix}_transport_hover']
            insertion_phases = [phase for phase in recipe['phases'] if phase['name'].startswith(f'{prefix}_insert_')]
            for insertion_index, insertion_phase in enumerate(insertion_phases):
                insertion_skill = insertion_phase['local_skill']
                lateral_tolerance = insertion_skill['target_object_lateral_position_tolerance']
                assert insertion_skill['target_object_lateral_alignment_enter_tolerance'] == lateral_tolerance
                assert insertion_skill['target_object_lateral_alignment_exit_tolerance'] == (
                    0.002 if insertion_index == len(insertion_phases) - 2 else lateral_tolerance
                )
                assert insertion_skill['target_object_lateral_alignment_stable_steps'] == (
                    8 if insertion_index == len(insertion_phases) - 1 else 1
                )
                assert insertion_skill['target_object_stable_steps'] == (
                    8 if insertion_index == len(insertion_phases) - 1 else 1
                )
                compliance_keys = {
                    'relax_fixed_attachment_stable_steps',
                    'relax_fixed_attachment_after_steps',
                    'relax_fixed_attachment_require_waypoint_proximity',
                    'relax_fixed_attachment_waypoint_position_tolerance',
                    'relax_fixed_attachment_minimum_gravity_alignment',
                    'relax_fixed_attachment_allow_pose_stable_override',
                    'compliant_servo_pause_linear_speed',
                    'compliant_servo_pause_angular_speed',
                    'compliant_servo_resume_linear_speed',
                    'compliant_servo_resume_angular_speed',
                    'compliant_servo_resume_stable_steps',
                    'compliant_servo_allow_pose_stable_resume',
                    'compliant_servo_velocity_rate_limit',
                    'compliant_servo_minimum_step_scale',
                    'compliant_servo_max_position_step',
                    'compliant_servo_max_lateral_step',
                    'compliant_servo_max_orientation_step',
                    'compliant_servo_orientation_correction_deadband',
                    'compliant_servo_hold_orientation_during_lateral_alignment',
                }
                assert compliance_keys.issubset(insertion_skill)
                assert insertion_skill['relax_fixed_attachment_stable_steps'] == 8
                assert insertion_skill['relax_fixed_attachment_after_steps'] == 0
                assert insertion_skill['relax_fixed_attachment_require_waypoint_proximity'] is True
                assert np.isclose(
                    insertion_skill['relax_fixed_attachment_minimum_gravity_alignment'],
                    0.70,
                )
                assert insertion_skill['relax_fixed_attachment_waypoint_position_tolerance'] == 0.010
                compliance_lateral_tolerance = insertion_skill[
                    'relax_fixed_attachment_waypoint_lateral_position_tolerance'
                ]
                assert (
                    insertion_skill['target_object_lateral_position_tolerance'] <= compliance_lateral_tolerance <= 0.002
                )
                assert insertion_skill['relax_fixed_attachment_allow_pose_stable_override'] is True
                assert insertion_skill['compliant_servo_pause_linear_speed'] == 0.20
                assert insertion_skill['compliant_servo_pause_angular_speed'] == 5.0
                assert insertion_skill['compliant_servo_resume_linear_speed'] == 0.03
                assert insertion_skill['compliant_servo_resume_angular_speed'] == 2.0
                assert insertion_skill['compliant_servo_resume_stable_steps'] == 8
                assert insertion_skill['compliant_servo_allow_pose_stable_resume'] is True
                assert insertion_skill['compliant_servo_velocity_rate_limit'] is True
                assert insertion_skill['compliant_servo_minimum_step_scale'] == 0.2
                assert insertion_skill['compliant_servo_max_position_step'] == 0.0005
                assert insertion_skill['compliant_servo_position_command_warm_start'] is True
                assert insertion_skill['compliant_servo_position_command_gate_overdrive'] is True
                assert insertion_skill['compliant_servo_position_command_accumulation_step'] == 0.0001
                assert insertion_skill['compliant_servo_position_command_lookahead'] == 0.004
                assert insertion_skill['compliant_servo_max_lateral_step'] == 0.0005
                assert insertion_skill['compliant_servo_max_orientation_step'] == 0.002
                assert insertion_skill['compliant_servo_hold_orientation_during_lateral_alignment'] is True
                assert insertion_skill['compliant_servo_orientation_correction_deadband'] == 0.005
                if insertion_index == len(insertion_phases) - 1:
                    assert 0.015 <= insertion_skill['relax_fixed_attachment_within_final_position_tolerance'] <= 0.060
                else:
                    assert insertion_skill['relax_fixed_attachment_within_final_position_tolerance'] >= 0.015
            final_insertion = max(
                insertion_phases,
                key=lambda phase: phase_order[phase['name']],
            )
            first_insertion = min(
                insertion_phases,
                key=lambda phase: phase_order[phase['name']],
            )
            assert first_insertion['local_skill']['target_object_lateral_alignment_axial_clearance'] > 0.01
            assert (
                final_insertion['local_skill']['target_object_lateral_alignment_axial_clearance']
                > first_insertion['local_skill']['target_object_lateral_alignment_axial_clearance']
            )
            insertion_axis = np.asarray(
                final_insertion['local_skill']['target_object_convergence_axis'],
                dtype=float,
            )
            insertion_path_depths = [
                phase['local_skill']['target_object_insertion_path_depth'] for phase in insertion_phases
            ]
            if abs(float(insertion_axis[2])) < 0.70:
                assert insertion_path_depths[-1] > insertion_path_depths[0]
            else:
                assert all(depth == 0.0 for depth in insertion_path_depths)
            retreat_offset = np.asarray(
                phases[f'{prefix}_retreat']['local_skill']['offset'],
                dtype=float,
            )
            np.testing.assert_allclose(np.linalg.norm(retreat_offset), 0.06)
            np.testing.assert_allclose(
                retreat_offset,
                -0.06 * insertion_axis,
            )
            assert phases[f'{prefix}_retreat']['local_skill']['lock_target_position'] is True
            assert phases[f'{prefix}_retreat']['local_skill']['lock_target_orientation'] is True
            assert (
                phase_order[f'{prefix}_release_and_lock']
                < phase_order[f'{prefix}_retreat']
                < phase_order[f'{prefix}_park']
            )
            park_offset = np.asarray(
                phases[f'{prefix}_park']['local_skill']['offset'],
                dtype=float,
            )
            assert phases[f'{prefix}_park']['local_skill']['lock_target_position'] is True
            assert phases[f'{prefix}_park']['local_skill']['lock_target_orientation'] is False
            assembly_robot = recipe['fabrica_canonical']['assembly_robot']
            robot_to_assembly = (
                np.asarray(robots[assembly_robot]['position'], dtype=float)[:2]
                - np.asarray(recipe['fabrica_canonical']['assembly_origin'], dtype=float)[:2]
            )
            assert np.dot(park_offset[:2], robot_to_assembly) > 0.0
            np.testing.assert_allclose(
                phases[f'{prefix}_park']['local_skill']['workspace_center'],
                robots[assembly_robot]['position'],
            )
            assert np.isclose(
                phases[f'{prefix}_park']['local_skill']['workspace_minimum_planar_radius'],
                0.28,
            )


def test_selected_pickup_layouts_fit_fixed_board_and_ur5e_reach_envelope():
    metadata = load_fabrica_canonical_metadata()
    for task_name in STAGED_TASKS:
        recipe = load_task_recipe(
            f'fabrica_{task_name}_ur5e_staged',
            scene_profile='taoyuan_grscenes_tabletop',
        )
        task = metadata['tasks'][task_name]
        resolved = recipe['fabrica_canonical_resolved']
        selection = resolved['pickup_layout_selection']
        objects = _object_map(recipe)
        robots = {entry['name']: entry for entry in recipe['robots']}

        board_position = np.asarray(objects['optical_board']['position'], dtype=float)
        board_min = board_position + np.asarray(task['optical_board']['bbox_min'])
        board_max = board_position + np.asarray(task['optical_board']['bbox_max'])
        fixture_position = np.asarray(objects['fabrica_fixture']['position'], dtype=float)
        fixture_orientation = np.asarray(
            objects['fabrica_fixture']['orientation'],
            dtype=float,
        )
        fixture_min = np.asarray(task['fixture']['bbox_min'], dtype=float)
        fixture_max = np.asarray(task['fixture']['bbox_max'], dtype=float)
        for x in (fixture_min[0], fixture_max[0]):
            for y in (fixture_min[1], fixture_max[1]):
                for z in (fixture_min[2], fixture_max[2]):
                    corner = fixture_position + _quat_rotate(
                        fixture_orientation,
                        [x, y, z],
                    )
                    assert np.all(corner[:2] >= board_min[:2] - 1e-9)
                    assert np.all(corner[:2] <= board_max[:2] + 1e-9)

        grasp_by_part = {
            str(task['base_part']): resolved['selected_base_grasp'],
            **resolved['selected_move_grasps'],
        }
        workspace_offset = np.asarray(recipe['workspace_offset'], dtype=float)
        for part_id, grasp in grasp_by_part.items():
            object_pose = objects[f'fabrica_{task_name}_{part_id}']
            tcp_position = _target_tcp_position(object_pose, grasp)
            tcp_position += workspace_offset + np.asarray([0.0, 0.0, 0.10])
            robot_name = 'franka_right' if part_id == str(task['base_part']) else 'franka_left'
            robot_position = np.asarray(robots[robot_name]['position'], dtype=float)
            if robots[robot_name].get('apply_workspace_offset', True):
                robot_position += workspace_offset
            assert np.linalg.norm(tcp_position - robot_position) <= 0.82 + 1e-9
            assert np.isclose(
                np.linalg.norm(tcp_position - robot_position),
                selection['pickup_tcp_reach_by_part'][part_id],
            )


def test_staged_randomization_moves_layouts_but_never_the_optical_board():
    for task_name in STAGED_TASKS:
        recipe = load_task_recipe(
            f'fabrica_{task_name}_ur5e_staged',
            scene_profile='taoyuan_grscenes_tabletop',
        )
        randomized_a, result_a = apply_domain_randomization(
            recipe,
            seed=17,
            enabled_override=True,
        )
        randomized_b, result_b = apply_domain_randomization(
            recipe,
            seed=18,
            enabled_override=True,
        )
        original_objects = _object_map(recipe)
        objects_a = _object_map(randomized_a)
        objects_b = _object_map(randomized_b)
        lights_a = {entry['name']: entry for entry in randomized_a['scene_lights']}

        np.testing.assert_allclose(
            objects_a['optical_board']['position'],
            original_objects['optical_board']['position'],
        )
        np.testing.assert_allclose(
            objects_b['optical_board']['position'],
            original_objects['optical_board']['position'],
        )
        assert 'domain_randomization_group' not in objects_a['optical_board']
        pickup_distance_a = np.linalg.norm(result_a['groups']['start_parts']['translation'][:2])
        assembly_distance_a = np.linalg.norm(result_a['groups']['assembly_base']['translation'][:2])
        assert 0.05 <= pickup_distance_a <= 0.12
        assert 0.05 <= assembly_distance_a <= 0.15
        assert result_a['groups']['start_parts']['translation'] != result_b['groups']['start_parts']['translation']
        assert result_a['groups']['assembly_base']['translation'] != result_b['groups']['assembly_base']['translation']
        assert set(result_a['appearance_groups']) == {'table_surface', 'background'}
        table_color = result_a['appearance_groups']['table_surface']['color']
        background_color = result_a['appearance_groups']['background']['color']
        np.testing.assert_allclose(
            objects_a['factory_tabletop_visual']['color'],
            table_color,
        )
        assert result_a['appearance_groups']['background']['objects'] == ['factory_background_visual']
        np.testing.assert_allclose(
            objects_a['factory_background_visual']['color'],
            background_color,
        )
        np.testing.assert_allclose(
            lights_a['warehouse_dome_fill']['color'],
            background_color,
        )


def test_staged_randomization_keeps_each_layout_rigid_and_independent():
    recipe = load_task_recipe(
        'fabrica_beam_ur5e_staged',
        scene_profile='taoyuan_grscenes_tabletop',
    )
    randomized, result = apply_domain_randomization(recipe, seed=4906, enabled_override=True)
    original_objects = _object_map(recipe)
    randomized_objects = _object_map(randomized)
    original_targets = _target_map(recipe)
    randomized_targets = _target_map(randomized)

    pickup_delta = np.asarray(result['groups']['start_parts']['translation'])
    for object_name in result['groups']['start_parts']['objects']:
        np.testing.assert_allclose(
            np.asarray(randomized_objects[object_name]['position'])
            - np.asarray(original_objects[object_name]['position']),
            pickup_delta,
        )

    assembly_delta = np.asarray(result['groups']['assembly_base']['translation'])
    for target_name in result['groups']['assembly_base']['targets']:
        np.testing.assert_allclose(
            np.asarray(randomized_targets[target_name]['position'])
            - np.asarray(original_targets[target_name]['position']),
            assembly_delta,
        )


def test_staged_randomization_respects_fixed_board_bounds_and_tcp_reach():
    for task_name in STAGED_TASKS:
        recipe = load_task_recipe(
            f'fabrica_{task_name}_ur5e_staged',
            scene_profile='taoyuan_grscenes_tabletop',
        )
        original_objects = _object_map(recipe)
        groups = recipe['domain_randomization']['groups']

        for seed in range(64):
            randomized, result = apply_domain_randomization(
                recipe,
                seed=seed,
                enabled_override=True,
            )
            randomized_objects = _object_map(randomized)
            np.testing.assert_allclose(
                randomized_objects['optical_board']['position'],
                original_objects['optical_board']['position'],
            )

            for group_name, group in groups.items():
                translation = np.asarray(result['groups'][group_name]['translation'], dtype=float)
                for constraint in group['translation_constraints']:
                    points = np.asarray(constraint['points'], dtype=float) + translation
                    if constraint['type'] == 'points_inside_bounds':
                        lower = np.asarray(constraint['lower'], dtype=float)
                        upper = np.asarray(constraint['upper'], dtype=float)
                        for axis in constraint['axes']:
                            assert np.all(points[:, axis] >= lower[axis] - 1e-12)
                            assert np.all(points[:, axis] <= upper[axis] + 1e-12)
                    elif constraint['type'] == 'points_within_distance':
                        distances = np.linalg.norm(
                            points - np.asarray(constraint['origin'], dtype=float),
                            axis=1,
                        )
                        assert np.all(distances <= float(constraint['maximum_distance']) + 1e-12)
                    else:
                        raise AssertionError(f'Unexpected constraint type: {constraint["type"]}')
