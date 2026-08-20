from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from internutopia_extension.robots.franka import FrankaRobot
from internutopia_extension.tasks.factory_dual_franka_assembly_task import (
    FactoryDualFrankaAssemblyTask,
)
from roboassemblybench.core.task_registry import load_task_recipe
from roboassemblybench.datasets.cartesian_episode import expected_replay_joint_widths
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import (
    AssemblyAtomicSkillAdapter,
)
from toolkits.factory_dual_franka_assembly.scene_builder import _build_robot_cfgs


FABRICA_TASKS = (
    'beam',
    'car',
    'cooling_manifold',
    'duct',
    'gamepad',
    'plumbers_block',
    'stool_circular',
)


def test_franka_replay_joint_signature_rejects_ur5e_widths():
    recipe = load_task_recipe('fabrica_beam_franka_staged')

    assert expected_replay_joint_widths(recipe['robots']) == [9, 9]
    assert expected_replay_joint_widths([{'type': 'UR5eRobot'}, {'type': 'UR5eRobot'}]) is None


def test_release_lock_uses_explicit_geometry_tolerance_over_target_default(monkeypatch):
    task = FactoryDualFrankaAssemblyTask.__new__(FactoryDualFrankaAssemblyTask)
    task._locked_targets = {}
    task._attachments = {}
    task.phase = 'part_release_and_lock'
    task.phase_step_counter = 1
    monkeypatch.setattr(
        task,
        '_resolve_object',
        lambda _name: SimpleNamespace(
            get_pose=lambda: (
                np.asarray([0.018, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0, 0.0]),
            )
        ),
    )
    monkeypatch.setattr(
        task,
        '_resolve_target_pose_spec',
        lambda _name: (
            'assembled',
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            {'position_tolerance': 0.015, 'orientation_tolerance': 0.10},
        ),
    )

    assert task._lock_ready(
        {'gripper_commands': {}},
        {
            'object': 'part',
            'target': 'assembled',
            'position_tolerance': 0.02136,
            'orientation_tolerance': 0.12,
        },
    )


def test_franka_arm_drive_is_written_to_physx_with_native_dtype():
    class PhysicsView:
        def __init__(self):
            self.stiffnesses = np.full((1, 9), 1.0, dtype=np.float32)
            self.dampings = np.full((1, 9), 2.0, dtype=np.float32)
            self.max_forces = np.full((1, 9), 3.0, dtype=np.float32)

        def get_dof_stiffnesses(self):
            return self.stiffnesses

        def get_dof_dampings(self):
            return self.dampings

        def get_dof_max_forces(self):
            return self.max_forces

        def set_dof_stiffnesses(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.stiffnesses = data.copy()

        def set_dof_dampings(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.dampings = data.copy()

        def set_dof_max_forces(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.max_forces = data.copy()

    physics_view = PhysicsView()
    robot = FrankaRobot.__new__(FrankaRobot)
    robot.config = SimpleNamespace(
        name='franka_test',
        arm_joint_stiffness=80000.0,
        arm_joint_damping=4000.0,
        arm_joint_max_force=600.0,
    )
    robot.articulation = SimpleNamespace(
        get_dof_index=lambda name: int(name.removeprefix('panda_joint')) - 1,
        _articulation_view=SimpleNamespace(_physics_view=physics_view),
    )

    robot._apply_configured_arm_drive()

    np.testing.assert_allclose(physics_view.stiffnesses[0, :7], 80000.0)
    np.testing.assert_allclose(physics_view.dampings[0, :7], 4000.0)
    np.testing.assert_allclose(physics_view.max_forces[0, :7], 600.0)
    np.testing.assert_allclose(physics_view.stiffnesses[0, 7:], 1.0)
    np.testing.assert_allclose(physics_view.dampings[0, 7:], 2.0)
    np.testing.assert_allclose(physics_view.max_forces[0, 7:], 3.0)


def test_franka_gripper_drive_is_written_to_physx_with_native_dtype():
    class PhysicsView:
        def __init__(self):
            self.stiffnesses = np.full((1, 9), 1.0, dtype=np.float32)
            self.dampings = np.full((1, 9), 2.0, dtype=np.float32)
            self.max_forces = np.full((1, 9), 3.0, dtype=np.float32)
            self.frictions = np.full((1, 9), 4.0, dtype=np.float32)

        def get_dof_stiffnesses(self):
            return self.stiffnesses

        def get_dof_dampings(self):
            return self.dampings

        def get_dof_max_forces(self):
            return self.max_forces

        def get_dof_friction_coefficients(self):
            return self.frictions

        def set_dof_stiffnesses(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.stiffnesses = data.copy()

        def set_dof_dampings(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.dampings = data.copy()

        def set_dof_max_forces(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.max_forces = data.copy()

        def set_dof_friction_coefficients(self, data, indices):
            assert data.dtype == np.float32
            assert indices == [0]
            self.frictions = data.copy()

    physics_view = PhysicsView()
    robot = FrankaRobot.__new__(FrankaRobot)
    robot.config = SimpleNamespace(
        name='franka_test',
        gripper_joint_stiffness=20000.0,
        gripper_joint_damping=1000.0,
        gripper_joint_max_force=300.0,
        gripper_joint_friction=5.0,
    )
    robot.articulation = SimpleNamespace(
        get_dof_index=lambda name: {'panda_finger_joint1': 7, 'panda_finger_joint2': 8}[name],
        _articulation_view=SimpleNamespace(_physics_view=physics_view),
    )

    robot._apply_configured_gripper_drive()

    np.testing.assert_allclose(physics_view.stiffnesses[0, :7], 1.0)
    np.testing.assert_allclose(physics_view.stiffnesses[0, 7:], 20000.0)
    np.testing.assert_allclose(physics_view.dampings[0, 7:], 1000.0)
    np.testing.assert_allclose(physics_view.max_forces[0, 7:], 300.0)
    np.testing.assert_allclose(physics_view.frictions[0, 7:], 5.0)


def test_franka_initial_state_synchronizes_position_velocity_and_drive_target():
    class Articulation:
        def __init__(self):
            self.positions = np.zeros(9, dtype=float)
            self.velocities = np.ones(9, dtype=float)
            self.action = None

        @staticmethod
        def get_dof_index(name):
            return int(name.removeprefix('panda_joint')) - 1

        def set_joint_positions(self, positions, joint_indices):
            self.positions[joint_indices] = positions

        def set_joint_velocities(self, velocities, joint_indices):
            self.velocities[joint_indices] = velocities

        def apply_action(self, action):
            self.action = action

        def get_joint_positions(self, joint_indices):
            return self.positions[joint_indices]

        def get_joint_velocities(self, joint_indices):
            return self.velocities[joint_indices]

    configured = {
        f'panda_joint{index}': value
        for index, value in enumerate([0.0, -0.7, 0.1, -2.3, 0.2, 1.5, 0.8], start=1)
    }
    robot = FrankaRobot.__new__(FrankaRobot)
    robot.config = SimpleNamespace(name='franka_test', initial_joint_positions=configured)
    robot.articulation = Articulation()

    robot._apply_configured_initial_joint_state()

    expected = np.asarray(list(configured.values()), dtype=float)
    indices = np.arange(7, dtype=np.int64)
    np.testing.assert_allclose(robot.articulation.positions[indices], expected)
    np.testing.assert_allclose(robot.articulation.velocities[indices], 0.0)
    np.testing.assert_allclose(robot.articulation.action.joint_positions, expected)
    np.testing.assert_allclose(robot.articulation.action.joint_velocities, 0.0)
    np.testing.assert_array_equal(robot.articulation.action.joint_indices, indices)


def test_scene_builder_preserves_franka_runtime_drive_configuration():
    robots, names = _build_robot_cfgs(
        {
            'robots': [
                {
                    'name': 'franka_left',
                    'type': 'FrankaRobot',
                    'prim_path': '/franka_left',
                    'position': [0.0, 0.0, 0.0],
                    'orientation': [1.0, 0.0, 0.0, 0.0],
                    'arm_joint_stiffness': 80000.0,
                    'arm_joint_damping': 4000.0,
                    'arm_joint_max_force': 600.0,
                    'gripper_joint_stiffness': 20000.0,
                    'gripper_joint_damping': 1000.0,
                    'gripper_joint_max_force': 300.0,
                    'gripper_joint_friction': 5.0,
                    'gripper_dof_name': 'panda_finger_joint1',
                    'gripper_open_position': 0.04,
                    'gripper_closed_position': 0.0,
                }
            ]
        }
    )

    assert names == ('franka_left',)
    assert robots[0].arm_joint_stiffness == pytest.approx(80000.0)
    assert robots[0].arm_joint_damping == pytest.approx(4000.0)
    assert robots[0].arm_joint_max_force == pytest.approx(600.0)
    assert robots[0].gripper_joint_stiffness == pytest.approx(20000.0)
    assert robots[0].gripper_joint_damping == pytest.approx(1000.0)
    assert robots[0].gripper_joint_max_force == pytest.approx(300.0)
    assert robots[0].gripper_joint_friction == pytest.approx(5.0)
    assert robots[0].gripper_dof_name == 'panda_finger_joint1'
    assert robots[0].gripper_open_position == pytest.approx(0.04)
    assert robots[0].gripper_closed_position == pytest.approx(0.0)


@pytest.mark.parametrize('assembly', FABRICA_TASKS)
def test_franka_canonical_task_compiles_with_panda_grasps(assembly):
    recipe = load_task_recipe(f'fabrica_{assembly}_franka_staged')
    resolved = recipe['fabrica_canonical_resolved']

    assert resolved['assembly'] == assembly
    assert resolved['robot_platform'] == 'franka'
    if assembly == 'car':
        assert recipe['fabrica_canonical']['base_robot'] == 'franka_left'
        assert recipe['fabrica_canonical']['assembly_robot'] == 'franka_right'
    expected_pose_overrides = (
        {
            'franka_left': {
                'position': [0.05, 0.25, 0.998051],
                'orientation': [0.707106781, 0.0, 0.0, -0.707106781],
            },
            'franka_right': {
                'position': [0.95, 0.25, 0.998051],
                'orientation': [0.707106781, 0.0, 0.0, -0.707106781],
            },
        }
        if assembly == 'car'
        else {}
    )
    assert resolved['robot_pose_overrides'] == expected_pose_overrides
    assert resolved['robot_home_joint_overrides'] == {}
    assert resolved['pickup_approach_mode'] == 'grasp_axis'
    assert recipe['phases']
    assert {robot['type'] for robot in recipe['robots']} == {'FrankaRobot'}
    expected_joint_names = {f'panda_joint{index}' for index in range(1, 8)}
    assert all(set(robot['initial_joint_positions']) == expected_joint_names for robot in recipe['robots'])
    assert all(robot['arm_joint_stiffness'] == pytest.approx(80000.0) for robot in recipe['robots'])
    assert all(robot['arm_joint_damping'] == pytest.approx(4000.0) for robot in recipe['robots'])
    assert all(robot['arm_joint_max_force'] == pytest.approx(600.0) for robot in recipe['robots'])
    assert all(robot['gripper_joint_stiffness'] == pytest.approx(20000.0) for robot in recipe['robots'])
    assert all(robot['gripper_joint_damping'] == pytest.approx(1000.0) for robot in recipe['robots'])
    assert all(robot['gripper_joint_max_force'] == pytest.approx(300.0) for robot in recipe['robots'])
    assert all(robot['gripper_joint_friction'] == pytest.approx(5.0) for robot in recipe['robots'])
    assert all(robot['gripper_dof_name'] == 'panda_finger_joint1' for robot in recipe['robots'])
    assert all(robot['gripper_open_position'] == pytest.approx(0.04) for robot in recipe['robots'])
    assert all(robot['gripper_closed_position'] == pytest.approx(0.0) for robot in recipe['robots'])

    grasps = [resolved['selected_base_grasp'], *resolved['selected_move_grasps'].values()]
    assert all(grasp['target_gripper'] == 'panda' for grasp in grasps)
    assert all(grasp['gripper_open_ratio'] == grasp['panda_open_ratio'] for grasp in grasps)
    assert all(len(grasp['object_in_tcp_position']) == 3 for grasp in grasps)
    assert all(len(grasp['object_in_tcp_orientation']) == 4 for grasp in grasps)
    if assembly == 'car':
        assert all(
            grasp['pickup_fixture_body_clearance'] >= 0.020
            for grasp in resolved['selected_move_grasps'].values()
        )
    assert resolved['selected_base_grasp']['ik_feasible'] is True
    assert resolved['selected_base_grasp']['pickup_fixture_body_clearance'] >= 0.0
    assert resolved['pickup_layout_selection']['base_ik_selected']['ik_feasible'] is True
    assert resolved['pickup_layout_selection']['base_ik_enforced'] is True
    assert resolved['pickup_layout_selection']['base_enforce_fixture_clearance'] is True
    assert resolved['pickup_layout_selection']['base_prioritize_fixture_clearance'] is True
    assert resolved['pickup_layout_selection']['base_selected_fixture_clearance'] >= 0.0
    selection = resolved['pickup_layout_selection']
    assert (
        resolved['selected_base_grasp']['source_collision_count']
        >= selection['base_minimum_source_collision_count']
    )
    assert selection['base_source_collision_fallback_used'] == (
        resolved['selected_base_grasp']['source_collision_count']
        > selection['base_minimum_source_collision_count']
    )
    assert resolved['pickup_layout_selection']['minimum_vertical_clearance'] == pytest.approx(0.70)
    assert (
        resolved['selected_base_grasp']['is_planner_grasp']
        or resolved['selected_base_grasp']['clearance_score'] >= 0.70
    )

    wrist_cameras = [camera for camera in recipe['camera_specs'] if camera['view_type'] == 'wrist']
    assert len(wrist_cameras) == 2
    assert all(camera['prim_path'].startswith('panda_hand/') for camera in wrist_cameras)
    assert all(camera['depth'] for camera in recipe['camera_specs'])

    robots = {robot['name']: robot for robot in recipe['robots']}
    expected_left_position = [0.05, 0.25, 0.998051] if assembly == 'car' else [0.518, 0.54, 0.998]
    expected_right_position = [0.95, 0.25, 0.998051] if assembly == 'car' else [0.474, -0.54, 0.998]
    assert robots['franka_left']['position'] == pytest.approx(expected_left_position)
    assert robots['franka_right']['position'] == pytest.approx(expected_right_position)
    assert robots['franka_left']['orientation'] == pytest.approx(
        [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]
    )
    assert robots['franka_left']['initial_joint_positions']['panda_joint1'] == pytest.approx(0.0)
    expected_right_orientation = (
        [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]
        if assembly == 'car'
        else [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    assert robots['franka_right']['orientation'] == pytest.approx(expected_right_orientation)
    expected_origins = (
        ([0.50, 0.05, 0.0125], [0.50, -0.10, 0.0125], [0.50, 0.05, 0.0125])
        if assembly == 'car'
        else ([0.35, -0.10, 0.0125], [0.70, 0.0, 0.0125], [0.47, 0.0, 0.0125])
    )
    assert recipe['fabrica_canonical']['pickup_origin'] == pytest.approx(expected_origins[0])
    assert recipe['fabrica_canonical']['assembly_origin'] == pytest.approx(expected_origins[1])
    assert recipe['fabrica_canonical']['board_origin'] == pytest.approx(expected_origins[2])
    park_phases = [phase for phase in recipe['phases'] if phase['name'].endswith('_park')]
    assert park_phases
    assert not recipe['phases'][-1]['name'].endswith('_park')
    assert resolved['post_release_park_mode'] == 'joint_home'
    assert resolved['post_release_park_joint_position_tolerance'] == pytest.approx(0.05)
    assert resolved['post_release_park_timeout_steps'] == 3000
    assert resolved['skip_terminal_park'] is True
    assert resolved['skip_terminal_retreat'] is (assembly in {'car', 'duct'})
    assert recipe['phases'][-1]['name'].endswith(
        '_release_and_lock' if assembly in {'car', 'duct'} else '_retreat'
    )
    assert resolved['pickup_departure_interpolation_waypoint_count'] == 2
    assert resolved['transport_interpolation_waypoint_count'] == 4
    assert resolved['assembly_approach_interpolation_waypoint_count'] == 4
    assert resolved['idle_robot_home'] is True
    assert resolved['idle_robot_home_joint_position_tolerance'] == pytest.approx(0.05)
    assert resolved['idle_robot_home_max_joint_step'] == pytest.approx(0.020)
    assert resolved['descend_timeout_steps'] == 2400
    assert resolved['transport_hover_object_position_tolerance'] == pytest.approx(0.0035)
    assert resolved['descend_orientation_tolerance'] == pytest.approx(0.18)
    assert resolved['close_position_tolerance'] == pytest.approx(0.015)
    assert resolved['close_orientation_tolerance'] == pytest.approx(0.18)
    assert resolved['close_phase_timeout_steps'] == 1200
    assert resolved['close_until_contact_timeout_steps'] == 720
    assert resolved['allow_initial_strict_contact_for_short_close'] is True
    assert resolved['allow_initial_strict_contact_without_closure'] is True
    assert resolved['physical_attach_surface_gap'] == pytest.approx(0.008)
    assert resolved['disable_collision_during_fixed_transport'] is True
    assert resolved['close_gate_recenter_min_gap_imbalance'] == pytest.approx(0.0005)
    assert resolved['close_gate_recenter_target_tolerance'] == pytest.approx(0.0025)
    assert resolved['close_gate_recenter_gap_gain'] == pytest.approx(0.5)
    assert resolved['close_gate_recenter_max_step'] == pytest.approx(0.008)
    assert resolved['insertion_lateral_alignment_enter_tolerance_ratio'] == pytest.approx(1.0)
    assert resolved['insertion_lateral_alignment_minimum_tolerance'] == pytest.approx(0.0035)
    assert resolved['release_lock_uses_placement_tolerance'] is True
    assert resolved['pickup_layout_selection']['move_grasp_ik_minimum_manipulability'] == pytest.approx(0.04)
    assert resolved['pickup_layout_selection']['move_grasp_ik_preferred_manipulability'] == pytest.approx(0.075)
    assert resolved['pickup_layout_selection']['move_grasp_prioritize_fixture_clearance'] is True
    assert resolved['pickup_layout_selection']['move_grasp_preferred_fixture_clearance_slack'] == pytest.approx(
        0.005
    )
    for grasp in grasps:
        ik_targets = grasp['ik_errors_by_target']
        assert 'pickup_clearance' in ik_targets
        assert 'pickup_departure_01' in ik_targets
        assert 'pickup_departure_02' in ik_targets
        assert 'assembly_approach_01' in ik_targets
        assert 'assembly_approach_02' in ik_targets
        assert 'assembly_approach_03' in ik_targets
        assert 'assembly_approach_04' in ik_targets
        assert all(f'transport_{index:02d}' in ik_targets for index in range(1, 5))
        assert 'assembly_clearance' in ik_targets
        assert 'transport_hover' in ik_targets
        assert 'release_retreat' in ik_targets

    phase_order = {phase['name']: index for index, phase in enumerate(recipe['phases'])}
    pickup_lift_phases = [name for name in phase_order if name.endswith('_lift')]
    assert pickup_lift_phases
    for lift_name in pickup_lift_phases:
        prefix = lift_name.removesuffix('_lift')
        departure_names = [
            f'{prefix}_pickup_departure_01',
            f'{prefix}_pickup_departure_02',
        ]
        assert all(name in phase_order for name in departure_names)
        assert (
            phase_order[lift_name]
            < phase_order[departure_names[0]]
            < phase_order[departure_names[1]]
            < phase_order[f'{prefix}_pickup_clearance']
        )
    assert resolved['insertion_compliance_waypoint_axial_tolerance_object_extent_scale'] == 0.8
    descend_phases = [phase for phase in recipe['phases'] if phase['name'].endswith('_descend')]
    assert descend_phases
    assert all(phase['timeout_steps'] == 2400 for phase in descend_phases)
    assert all(phase['local_skill']['orientation_tolerance'] == pytest.approx(0.18) for phase in descend_phases)
    for phase in park_phases:
        assert phase['timeout_steps'] == 3000
        local_skill = phase['local_skill']
        assert local_skill['name'] == 'move_arm_to_joint_positions'
        robot = robots[local_skill['robot']]
        expected = [robot['initial_joint_positions'][f'panda_joint{index}'] for index in range(1, 8)]
        np.testing.assert_allclose(local_skill['joint_positions'], expected)
        assert local_skill['joint_position_tolerance'] == pytest.approx(0.05)
        assert local_skill['unwrap_revolute_joints'] is False
        assert 'workspace_minimum_planar_radius' not in local_skill

    robot_names = set(robots)
    for phase in recipe['phases']:
        active_robots = set(phase.get('robot_targets', {})) | set(phase.get('gripper_commands', {}))
        local_skill = phase.get('local_skill') or {}
        if local_skill.get('robot') is not None:
            active_robots.add(local_skill['robot'])
        attach_entries = phase.get('attach') or []
        if isinstance(attach_entries, dict):
            attach_entries = [attach_entries]
        active_robots.update(
            entry['robot']
            for entry in attach_entries
            if isinstance(entry, dict) and entry.get('robot') is not None
        )
        idle_skills = phase.get('local_skills', {})
        active_robots.update(
            robot_name
            for robot_name, skill in idle_skills.items()
            if isinstance(skill, dict) and not skill.get('idle_home', False)
        )
        idle_robots = robot_names - active_robots
        assert idle_robots <= set(idle_skills)
        for robot_name in idle_robots:
            idle_skill = idle_skills[robot_name]
            assert idle_skill['name'] == 'move_arm_to_joint_positions'
            assert idle_skill['idle_home'] is True
            expected = [
                robots[robot_name]['initial_joint_positions'][f'panda_joint{index}']
                for index in range(1, 8)
            ]
            np.testing.assert_allclose(idle_skill['joint_positions'], expected)

    insertion_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if '_insert_' in phase['name']
    ]
    assert insertion_skills
    assert all(
        skill['target_object_lateral_alignment_enter_tolerance']
        == pytest.approx(max(skill['target_object_lateral_position_tolerance'], 0.0035))
        for skill in insertion_skills
    )
    assert all(
        skill['target_object_lateral_alignment_exit_tolerance']
        >= skill['target_object_lateral_alignment_enter_tolerance']
        for skill in insertion_skills
    )
    assert all(
        skill['target_object_lateral_position_tolerance'] == pytest.approx(0.006)
        for skill in insertion_skills
    )
    assert all(
        skill['target_object_lateral_alignment_enter_tolerance']
        == pytest.approx(skill['target_object_lateral_position_tolerance'])
        for skill in insertion_skills
    )
    final_insertion_skills = [
        skill
        for skill in insertion_skills
        if skill['target_object_target'].endswith('_assembled')
    ]
    assert final_insertion_skills
    assert all(
        skill['target_object_axial_position_tolerance']
        == pytest.approx(skill['relax_fixed_attachment_within_final_position_tolerance'])
        for skill in final_insertion_skills
    )
    cartesian_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase.get('local_skill', {}).get('cartesian_servo')
    ]
    assert cartesian_skills
    assert all(skill['unwrap_revolute_joints'] is False for skill in cartesian_skills)
    assert all(skill['guard_ik_branch_jump'] is True for skill in cartesian_skills)
    assert all(skill['require_warm_start_ik'] is True for skill in cartesian_skills)
    assert all(skill['warm_start_ik_only'] is True for skill in cartesian_skills)
    assert all(skill['ik_branch_jump_reference_mode'] == 'previous_target' for skill in cartesian_skills)
    assert all(skill['lock_transport_ik_target'] is True for skill in cartesian_skills)
    assert all(skill['lock_pickup_ik_target'] is True for skill in cartesian_skills)
    assert all(skill['pickup_terminal_ik_position_window'] == pytest.approx(0.04) for skill in cartesian_skills)
    assert all(skill['pickup_terminal_ik_orientation_window'] == pytest.approx(0.10) for skill in cartesian_skills)
    close_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase.get('local_skill', {}).get('require_close_pose_gate')
    ]
    assert close_skills
    attach_specs = [
        attach
        for phase in recipe['phases']
        for attach in phase.get('attach', [])
    ]
    assert attach_specs
    assert all(attach['filter_gripper_collisions_on_attach'] is True for attach in attach_specs)
    assert all(attach['disable_collision_on_attach'] is True for attach in attach_specs)
    transport_interpolation_phases = [
        phase
        for phase in recipe['phases']
        if phase['name'].endswith(tuple(f'_transport_{index:02d}' for index in range(1, 5)))
    ]
    assert len(transport_interpolation_phases) == 4 * len(close_skills)
    assembly_approach_phases = [
        phase
        for phase in recipe['phases']
        if '_assembly_approach_' in phase['name']
    ]
    assert len(assembly_approach_phases) == 4 * len(close_skills)
    assert all(skill['allow_initial_strict_contact_for_short_close'] is True for skill in close_skills)
    assert all(skill['allow_initial_strict_contact_without_closure'] is True for skill in close_skills)
    assert all(skill['physical_attach_surface_gap'] == pytest.approx(0.008) for skill in close_skills)
    assert all(
        skill['close_gate_recenter_min_gap_imbalance'] == pytest.approx(0.0005)
        for skill in close_skills
    )
    assert all(
        skill['close_gate_recenter_target_tolerance'] == pytest.approx(0.0025)
        for skill in close_skills
    )
    assert all(skill['close_gate_recenter_gap_gain'] == pytest.approx(0.5) for skill in close_skills)
    assert all(skill['close_gate_recenter_max_step'] == pytest.approx(0.008) for skill in close_skills)
    assert all(skill['close_position_tolerance'] == pytest.approx(0.015) for skill in close_skills)
    assert all(skill['close_orientation_tolerance'] == pytest.approx(0.18) for skill in close_skills)
    assert all(skill['close_until_contact_timeout_steps'] == 720 for skill in close_skills)
    relative_lift_and_retreat_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase['name'].endswith(('_lift', '_retreat'))
    ]
    assert relative_lift_and_retreat_skills
    assert all(skill['relative_to_current_tcp'] is True for skill in relative_lift_and_retreat_skills)
    assert all(skill['lock_target_position'] is True for skill in relative_lift_and_retreat_skills)
    assert all(skill['lock_target_orientation'] is True for skill in relative_lift_and_retreat_skills)
    transport_hover_skills = [
        phase['local_skill'] for phase in recipe['phases'] if phase['name'].endswith('_transport_hover')
    ]
    assert transport_hover_skills
    assert all(
        skill['target_object_position_tolerance'] == pytest.approx(0.0035)
        for skill in transport_hover_skills
    )
    assert all(skill['target_object_capture_requires_tcp'] is False for skill in transport_hover_skills)
    assert all(skill['require_target_object_static'] is True for skill in transport_hover_skills)
    assert all(skill['hold_for_target_object_settle'] is True for skill in transport_hover_skills)
    assert all(skill['close_gate_guard_ik_branch_jump'] is True for skill in close_skills)
    assert all(skill['require_warm_start_ik'] is True for skill in close_skills)
    assert all(skill['warm_start_ik_only'] is True for skill in close_skills)
    free_transport_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase.get('local_skill', {}).get('requires_held_object')
        and phase.get('local_skill', {}).get('target_object_target') is not None
        and phase.get('local_skill', {}).get('target_object_convergence_axis') is None
        and phase.get('local_skill', {}).get('target_object_final_target') is None
    ]
    insertion_skills_with_axis = [
        skill for skill in insertion_skills if skill.get('target_object_convergence_axis') is not None
    ]
    assert free_transport_skills
    assert insertion_skills_with_axis
    assert all(skill['guard_ik_branch_jump'] is True for skill in insertion_skills_with_axis)
    assert all(skill['require_warm_start_ik'] is True for skill in insertion_skills_with_axis)
    assert all(skill['warm_start_ik_only'] is True for skill in insertion_skills_with_axis)
    assert all(AssemblyAtomicSkillAdapter._use_locked_transport_ik_target(skill) for skill in free_transport_skills)
    assert not any(
        AssemblyAtomicSkillAdapter._use_locked_transport_ik_target(skill)
        for skill in insertion_skills_with_axis
    )
    move_above_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase['name'].endswith('_move_above')
    ]
    lift_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase['name'].endswith('_lift')
    ]
    assert move_above_skills
    assert len(lift_skills) == len(move_above_skills)
    assert all(skill['prealign_target_ik'] is True for skill in move_above_skills)
    assert all(skill['prealign_terminal_ik_seed'] is True for skill in move_above_skills)
    assert all(skill['prealign_until_converged'] is True for skill in move_above_skills)
    assert all(skill['prealign_joint_position_tolerance'] == pytest.approx(0.05) for skill in move_above_skills)
    assert all(skill['prealign_joint_velocity_tolerance'] == pytest.approx(0.15) for skill in move_above_skills)
    assert all(skill['prealign_ready_stable_steps'] == 6 for skill in move_above_skills)
    assert all(skill['prealign_max_joint_step'] == pytest.approx(0.06) for skill in move_above_skills)
    assert all(skill['prealign_max_command_tracking_error'] == pytest.approx(0.06) for skill in move_above_skills)
    assert all(skill['prealign_timeout_steps'] == 2400 for skill in move_above_skills)
    assert all(skill['offset_frame'] == 'object' for skill in move_above_skills)
    descend_skills = [
        phase['local_skill']
        for phase in recipe['phases']
        if phase['name'].endswith('_descend')
    ]
    assert descend_skills
    assert all(
        AssemblyAtomicSkillAdapter._use_locked_pickup_ik_target(
            skill_name=skill['name'],
            spec=skill,
        )
        for skill in descend_skills
    )
    approach_height = recipe['fabrica_canonical'].get('approach_height', 0.10)
    expected_grasps = [resolved['selected_base_grasp'], *resolved['selected_move_grasps'].values()]
    for skill, grasp in zip(move_above_skills, expected_grasps):
        direction = np.asarray(grasp['assembly_approach_direction'], dtype=float)
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(skill['offset'], direction * approach_height)
    for move_above_skill, lift_skill in zip(move_above_skills, lift_skills):
        np.testing.assert_allclose(lift_skill['offset'], move_above_skill['offset'])
        assert lift_skill['offset_frame'] == move_above_skill['offset_frame']
    for grasp in resolved['selected_move_grasps'].values():
        assert grasp['ik_preferred_manipulability'] == pytest.approx(0.075)
        assert grasp['ik_preferred_manipulability_met'] == (
            grasp['ik_minimum_path_manipulability'] >= 0.075
        )
    for diagnostics in resolved['move_grasp_selection'].values():
        assert diagnostics['prioritize_fixture_clearance'] is True
        assert (
            diagnostics['preferred_fixture_candidate_count'] > 0
            or diagnostics['planner_priority_selected']
        )
        if diagnostics['planner_priority_selected']:
            assert diagnostics['selected']['is_planner_grasp'] is True
            assert diagnostics['selected']['ik_feasible'] is True
        elif diagnostics['fixture_clearance_fallback_used']:
            assert diagnostics['preferred_fixture_ik_feasible_candidate_count'] == 0
        else:
            assert diagnostics['selected']['pickup_fixture_body_clearance'] >= diagnostics[
                'preferred_fixture_clearance'
            ]
    assert all(
        0.010 <= skill['relax_fixed_attachment_waypoint_axial_position_tolerance'] <= 0.020
        for skill in insertion_skills
    )
    release_phases = [
        phase for phase in recipe['phases'] if phase['name'].endswith('_release_and_lock')
    ]
    assert release_phases
    assert all(
        phase['lock'][0]['position_tolerance']
        >= resolved['insertion_relaxed_position_tolerance']
        for phase in release_phases
    )

    base_place = next(phase['local_skill'] for phase in recipe['phases'] if phase['name'].endswith('_place'))
    assert base_place['target_object_final_target'].endswith('_assembled')
    assert (
        base_place['relax_fixed_attachment_within_final_position_tolerance']
        >= resolved['base_support_release_position_tolerance']
    )
    assert base_place['cartesian_position_step'] == 0.0005
    assert base_place['target_object_servo_position_command_accumulation_step'] == 0.0005


def test_pickup_terminal_ik_lock_activates_only_near_contact():
    spec = {
        'lock_pickup_ik_target': True,
        'object': 'part',
        'pickup_terminal_ik_position_window': 0.04,
        'pickup_terminal_ik_orientation_window': 0.50,
    }
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }

    assert not AssemblyAtomicSkillAdapter._use_locked_pickup_ik_target(
        skill_name='ur5e_descend_to_grasp',
        spec=spec,
        current_pose={
            'position': np.asarray([0.0, 0.0, 0.10]),
            'orientation': target_pose['orientation'],
        },
        target_pose=target_pose,
    )
    assert AssemblyAtomicSkillAdapter._use_locked_pickup_ik_target(
        skill_name='ur5e_descend_to_grasp',
        spec=spec,
        current_pose={
            'position': np.asarray([0.0, 0.0, 0.10]),
            'orientation': target_pose['orientation'],
        },
        target_pose=target_pose,
        already_locked=True,
    )
    assert AssemblyAtomicSkillAdapter._use_locked_pickup_ik_target(
        skill_name='ur5e_descend_to_grasp',
        spec=spec,
        current_pose={
            'position': np.asarray([0.0, 0.0, 0.03]),
            'orientation': target_pose['orientation'],
        },
        target_pose=target_pose,
    )
    assert not AssemblyAtomicSkillAdapter._use_locked_pickup_ik_target(
        skill_name='ur5e_descend_to_grasp',
        spec=spec,
        current_pose={
            'position': np.asarray([0.0, 0.0, 0.03]),
            'orientation': np.asarray([np.cos(0.30), 0.0, 0.0, np.sin(0.30)]),
        },
        target_pose=target_pose,
    )


def test_close_gate_continues_toward_matching_pickup_terminal_seed(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase='part_close_and_attach', phase_step_counter=0, step_counter=50)
    phase_key = ('close-terminal-seed',)
    robot_name = 'franka_left'
    object_name = 'fabrica_test_part'
    current_q = np.zeros(7, dtype=float)
    terminal_seed_q = np.asarray([0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    terminal_target_q = np.asarray([0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    current_pose = {
        'position': np.asarray([0.02, 0.0, 0.0], dtype=float),
        'orientation': target_pose['orientation'].copy(),
    }
    solve_calls = []

    monkeypatch.setattr(adapter, '_target_pose', lambda **_kwargs: target_pose)
    monkeypatch.setattr(adapter, '_locked_target_pose', lambda **kwargs: kwargs['target_pose'])
    monkeypatch.setattr(adapter, '_current_robot_pose', lambda **_kwargs: current_pose)
    monkeypatch.setattr(adapter, '_current_tcp_pose', lambda **kwargs: kwargs['current_pose'])
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_args: current_q.copy())
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_kwargs: current_q.copy())
    monkeypatch.setattr(adapter, '_ik_target_pose', lambda **kwargs: kwargs['target_pose'])

    def solve_ik(**kwargs):
        solve_calls.append(kwargs)
        return terminal_target_q.copy()

    monkeypatch.setattr(adapter, '_solve_ik', solve_ik)
    monkeypatch.setattr(adapter, '_continuous_command_q', lambda **kwargs: kwargs['command_q'])
    monkeypatch.setattr(adapter, '_remember_arm_command', lambda *_args: None)
    adapter._pickup_terminal_ik_seeds[(id(task), robot_name, object_name)] = terminal_seed_q.copy()

    ready, action, detail = adapter._close_pose_gate_action(
        phase_key=phase_key,
        task=task,
        robot_name=robot_name,
        spec={
            'object': object_name,
            'unwrap_revolute_joints': False,
            'close_gate_lock_terminal_ik_target': True,
            'close_gate_terminal_ik_position_window': 0.04,
            'close_gate_terminal_ik_orientation_window': 0.10,
            'close_gate_pickup_terminal_seed_tolerance': 0.05,
            'close_gate_guard_ik_branch_jump': True,
            'ik_branch_jump_limit': 0.18,
            'close_gate_max_joint_step': 0.01,
            'close_position_tolerance': 0.005,
            'limit_command_to_measured_state': False,
        },
        tracked_robots={},
        tracked_objects={},
    )

    assert ready is False
    np.testing.assert_allclose(solve_calls[0]['warm_start'], terminal_seed_q)
    assert detail['terminal_ik_locked'] is True
    assert detail['pickup_terminal_seed_match'] is True
    assert detail['branch_jump_bypassed_for_pickup_terminal_seed'] is True
    np.testing.assert_allclose(action['arm_joint_controller'][0][0], 0.01)


def test_shared_atomic_skill_adapter_preserves_six_and_seven_dof_commands():
    ur5e_q = np.arange(6, dtype=float)
    franka_q = np.arange(7, dtype=float)

    np.testing.assert_array_equal(
        AssemblyAtomicSkillAdapter._coerce_arm_q(ur5e_q, bound_revolute=False),
        ur5e_q,
    )
    np.testing.assert_array_equal(
        AssemblyAtomicSkillAdapter._coerce_arm_q(franka_q, bound_revolute=False),
        franka_q,
    )


def test_prealign_waits_for_measured_joint_convergence_past_fixed_step_budget(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    measured_q = np.zeros(7, dtype=float)
    task = SimpleNamespace(phase='part_move_above', phase_step_counter=240, step_counter=500)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_args: measured_q.copy())
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_kwargs: measured_q.copy())

    action = adapter._prealign_action(
        phase_key=('prealign-wait',),
        task=task,
        robot_name='franka_left',
        target_pose={
            'position': np.zeros(3, dtype=float),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        },
        spec={
            'prealign_steps': 240,
            'prealign_joint_positions': np.ones(7).tolist(),
            'prealign_max_joint_step': 0.1,
            'prealign_until_converged': True,
            'prealign_joint_position_tolerance': 0.05,
            'prealign_timeout_steps': 1200,
            'unwrap_revolute_joints': False,
        },
    )

    assert '__local_skill_failure__' not in action
    assert np.max(np.asarray(action['arm_joint_controller'][0], dtype=float)) > 0.0


def test_prealign_seeds_approach_ik_from_physical_grasp_endpoint(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    reference_q = np.zeros(7, dtype=float)
    terminal_q = np.full(7, 0.4, dtype=float)
    approach_q = terminal_q + 0.1
    calls = []
    task = SimpleNamespace(phase='part_move_above', phase_step_counter=0, step_counter=0)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_args: reference_q.copy())
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_kwargs: reference_q.copy())

    def solve_ik(**kwargs):
        calls.append(kwargs)
        return terminal_q.copy() if len(calls) == 1 else approach_q.copy()

    monkeypatch.setattr(adapter, '_solve_ik', solve_ik)
    target_pose = {
        'position': np.asarray([0.4, -0.2, 1.2]),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    spec = {
        'prealign_steps': 240,
        'prealign_target_ik': True,
        'prealign_terminal_ik_seed': True,
        'offset': [0.0, 0.0, 0.1],
        'offset_frame': 'world',
        'object': 'fabrica_test_part',
        'unwrap_revolute_joints': False,
    }

    adapter._prealign_action(
        phase_key=('terminal-seeded-prealign',),
        task=task,
        robot_name='franka_left',
        target_pose=target_pose,
        spec=spec,
    )

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0]['target_pose']['position'], [0.4, -0.2, 1.1])
    np.testing.assert_allclose(calls[0]['warm_start'], reference_q)
    np.testing.assert_allclose(calls[1]['target_pose']['position'], target_pose['position'])
    np.testing.assert_allclose(calls[1]['warm_start'], terminal_q)
    assert calls[1]['spec']['require_warm_start_ik'] is True
    assert calls[1]['spec']['warm_start_ik_only'] is True
    np.testing.assert_allclose(
        adapter._pickup_terminal_ik_seeds[(id(task), 'franka_left', 'fabrica_test_part')],
        terminal_q,
    )


def test_distant_locked_transport_target_falls_back_to_continuous_cartesian(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    phase_key = ('transport-fallback',)
    reference_q = np.zeros(7, dtype=float)
    monkeypatch.setattr(adapter, '_solve_ik', lambda **_kwargs: np.full(7, 3.0))

    result = adapter._locked_transport_ik_target(
        phase_key=phase_key,
        task=SimpleNamespace(step_counter=10, phase='transport_hover'),
        robot_name='franka_right',
        target_pose={
            'position': np.asarray([0.5, 0.0, 1.2]),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
        reference_q=reference_q,
        spec={
            'unwrap_revolute_joints': False,
            'locked_transport_fallback_joint_delta': 2.5,
        },
    )

    assert result is None
    assert phase_key in adapter._continuous_transport_ik_fallbacks
    assert phase_key not in adapter._locked_transport_ik_targets


def test_prealign_enters_cartesian_servo_only_after_measured_joint_convergence(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    measured_q = np.full(7, 0.98, dtype=float)
    task = SimpleNamespace(phase='part_move_above', phase_step_counter=320, step_counter=580)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_args: measured_q.copy())
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_kwargs: measured_q.copy())

    action = adapter._prealign_action(
        phase_key=('prealign-converged',),
        task=task,
        robot_name='franka_left',
        target_pose={
            'position': np.zeros(3, dtype=float),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        },
        spec={
            'prealign_steps': 240,
            'prealign_joint_positions': np.ones(7).tolist(),
            'prealign_until_converged': True,
            'prealign_joint_position_tolerance': 0.05,
            'prealign_timeout_steps': 1200,
            'unwrap_revolute_joints': False,
        },
    )

    assert action is None
    monkeypatch.setattr(
        adapter,
        '_current_arm_q',
        lambda *_args: pytest.fail('converged prealignment must not restart or reread joint state'),
    )
    assert (
        adapter._prealign_action(
            phase_key=('prealign-converged',),
            task=task,
            robot_name='franka_left',
            target_pose={},
            spec={
                'prealign_steps': 240,
                'prealign_until_converged': True,
            },
        )
        is None
    )


def test_prealign_timeout_returns_explicit_local_skill_failure(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    measured_q = np.zeros(7, dtype=float)
    task = SimpleNamespace(phase='part_move_above', phase_step_counter=1200, step_counter=1460)
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_args: measured_q.copy())
    monkeypatch.setattr(adapter, '_command_reference_q', lambda **_kwargs: measured_q.copy())

    action = adapter._prealign_action(
        phase_key=('prealign-timeout',),
        task=task,
        robot_name='franka_left',
        target_pose={
            'position': np.zeros(3, dtype=float),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        },
        spec={
            'name': 'ur5e_move_above_part',
            'require_success': True,
            'prealign_steps': 240,
            'prealign_joint_positions': np.ones(7).tolist(),
            'prealign_until_converged': True,
            'prealign_joint_position_tolerance': 0.05,
            'prealign_timeout_steps': 1200,
            'unwrap_revolute_joints': False,
        },
    )

    assert action['__local_skill_failure__'] is True
    assert action['reason'] == 'prealign_timeout'
    assert action['diagnostics']['joint_error'] == pytest.approx(1.0)


def test_prealign_without_convergence_policy_keeps_fixed_step_behavior(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(phase_step_counter=240)
    monkeypatch.setattr(
        adapter,
        '_current_arm_q',
        lambda *_args: pytest.fail('fixed-step prealignment should stop before reading joint state'),
    )

    action = adapter._prealign_action(
        phase_key=('legacy-prealign',),
        task=task,
        robot_name='ur5e_left',
        target_pose={},
        spec={'prealign_steps': 240},
    )

    assert action is None


def test_shared_atomic_skill_adapter_reseeds_a_stalled_cartesian_ik_solution():
    adapter = AssemblyAtomicSkillAdapter({})
    reference_q = np.zeros(7, dtype=float)
    current_pose = {
        'position': np.zeros(3, dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    command_target_pose = {
        'position': np.asarray([0.001, 0.0, 0.0], dtype=float),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    spec = {'cartesian_servo': True, 'cartesian_position_step': 0.001}

    assert adapter._ik_solution_stalled(
        ik_result=reference_q,
        measured_q=reference_q,
        warm_start_q=reference_q,
        current_pose=current_pose,
        command_target_pose=command_target_pose,
        spec=spec,
    )
    assert not adapter._ik_solution_stalled(
        ik_result=reference_q + 0.01,
        measured_q=reference_q,
        warm_start_q=reference_q,
        current_pose=current_pose,
        command_target_pose=command_target_pose,
        spec=spec,
    )

    lagging_measured_q = reference_q - 0.02
    assert adapter._ik_solution_stalled(
        ik_result=reference_q,
        measured_q=lagging_measured_q,
        warm_start_q=reference_q,
        current_pose=current_pose,
        command_target_pose=command_target_pose,
        spec=spec,
    )


def test_warm_start_only_ik_temporarily_excludes_default_cspace_seeds():
    class RawSolver:
        def __init__(self):
            self.default_seeds = [np.full(7, 0.5)]
            self.seeds_seen_during_solve = None

        def get_default_cspace_seeds(self):
            return self.default_seeds

        def set_default_cspace_seeds(self, seeds):
            self.default_seeds = list(seeds)

        def compute_inverse_kinematics(self, _frame, _position, _orientation, *, warm_start, **_kwargs):
            self.seeds_seen_during_solve = list(self.default_seeds)
            return np.asarray(warm_start, dtype=float) + 0.01, True

    raw_solver = RawSolver()
    solver_wrapper = SimpleNamespace(
        set_robot_base_pose=lambda **_kwargs: None,
        get_kinematics_solver=lambda: raw_solver,
        get_end_effector_frame=lambda: 'panda_hand',
    )
    ik_controller = SimpleNamespace(
        _kinematics_solver=solver_wrapper,
        _robot_scale=np.ones(3),
        get_ik_base_world_pose=lambda: (np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
    )
    task = SimpleNamespace(
        robots={'franka_left': SimpleNamespace(controllers={'arm_ik_controller': ik_controller})}
    )
    warm_start = np.zeros(7)

    result = AssemblyAtomicSkillAdapter({})._solve_ik(
        task=task,
        robot_name='franka_left',
        target_pose={
            'position': np.asarray([0.5, 0.0, 1.0]),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
        warm_start=warm_start,
        spec={
            'use_command_warm_start': True,
            'require_warm_start_ik': True,
            'warm_start_ik_only': True,
        },
    )

    np.testing.assert_allclose(result, warm_start + 0.01)
    assert raw_solver.seeds_seen_during_solve == []
    np.testing.assert_allclose(raw_solver.default_seeds[0], np.full(7, 0.5))


def test_bounded_franka_joint_targets_are_not_unwrapped_past_joint_limits():
    target_q = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.2])
    reference_q = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.8])

    bounded = AssemblyAtomicSkillAdapter._joint_target_near_reference(
        target_q=target_q,
        reference_q=reference_q,
        spec={'unwrap_revolute_joints': False},
    )
    continuous = AssemblyAtomicSkillAdapter._joint_target_near_reference(
        target_q=target_q,
        reference_q=reference_q,
        spec={'unwrap_revolute_joints': True},
    )

    assert bounded[-1] == pytest.approx(-2.2)
    assert continuous[-1] > np.pi


def test_previous_target_branch_guard_falls_back_to_current_reference_on_phase_entry():
    adapter = AssemblyAtomicSkillAdapter({})
    phase_key = (1, 2, 3, 'franka_left', 'move_part_to_target')
    reference_q = np.arange(7, dtype=float)
    spec = {'ik_branch_jump_reference_mode': 'previous_target'}

    np.testing.assert_array_equal(
        adapter._ik_branch_reference_q(
            phase_key=phase_key,
            reference_q=reference_q,
            spec=spec,
        ),
        reference_q,
    )

    previous_target_q = reference_q + 0.25
    adapter._last_targets[phase_key] = {'target_q': previous_target_q}
    np.testing.assert_array_equal(
        adapter._ik_branch_reference_q(
            phase_key=phase_key,
            reference_q=reference_q,
            spec=spec,
        ),
        previous_target_q,
    )


def test_previous_target_branch_guard_can_accept_bounded_initial_endpoint():
    adapter = AssemblyAtomicSkillAdapter({})
    phase_key = (1, 2, 3, 'franka_right', 'move_above_part')
    reference_q = np.arange(6, dtype=float)
    spec = {
        'ik_branch_jump_reference_mode': 'previous_target',
        'allow_initial_ik_branch_jump': True,
    }

    assert adapter._ik_branch_reference_q(
        phase_key=phase_key,
        reference_q=reference_q,
        spec=spec,
    ) is None

    previous_target_q = reference_q + 0.25
    adapter._last_targets[phase_key] = {'target_q': previous_target_q}
    np.testing.assert_array_equal(
        adapter._ik_branch_reference_q(
            phase_key=phase_key,
            reference_q=reference_q,
            spec=spec,
        ),
        previous_target_q,
    )


def test_joint_space_park_uses_bounded_commands_and_stable_completion(monkeypatch):
    adapter = AssemblyAtomicSkillAdapter({})
    target_q = np.asarray([0.0, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7], dtype=float)
    measured_q = np.zeros(7, dtype=float)
    completed = []
    task = SimpleNamespace(
        phase_index=4,
        phase_entry_step=100,
        phase_step_counter=0,
        mark_local_skill_complete=lambda **kwargs: completed.append(kwargs),
    )
    monkeypatch.setattr(adapter, '_current_arm_q', lambda *_args: measured_q.copy())
    monkeypatch.setattr(adapter, '_gripper_command_value', lambda **_kwargs: 1.0)

    spec = {
        'joint_positions': target_q.tolist(),
        'reference_mode': 'current',
        'max_joint_step': 0.02,
        'max_command_joint_step': 0.02,
        'joint_position_tolerance': 0.025,
        'joint_target_stable_steps': 3,
        'gripper_command': 'open',
    }
    phase_key = (id(task), 4, 100, 'franka_right', 'move_arm_to_joint_positions')
    action = adapter._move_arm_to_joint_positions_action(
        phase_key=phase_key,
        task=task,
        robot_name='franka_right',
        skill_name='move_arm_to_joint_positions',
        spec=spec,
    )

    command_q = np.asarray(action['arm_joint_controller'][0], dtype=float)
    assert np.max(np.abs(command_q - measured_q)) <= 0.02 + 1e-12
    assert action['gripper_controller'] == [1.0]
    assert not completed

    measured_q[:] = target_q
    for step in range(3):
        task.phase_step_counter = step + 1
        adapter._move_arm_to_joint_positions_action(
            phase_key=phase_key,
            task=task,
            robot_name='franka_right',
            skill_name='move_arm_to_joint_positions',
            spec=spec,
        )
    assert len(completed) == 1
    assert completed[0]['detail']['joint_error'] == 0.0


def test_close_recenter_latches_direction_across_contact_face_jitter():
    state = {}
    spec = {
        'close_gate_recenter_single_finger_contact': True,
        'close_gate_recenter_contact_distance': 0.008,
        'close_gate_recenter_min_gap_imbalance': 0.0005,
        'close_gate_recenter_stable_steps': 1,
        'close_gate_recenter_step': 0.00075,
        'close_gate_recenter_gap_gain': 0.5,
        'close_gate_recenter_max_step': 0.008,
        'close_gate_recenter_max_offset': 0.025,
    }

    def detail(*, axis, local_point):
        return {
            'contact_detail': {
                'contact_metrics': {
                    'contact_box_orientation': [1.0, 0.0, 0.0, 0.0],
                    'left_finger': {
                        'local_contact': {
                            'best_surface_gap': 0.003,
                            'best_axis': axis,
                            'local_point': local_point,
                        }
                    },
                    'right_finger': {
                        'local_contact': {
                            'best_surface_gap': 0.025,
                            'best_axis': axis,
                            'local_point': local_point,
                        }
                    },
                }
            }
        }

    first = AssemblyAtomicSkillAdapter._update_close_recenter_offset(
        state=state,
        close_detail=detail(axis='x', local_point=[0.01, 0.0, 0.0]),
        spec=spec,
        close_ready=False,
    )
    state['recenter_target_ready'] = True
    second = AssemblyAtomicSkillAdapter._update_close_recenter_offset(
        state=state,
        close_detail=detail(axis='y', local_point=[0.0, -0.01, 0.0]),
        spec=spec,
        close_ready=False,
    )

    first_offset = np.asarray(first['offset_world'], dtype=float)
    second_offset = np.asarray(second['offset_world'], dtype=float)
    assert first['direction_latched'] is False
    assert second['direction_latched'] is True
    assert second['axis'] == 'x'
    assert second_offset[0] > first_offset[0] > 0.0
    np.testing.assert_allclose(second_offset[1:], 0.0, atol=1e-12)


def test_close_recenter_uses_physical_finger_axis_before_sampled_object_face():
    state = {}
    spec = {
        'close_gate_recenter_single_finger_contact': True,
        'close_gate_recenter_contact_distance': 0.008,
        'close_gate_recenter_min_gap_imbalance': 0.0005,
        'close_gate_recenter_stable_steps': 1,
        'close_gate_recenter_step': 0.00075,
        'close_gate_recenter_gap_gain': 0.5,
        'close_gate_recenter_max_step': 0.008,
        'close_gate_recenter_max_offset': 0.025,
    }
    close_detail = {
        'contact_detail': {
            'contact_metrics': {
                'contact_box_center': [0.0, 0.006, 0.0],
                'contact_box_orientation': [1.0, 0.0, 0.0, 0.0],
                'left_finger': {
                    'best_sample_position': [0.0, 0.04, 0.0],
                    'local_contact': {
                        'best_surface_gap': 0.003,
                        'best_axis': 'x',
                        'local_point': [-0.01, 0.0, 0.0],
                    },
                },
                'right_finger': {
                    'best_sample_position': [0.0, -0.04, 0.0],
                    'local_contact': {
                        'best_surface_gap': 0.025,
                        'best_axis': 'z',
                        'local_point': [0.0, 0.0, 0.01],
                    },
                },
            }
        }
    }

    result = AssemblyAtomicSkillAdapter._update_close_recenter_offset(
        state=state,
        close_detail=close_detail,
        spec=spec,
        close_ready=False,
    )

    assert result['axis'] == 'finger_pair'
    assert result['finger_pair_center_error'] == pytest.approx(0.006)
    assert result['offset_world'][1] > 0.0
    np.testing.assert_allclose(
        [result['offset_world'][0], result['offset_world'][2]],
        0.0,
        atol=1e-12,
    )
