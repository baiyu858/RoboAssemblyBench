"""Empty-arm trajectory references must accumulate without bypassing grasp guards."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter as Adapter


def pose(p):
    return {'position': np.array(p, dtype=float), 'orientation': np.array([1., 0., 0., 0.])}


@pytest.mark.parametrize('stage,position', [('lift', [0., 0., 1.3]), ('orient', [0., 0., 1.2]),
    ('transfer', [.5, .2, 1.2]), ('lower', [.5, .2, 1.1]), ('approach', [.5, .2, 1.1])])
def test_stage_targets_separate_height_rotation_and_xy(stage, position):
    target = pose([.5, .2, 1.1]); target['orientation'] = np.array([0., 1., 0., 0.])
    result = Adapter._free_space_stage_target(current=pose([0., 0., 1.2]), target=target,
        object_pose=pose([.4, .2, 1.]), spec={'free_space_stage': stage,
                                            'coarse_tcp_height_above_base_origin_m': .3})
    np.testing.assert_allclose(result['position'], position)
    np.testing.assert_allclose(result['orientation'], [1., 0., 0., 0.] if stage == 'lift' else target['orientation'])
    np.testing.assert_array_equal(target['position'], [.5, .2, 1.1])


def test_lift_never_lowers_an_already_high_arm():
    result = Adapter._free_space_stage_target(current=pose([0., 0., 1.5]), target=pose([1., 0., 1.]),
        object_pose=pose([0., 0., 1.]), spec={'free_space_stage': 'lift',
                                             'coarse_tcp_height_above_base_origin_m': .3})
    assert result['position'][2] == 1.5


def runtime():
    adapter = Adapter({})
    measured = np.zeros(6)
    task = NS(step_counter=0, phase='free_transfer', _attachments={}, _contact_physics_dt=lambda: 1/240)
    adapter._current_arm_q = lambda *_: measured.copy()
    adapter._current_robot_pose = lambda **_: pose(measured[:3])
    adapter._grasp_command_fk_pose = lambda **k: pose(np.array(k['command_q'])[:3])
    adapter._gripper_command_value = lambda **_: 1.
    adapter._maybe_mark_complete = lambda **_: None
    adapter._solve_ik = lambda **_: pytest.fail('Empty motion must not request a global IK branch')
    def local(**kwargs):
        result = kwargs['reference_q'].copy()
        result[:3] = kwargs['target_pose']['position']
        return result
    adapter._local_grasp_ik = local
    spec = dict(name='ur5e_move_above_part', free_space_trajectory=True, free_space_stage='transfer',
        require_success=True, max_command_joint_step=.006, max_command_wrist_joint_step=.006,
        free_space_linear_speed=.06, free_space_angular_speed=.35,
        free_space_joint_speed=.6, free_space_joint_acceleration=1.2,
        free_space_position_lookahead=.015, free_space_orientation_lookahead=.06,
        free_space_joint_lookahead=.12)
    return adapter, task, measured, spec


def test_half_metre_transfer_advances_published_reference_despite_50ms_drive_lag():
    adapter, task, measured, spec = runtime()
    previous = np.zeros(6); velocity = np.zeros(6)
    for step in range(240 * 12):
        task.step_counter = step
        action = adapter._free_space_motion_action(phase_key=('transfer',), task=task, robot_name='right',
            spec=spec, target_pose=pose([.54, 0., 0.]), tracked_robots={}, tracked_objects={})
        assert '__local_skill_failure__' not in action
        command = np.array(action['arm_joint_controller'][0])
        next_velocity = (command-previous)*240
        assert np.max(np.abs(next_velocity)) <= .6 + 1e-9
        assert np.max(np.abs(next_velocity-velocity))*240 <= 1.2 + 1e-8
        assert np.linalg.norm(command[:3]-measured[:3]) <= .015 + 1e-10
        measured += (1-np.exp(-1/240/.05))*(command-measured)
        previous, velocity = command, next_velocity
    assert abs(measured[0]-.54) < .001
    assert np.max(np.abs(velocity)) < .005


def test_loaded_arm_is_rejected_before_ik_or_state_capture():
    adapter, task, measured, spec = runtime()
    task._attachments['base'] = {'robot_name': 'right'}
    result = adapter._free_space_motion_action(phase_key=('transfer',), task=task, robot_name='right',
        spec=spec, target_pose=pose([.54, 0., 0.]), tracked_robots={}, tracked_objects={})
    assert result['reason'] == 'free_space_arm_is_loaded'
    assert adapter._free_space_commands == {}


def test_entry_reanchors_stale_idle_cache_without_changing_measured_joints():
    adapter, task, measured, spec = runtime()
    adapter._remember_arm_command(task, 'right', np.array([.3, 0., 0., 0., 0., 0.]))
    result = adapter._free_space_motion_action(phase_key=('transfer',), task=task, robot_name='right',
        spec=spec, target_pose=pose([.54, 0., 0.]), tracked_robots={}, tracked_objects={})
    assert '__local_skill_failure__' not in result
    command = np.array(result['arm_joint_controller'][0])
    np.testing.assert_array_equal(measured, np.zeros(6))
    np.testing.assert_allclose(adapter._last_command_q(task=task, robot_name='right'), command)
    assert 0. < command[0] <= 1.2 / 240**2


def test_adjacent_free_stages_stop_before_handoff_and_preserve_acceleration_limit():
    adapter, task, measured, spec = runtime()
    completed = []

    def complete(**kwargs):
        if np.linalg.norm(kwargs['current_pose']['position']-kwargs['target_pose']['position']) < .0002:
            completed.append(True)

    adapter._maybe_mark_complete = complete
    previous = np.zeros(6)
    previous_velocity = np.zeros(6)
    for index, target in enumerate(([.03, 0., 0.], [.03, .03, 0.])):
        task.phase = f'transfer_{index}'
        completed.clear()
        for _ in range(1000):
            task.step_counter += 1
            action = adapter._free_space_motion_action(phase_key=(task.phase,), task=task, robot_name='right',
                spec=spec, target_pose=pose(target), tracked_robots={}, tracked_objects={})
            assert '__local_skill_failure__' not in action
            command = np.array(action['arm_joint_controller'][0])
            velocity = (command-previous)*240
            assert np.max(np.abs(velocity-previous_velocity))*240 <= 1.2 + 1e-8
            measured += (1-np.exp(-1/240/.05))*(command-measured)
            previous, previous_velocity = command, velocity
            if completed:
                assert np.max(np.abs(velocity)) <= 1e-10
                break
        else:
            pytest.fail('Free-space stage failed to settle before handoff')
