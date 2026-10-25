"""SO(3) governor and opt-in runtime wiring; no simulator required."""
import copy
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from toolkits.factory_dual_franka_assembly.physical_rotation_command import (
    bounded_rotation_command, rotation_vector,
)
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter


IDENTITY = np.array([1., 0., 0., 0.])
LIMITS = dict(speed=.1, acceleration=.3, lookahead=.009)


def quaternion(angle, axis=(0., 1., 0.)):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]


def command(state, measured, target, step, dt=1 / 240, **limits):
    return bounded_rotation_command(state=state, measured_orientation=measured,
        target_orientation=target, initial_orientation=IDENTITY, step=step, dt=dt,
        **{**LIMITS, **limits})


@pytest.mark.parametrize('hz', [80, 240])
def test_stalled_wrist_brakes_without_accumulating_or_resetting(hz):
    state = None
    for step in range(hz * 5):
        orientation, state, detail = command(state, IDENTITY, quaternion(1.), step, dt=1 / hz)
        assert detail['angular_speed'] <= .1 + 1e-10
        assert detail['angular_acceleration'] <= .3 + 1e-10
        assert detail['orientation_lead'] <= .009 + 1e-10
    assert np.linalg.norm(rotation_vector(orientation, IDENTITY)) == pytest.approx(.009, abs=1e-7)
    assert detail['angular_speed'] < 1e-6


@pytest.mark.parametrize('hz', [80, 240])
def test_large_turn_tracks_smoothly_and_stops_at_goal(hz):
    measured, state = IDENTITY.copy(), None
    for step in range(hz * 15):
        orientation, state, detail = command(state, measured, quaternion(1.), step, dt=1 / hz)
        measured = orientation.copy()  # Ideal tracking, not reference rebasing.
        assert detail['angular_speed'] <= .1 + 1e-10
        assert detail['angular_acceleration'] <= .3 + 1e-10
    assert np.linalg.norm(rotation_vector(orientation, quaternion(1.))) < 1e-7
    assert detail['angular_speed'] < 1e-6


def test_target_reversal_brakes_instead_of_flipping_velocity():
    measured, state, velocities = IDENTITY.copy(), None, []
    for step in range(600):
        target = quaternion(.5 if step < 120 else -.5)
        orientation, state, detail = command(state, measured, target, step)
        measured = orientation
        velocities.append(state['velocity'][1])
        assert detail['angular_acceleration'] <= .3 + 1e-10
    assert velocities[120] > 0 and min(velocities[120:]) < 0


def test_quaternion_sign_and_near_pi_shortest_arc():
    target = quaternion(math.pi - 1e-8, (1., 2., 3.))
    a, _, _ = command(None, IDENTITY, target, 0)
    b, _, _ = command(None, -IDENTITY, -target, 0)
    assert np.linalg.norm(rotation_vector(a, b)) < 1e-12


def test_rejected_proposal_does_not_mutate_state_or_integrate_skipped_time():
    _, state, _ = command(None, IDENTITY, quaternion(1.), 0)
    original = copy.deepcopy(state)
    command(state, IDENTITY, quaternion(1.), 1)
    assert state == original
    orientation, resumed, detail = command(state, IDENTITY, quaternion(1.), 100)
    assert detail['restarted_after_gap']
    assert np.linalg.norm(rotation_vector(orientation, state['orientation'])) <= .3 / 240**2 + 1e-12
    with pytest.raises(ValueError, match='new simulation step'):
        command(resumed, IDENTITY, quaternion(1.), 100)


@pytest.mark.parametrize('key', ['dt', 'speed', 'acceleration', 'lookahead'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_rejects_invalid_time_and_limits(key, value):
    with pytest.raises(ValueError):
        command(None, IDENTITY, IDENTITY, 0, **{key: value})


def test_invalid_quaternion_and_external_tracking_jump_fail_closed():
    with pytest.raises(ValueError, match='quaternion'):
        command(None, [0., 0., 0., 0.], IDENTITY, 0)
    _, state, _ = command(None, IDENTITY, quaternion(1.), 0)
    with pytest.raises(ValueError, match='rotation_tracking_boundary'):
        command(state, quaternion(.1), quaternion(1.), 1)


@pytest.fixture
def runtime():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    saved = {'object': 'part', 'pose': {'position': np.zeros(3), 'orientation': IDENTITY.copy()},
             'arm_q': np.zeros(6)}
    task = NS(step_counter=10, phase='carry_a', _ur5e_contact_hold_commands={'right': saved},
              _contact_physics_dt=lambda: 1 / 240)
    adapter._object_name_from_spec = lambda _: 'part'
    adapter._current_robot_pose = lambda **_: saved['pose']
    adapter._grasp_command_fk_pose = lambda **_: saved['pose']
    spec = dict(physical_rotation_governor=True, physical_rotation_speed=.1,
                physical_rotation_acceleration=.3, physical_rotation_lookahead=.01)
    return adapter, task, saved, spec


def propose(runtime):
    adapter, task, saved, spec = runtime
    return adapter._physical_rotation_target(task=task, robot_name='right', spec=spec,
        current_pose=saved['pose'], target_pose={'orientation': quaternion(1.)}, command_pose=saved['pose'])


def test_duplicate_runtime_calls_are_idempotent_and_state_only_commits_on_accept(runtime):
    adapter, task, saved, spec = runtime
    first, second = propose(runtime), propose(runtime)
    np.testing.assert_array_equal(first['orientation'], second['orientation'])
    assert 'rotation_command_state' not in saved
    assert adapter._commit_physical_rotation(task=task, robot_name='right', spec=spec,
        saved=saved, command_pose=first, published_pose=first)
    assert saved['rotation_command_state']['step'] == 10
    task.step_counter += 1
    task.phase = 'carry_b'  # No phase-boundary reset or fresh warm start.
    propose(runtime)
    assert saved['rotation_command_pending']['detail']['angular_speed'] == pytest.approx(2 * .3 / 240)


def test_modified_ik_waypoint_and_published_fk_outside_lookahead_cannot_commit(runtime):
    adapter, task, saved, spec = runtime
    candidate = propose(runtime)
    for requested, published in [({'orientation': quaternion(.05)}, candidate),
                                  (candidate, {'orientation': quaternion(.011)}),
                                  (candidate, {'orientation': quaternion(.0001)})]:
        assert not adapter._commit_physical_rotation(task=task, robot_name='right', spec=spec,
            saved=saved, command_pose=requested, published_pose=published)
        assert 'rotation_command_state' not in saved


def test_initial_reference_uses_published_joint_fk_not_previous_ik_target(runtime):
    adapter, task, saved, spec = runtime
    initial = quaternion(.0007)
    adapter._grasp_command_fk_pose = lambda **_: {'orientation': initial}
    candidate = propose(runtime)
    assert np.linalg.norm(rotation_vector(candidate['orientation'], initial)) == pytest.approx(.3 / 240**2)
    pending = saved['rotation_command_pending']
    np.testing.assert_allclose(pending['previous_published_orientation'], initial)
    assert adapter._commit_physical_rotation(task=task, robot_name='right', spec=spec,
        saved=saved, command_pose=candidate, published_pose=candidate)
    assert saved['rotation_published_state']['step'] == task.step_counter


def test_legacy_completion_has_no_governor_dependency():
    assert UR5eAssemblyAtomicSkillAdapter._physical_rotation_at_rest(NS(), 'right', {})


def test_ik_backtracking_preserves_governed_orientation_only_in_opt_in_mode():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current = {'position': np.zeros(3), 'orientation': IDENTITY}
    target = {'position': np.ones(3), 'orientation': quaternion(.01)}
    for spec in ({}, {'physical_rotation_governor': True}):
        candidate = adapter._ik_backtrack_pose(current_pose=current, target_pose=target, ratio=.25, spec=spec)
        np.testing.assert_allclose(candidate['position'], [.25, .25, .25])
        angle = .01 if spec else .0025
        assert np.linalg.norm(rotation_vector(candidate['orientation'], IDENTITY)) == pytest.approx(angle, abs=1e-8)


def test_governed_ik_uses_precise_local_solve_without_global_branch_reseed():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    wrapper = NS(set_robot_base_pose=lambda **_: None)
    controller = NS(_kinematics_solver=wrapper, _robot_scale=1.,
                    get_ik_base_world_pose=lambda: (np.zeros(3), IDENTITY))
    task = NS(robots={'right': NS(controllers={'arm_ik_controller': controller})})
    calls = []
    adapter._local_grasp_ik = lambda **kw: calls.append(kw) or np.zeros(6)
    pose = {'position': np.zeros(3), 'orientation': quaternion(.00001)}
    spec = dict(physical_rotation_governor=True, max_command_joint_step=.008,
                ik_position_tolerance=5e-6, ik_orientation_tolerance=.001)
    result = adapter._solve_ik(task=task, robot_name='right', target_pose=pose,
                               warm_start=np.zeros(6), spec=spec)
    np.testing.assert_array_equal(result, np.zeros(6))
    assert calls[0]['spec']['ik_orientation_tolerance'] == 1e-8
    assert calls[0]['spec']['ik_position_tolerance'] == 1e-6
    assert calls[0]['joint_radius'] == .008
    assert adapter._solve_ik(task=task, robot_name='right', target_pose=pose, spec=spec) is None
    assert len(calls) == 1


def test_governed_branch_recovery_does_not_switch_to_measured_joint_seed():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    calls = []
    adapter._solve_ik = lambda **kw: calls.append(kw)
    current = {'position': np.zeros(3), 'orientation': IDENTITY}
    target = {'position': np.ones(3), 'orientation': quaternion(.01)}
    reference, measured = np.zeros(6), np.ones(6) * .004
    spec = dict(physical_rotation_governor=True, ik_branch_backtrack_ratios=(.5,),
                ik_branch_translation_only_recovery=True)
    assert adapter._recover_ik_branch_jump(task=NS(), robot_name='right', skill_name='carry', spec=spec,
        current_pose=current, command_target_pose=target, reference_q=reference,
        branch_reference_q=reference, measured_q=measured) is None
    assert len(calls) == 2
    for call in calls:
        np.testing.assert_array_equal(call['warm_start'], reference)
        np.testing.assert_array_equal(call['target_pose']['orientation'], target['orientation'])
