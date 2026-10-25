"""Weak-side carrying control, motion gating and legacy isolation; no simulator."""
import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from toolkits.factory_dual_franka_assembly.physical_hold_force import (
    DEFAULTS, hold_force_command, hold_force_settings,
)
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter


CONFIG = {'enabled': True, **DEFAULTS}
LIMIT_AND_SLOW = {**CONFIG, 'soft_limit': 14., 'unload': 18.,
                  'soft_limit_action': 'limit_and_slow', 'soft_limit_motion_scale': .5}


def tick(state, forces=(8., 8.), *, step=None, config=None, **kwargs):
    return hold_force_command(state=state, config=CONFIG if config is None else config,
        forces=forces, probes_valid=kwargs.pop('probes_valid', True),
        gripper_q=kwargs.pop('gripper_q', state.get('contact_command_closure', .4)),
        open_q=kwargs.pop('open_q', 0.), closed_q=kwargs.pop('closed_q', .8),
        min_openness=.08, step=state.get('step', -1) + 1 if step is None else step,
        dt=kwargs.pop('dt', 1 / 240), position_slip=kwargs.pop('position_slip', 0.), **kwargs)


def test_preload_uses_weak_side_and_latches_until_target_not_only_low():
    state = {}
    first = tick(state, (5., 8.))
    assert first['command_q'] > .4
    assert first['mode'] == 'preload' and not first['motion_ready']
    # The 6 N trigger does not end recovery at 6.1 N.
    for _ in range(40):
        result = tick(state, (6.5, 8.))
        assert result['mode'] == 'preload' and not result['motion_ready']
    for _ in range(100):
        result = tick(state, (8.5, 8.5))
    assert result['mode'] == 'hold' and result['motion_ready']


def test_strong_side_ceiling_prevents_squeezing_weak_side_indefinitely():
    state = {}
    result = tick(state, (3., 10.))
    assert result['mode'] == 'force_limited'
    assert result['command_q'] == .4
    assert not result['motion_ready']


def test_raw_overload_is_not_hidden_by_filter_and_hard_limit_fails():
    state = {}
    for _ in range(30):
        tick(state)
    before = state['contact_command_closure']
    result = tick(state, (8., 14.))
    assert result['mode'] == 'unload'
    assert result['command_q'] < before and not result['motion_ready']
    result = tick(state, (8., 41.))
    assert result['failure'] == 'loaded_gripper_force_limit'
    assert not result['force_within_limit']


def test_single_low_force_sample_is_filtered_but_sustained_loss_triggers_preload():
    state = {}
    for _ in range(30):
        tick(state)
    result = tick(state, (5., 8.))
    assert result['mode'] == 'hold' and result['motion_ready']
    for _ in range(20):
        result = tick(state, (5., 8.))
    assert result['mode'] == 'preload' and not result['motion_ready']


def test_exact_target_eventually_ends_preload_despite_filter_roundoff():
    state = {}
    tick(state, (5., 8.))
    for _ in range(180):
        result = tick(state, (8., 8.))
    assert result['mode'] == 'hold' and result['motion_ready']


@pytest.mark.parametrize('forces,valid,error', [
    ((8., 0.), True, 'loaded_force_contact_missing'),
    ((8., float('nan')), True, 'loaded_force_probe_invalid'),
    ((8., 8.), False, 'loaded_force_probe_invalid'),
    ((float('nan'), 45.), False, 'loaded_gripper_force_limit'),
])
def test_bad_probes_or_missing_target_contact_never_add_force(forces, valid, error):
    result = tick({}, forces, probes_valid=valid)
    assert result['failure'] == error
    assert result['command_q'] == .4 and not result['motion_ready']


def test_same_step_is_idempotent_even_when_caller_reenters_publish():
    state = {}
    first = tick(state, (5., 8.), step=10)
    snapshot = copy.deepcopy(state)
    assert tick(state, (5., 8.), step=10) == first
    assert state == snapshot
    with pytest.raises(ValueError, match='previous simulation step'):
        tick(state, step=9)


def test_recovery_timeout_requires_new_grasp_instead_of_unbounded_pause():
    state = {}
    c = {**CONFIG, 'stable_steps': 2, 'recovery_timeout_steps': 5}
    for _ in range(4):
        assert tick(state, (3., 11.), config=c)['failure'] is None
    assert tick(state, (3., 11.), config=c)['failure'] == 'loaded_force_recovery_timeout'


def test_stability_window_and_smooth_resume_reset_on_unstable_force():
    state = {}
    for _ in range(23):
        assert not tick(state)['motion_ready']
    first = tick(state)
    assert first['motion_ready'] and 0 < first['motion_scale'] < .01
    scales = [tick(state)['motion_scale'] for _ in range(47)]
    assert scales == sorted(scales) and scales[-1] == 1.
    assert not tick(state, (8., 12.5))['motion_ready']
    assert state['resume_count'] == 0


def test_missed_steps_do_not_integrate_time_or_reuse_stability():
    state = {}
    for _ in range(30):
        tick(state)
    result = tick(state, (5., 8.), step=1000)
    assert result['command_q'] - .4 == pytest.approx(.025 / 240)
    assert not result['motion_ready'] and state['stable_steps'] == 0


@pytest.mark.parametrize('hz', [80, 240])
def test_joint_rate_lead_and_total_travel_remain_bounded(hz):
    state = {}
    previous = .4
    for _ in range(hz):
        r = tick(state, (5., 8.), dt=1 / hz, gripper_q=.4)
        assert 0 <= r['command_q'] - previous <= .025 / hz + 1e-12
        assert r['command_q'] <= .401 + 1e-12  # No windup against a stalled joint.
        previous = r['command_q']
    state = {}
    c = {**CONFIG, 'max_closing_travel': .0003}
    for _ in range(10):
        r = tick(state, (5., 8.), config=c)
        if r['failure']:
            break
    assert r['failure'] == 'loaded_force_travel_limit'
    assert r['closing_travel'] <= .0003 + 1e-12


def test_reversed_gripper_joint_direction_uses_same_force_policy():
    state = {}
    r = tick(state, (5., 8.), gripper_q=.4, open_q=.8, closed_q=0.)
    assert r['command_q'] < .4
    assert r['command_openness'] < .5


def test_sustained_slip_pauses_without_raising_force_target_or_recapturing():
    state = {}
    for _ in range(80):
        tick(state)
    original = state['contact_capture_closure']
    for i in range(55):
        r = tick(state, position_slip=(i + 1) * .00001)
    assert r['slipping'] and not r['motion_ready']
    assert r['mode'] == 'slip_hold' and r['command_q'] == original
    assert state['contact_capture_closure'] == original
    assert r['target_force'] == 8.


@pytest.mark.parametrize('key', list(DEFAULTS))
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True])
def test_bad_hold_settings_fail_early(key, value):
    with pytest.raises(ValueError, match='hold_force_control'):
        hold_force_settings({**CONFIG, key: value})


def test_bad_order_unknown_names_and_noninteger_counts_are_rejected():
    for change in [{'low': 8.}, {'target': 11.}, {'unload': 50.},
                   {'stable_steps': 2.5}, {'recovery_timeout_steps': 24}, {'typo': 6},
                   {'soft_limit_action': 'ignore'}, {'soft_limit_action': True},
                   {'soft_limit_motion_scale': 1.01}]:
        with pytest.raises(ValueError, match='hold_force_control'):
            hold_force_settings({**CONFIG, **change})


def test_new_policy_accepts_previous_failure_force_band_without_timeout():
    state = {}
    for _ in range(600):
        result = tick(state, (8.20, 10.73), config=LIMIT_AND_SLOW)
        assert result['failure'] is None
    assert result['mode'] == 'hold' and result['motion_ready']
    assert result['motion_scale'] == 1.
    assert result['command_q'] == .4  # Raising a ceiling does not command more force.


def test_mild_overload_relief_reduces_grip_command_without_pausing_established_motion():
    state = {}
    for _ in range(80):
        tick(state, config=LIMIT_AND_SLOW)
    before = state['contact_command_closure']
    result = tick(state, (8.5, 15.), config=LIMIT_AND_SLOW)
    assert result['mode'] == 'force_relief' and result['motion_ready']
    assert before - result['command_q'] == pytest.approx(.025 / 240)
    assert result['motion_scale'] == pytest.approx(.875)
    assert state['resume_count'] == 48 and state['paused_steps'] == 0


def test_mild_overload_does_not_hide_pending_weak_side_recovery():
    result = tick({}, (3., 15.), config=LIMIT_AND_SLOW)
    assert result['mode'] == 'force_limited'
    assert not result['motion_ready'] and result['motion_scale'] == 0.
    assert result['command_q'] == .4


def test_mild_overload_can_continue_in_hold_band_without_unloading_weak_side():
    state = {}
    for _ in range(80):
        tick(state, config=LIMIT_AND_SLOW)
    before = state['contact_command_closure']
    for _ in range(60):
        result = tick(state, (7., 15.), config=LIMIT_AND_SLOW)
        assert result['motion_ready'] and result['failure'] is None
    assert result['mode'] == 'force_limited' and result['command_q'] == before
    assert result['motion_scale'] == pytest.approx(.875)


def test_soft_relief_does_not_unload_when_raw_weak_force_dips_below_low():
    state = {}
    for _ in range(80):
        tick(state, (10., 10.), config=LIMIT_AND_SLOW)
    before = state['contact_command_closure']
    result = tick(state, (5., 15.), config=LIMIT_AND_SLOW)
    assert result['weak_force'] > 8.  # Filter has not yet followed the weak-side dip.
    assert result['mode'] == 'force_limited' and result['command_q'] == before


def test_new_policy_never_bypasses_slip_guard_while_force_limited():
    state = {}
    for _ in range(80):
        tick(state, config=LIMIT_AND_SLOW)
    before = state['contact_command_closure']
    for i in range(55):
        result = tick(state, (7., 15.), config=LIMIT_AND_SLOW, position_slip=(i + 1) * .00001)
    assert result['slipping'] and not result['motion_ready']
    assert result['mode'] == 'force_limited' and result['command_q'] == before


def test_slip_cancels_soft_relief_as_well_as_motion():
    state = {}
    for _ in range(80):
        tick(state, config=LIMIT_AND_SLOW)
    saw_slip = False
    for i in range(55):
        before = state['contact_command_closure']
        result = tick(state, (8.5, 15.), config=LIMIT_AND_SLOW, position_slip=(i + 1) * .00001)
        if result['slipping']:
            saw_slip = True
            assert not result['motion_ready'] and result['command_q'] == before
    assert saw_slip


def test_new_policy_still_pauses_on_unload_and_stops_on_raw_hard_overload():
    state = {}
    for _ in range(80):
        tick(state, config=LIMIT_AND_SLOW)
    before = state['contact_command_closure']
    result = tick(state, (8., 18.1), config=LIMIT_AND_SLOW)
    assert result['mode'] == 'unload' and result['command_q'] < before
    assert not result['motion_ready'] and result['motion_scale'] == 0.
    result = tick(state, (8., 41.), config=LIMIT_AND_SLOW)
    assert result['failure'] == 'loaded_gripper_force_limit'


def test_mild_band_needs_initial_stability_window_before_first_movement():
    state = {}
    for _ in range(23):
        assert not tick(state, (8.5, 15.), config=LIMIT_AND_SLOW)['motion_ready']
    result = tick(state, (8.5, 15.), config=LIMIT_AND_SLOW)
    assert result['motion_ready'] and 0 < result['motion_scale'] < .01


@pytest.mark.parametrize('forces,valid,error', [
    ((8., 0.), True, 'loaded_force_contact_missing'),
    ((8., 15.), False, 'loaded_force_probe_invalid'),
])
def test_new_policy_never_bypasses_probe_or_contact_failure(forces, valid, error):
    result = tick({}, forces, probes_valid=valid, config=LIMIT_AND_SLOW)
    assert result['failure'] == error and not result['motion_ready']


@pytest.fixture
def runtime(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    saved = {'object': 'part', 'arm_q': np.arange(6) * .01,
             'pose': {'position': np.zeros(3), 'orientation': np.array([1., 0., 0., 0.])},
             'gripper_openness': .5}
    attachment = {'robot_name': 'right', 'attach_spec': {'hold_force_control': copy.deepcopy(CONFIG)},
                  'hold_grasp_reference': {'captured_step': 100}}
    observation = {side: {'force': 8., 'force_probe_valid': True}
                   for side in ('left_finger', 'right_finger')}
    task = NS(step_counter=101, phase='carry', phase_index=1, phase_entry_step=101,
              robots={'right': NS(config=NS(gripper_close_openness=.08))},
              _ur5e_contact_hold_commands={'right': saved}, _attachments={'part': attachment},
              _contact_physics_dt=lambda: 1 / 240)
    valid = {'value': True}

    def hold_check(*args, **kwargs):
        attachment.update(current_contact_metrics=observation, current_contact_step=task.step_counter,
            last_hold_validation={'step': task.step_counter, 'position_slip_reference': 'grasp_point',
                                  'position_slip': 0.})
        return valid['value']

    task._physical_hold_valid = hold_check
    adapter._current_gripper_q = lambda **_: .8 * (1 - saved['gripper_openness'])
    adapter._gripper_open_closed_q = lambda **_: (0., .8)
    adapter._current_arm_q = lambda *_: saved['arm_q'].copy()
    adapter._limit_command_to_measured_state = lambda **kw: kw['command_q']
    adapter._current_robot_pose = lambda **_: saved['pose']
    adapter._current_tcp_pose = lambda **kw: kw['current_pose']
    adapter._debug_joint_step = lambda **_: None
    adapter._remember_arm_command = lambda *args: None
    adapter._remember_gripper_hold_openness = lambda **_: None
    monkeypatch.delenv('BEAM_HOLD_TRACE_PATH', raising=False)
    spec = {'name': 'carry', 'object': 'part', 'requires_held_object': True,
            'regulate_physical_carry': True, 'require_success': True}
    return NS(adapter=adapter, task=task, saved=saved, attachment=attachment,
              observation=observation, valid=valid, spec=spec)


def guard(r):
    return r.adapter._loaded_hold_force_guard(task=r.task, robot_name='right', spec=r.spec, tracked_robots={})


def test_runtime_pause_keeps_published_arm_pose_and_reference_not_measured(runtime):
    r = runtime
    r.observation['left_finger']['force'] = 5.
    anchor = copy.deepcopy(r.saved)
    r.adapter._current_arm_q = lambda *_: anchor['arm_q'] + .001
    _, action = guard(r)
    np.testing.assert_array_equal(action['arm_joint_controller'][0], anchor['arm_q'])
    np.testing.assert_array_equal(r.saved['pose']['position'], anchor['pose']['position'])
    assert action['gripper_controller'][0] < .5
    assert r.attachment['hold_grasp_reference'] == {'captured_step': 100}


def test_act_gates_before_target_generation_or_completion(runtime):
    r = runtime
    r.adapter._preshape_gripper_action = lambda **_: pytest.fail('Force pause must return before motion dispatch')
    action = r.adapter.act(task=r.task, robot_name='right', phase_spec={},
        skill_spec={**r.spec, 'name': 'ur5e_preshape_gripper'}, tracked_robots={}, tracked_objects={})
    np.testing.assert_array_equal(action['arm_joint_controller'][0], r.saved['arm_q'])


def test_runtime_update_and_publish_integrate_only_once_per_physics_step(runtime):
    r = runtime
    r.observation['left_finger']['force'] = 5.
    guard(r)
    first = copy.deepcopy(r.saved['loaded_force_state'])
    r.adapter._publish_loaded_action(task=r.task, robot_name='right', spec=r.spec,
                                    saved=r.saved, command_q=r.saved['arm_q'])
    assert r.saved['loaded_force_state'] == first


def test_stability_survives_phase_change_and_resumes_with_scaled_motion(runtime):
    r = runtime
    for _ in range(23):
        assert guard(r)[1] is not None
        r.task.step_counter += 1
    r.task.phase = 'carry_next'
    updated, action = guard(r)
    assert action is None
    assert 0 < updated['hold_force_motion_scale'] < .01
    assert 'hold_force_motion_scale' not in r.spec
    assert r.attachment['hold_grasp_reference']['captured_step'] == 100


def test_runtime_soft_relief_dispatches_motion_instead_of_frozen_arm_command(runtime):
    r = runtime
    r.attachment['attach_spec']['hold_force_control'] = copy.deepcopy(LIMIT_AND_SLOW)
    for _ in range(80):
        guard(r)
        r.task.step_counter += 1
    old_openness = r.saved['gripper_openness']
    r.observation['left_finger']['force'] = 8.5
    r.observation['right_finger']['force'] = 15.
    updated, action = guard(r)
    assert action is None  # The caller proceeds to its ordinary arm-motion dispatch.
    assert updated['hold_force_motion_scale'] == pytest.approx(.875)
    assert r.saved['gripper_openness'] > old_openness
    assert r.saved['loaded_force_state']['last_control']['mode'] == 'force_relief'
    assert r.attachment['hold_grasp_reference']['captured_step'] == 100


def test_ordinary_physical_hold_guard_still_rejects_bad_grasp(runtime):
    runtime.valid['value'] = False
    _, action = guard(runtime)
    assert action['__local_skill_failure__'] and action['reason'] == 'held_object_not_grasped'


def test_disabled_or_other_recipe_is_noop_without_accessing_force_sensors(runtime):
    r = runtime
    r.attachment['attach_spec'].pop('hold_force_control')
    r.task._physical_hold_valid = lambda *a, **kw: pytest.fail('Legacy path must remain untouched')
    assert guard(r) == (r.spec, None)


def test_force_pause_preserves_rotation_and_cartesian_command_caches(runtime):
    r = runtime
    r.saved['rotation_command_state'] = {'step': 100, 'orientation': [1., 0., 0., 0.], 'velocity': [0., .01, 0.]}
    before = copy.deepcopy(r.saved['rotation_command_state'])
    r.adapter._cartesian_command_positions['phase'] = np.ones(3)
    guard(r)
    assert r.saved['rotation_command_state'] == before
    np.testing.assert_array_equal(r.adapter._cartesian_command_positions['phase'], np.ones(3))


def test_target_object_servo_receives_smaller_steps_during_force_resume(runtime):
    r = runtime
    observed = {}

    def servo(**kw):
        observed.update(kw)
        return kw['target_pose']

    r.adapter._cartesian_command_servo_target_pose = servo
    target = {'position': np.ones(3), 'orientation': np.array([1., 0., 0., 0.])}
    r.adapter._target_object_servo_pose(phase_key=(), task=r.task, robot_name='right',
        spec={'servo_target_object_pose': False, 'cartesian_position_step': .001,
              'cartesian_orientation_step': .01, 'hold_force_motion_scale': .1},
        tracked_robots={}, tracked_objects={}, current_pose=r.saved['pose'], target_pose=target)
    assert observed['max_position_step'] == pytest.approx(.0001)
    assert observed['max_orientation_step'] == pytest.approx(.001)
