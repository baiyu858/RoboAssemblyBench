"""Real contact and sustained clamp load, never an aperture-only grasp."""
from types import SimpleNamespace as NS

import pytest

from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter


SPEC = dict(object='part', continuous_physics=True, close_force_regulation=True,
            close_force_seek_until_contact=True, close_force_require_clamp_band=True,
            jaw_width_thickness_m=.04821480592001373, jaw_width_stroke_m=.085,
            close_approach_joint_margin=.06, close_approach_joint_rate=.10,
            contact_force_threshold=.1, close_force_min_stable=4., close_force_low=4.,
            close_force_high=12., close_force_limit=40., close_force_validation_steps=96,
            close_contact_stable_steps=96, close_until_contact_min_steps=24)
NOMINAL_OPENNESS = .5322330108236909
NOMINAL_Q = .8 * (1. - NOMINAL_OPENNESS)


def metrics(left=0., right=0., valid=True):
    result = {side: dict(force=force, force_probe_valid=valid,
                        force_observation={'physics_dt': 1/240})
              for side, force in [('left_finger', left), ('right_finger', right)]}
    result.update(contact_box_center=[0., 0., 0.], contact_box_orientation=[1., 0., 0., 0.])
    return result


def control(state, observation, q=NOMINAL_Q, spec=SPEC):
    return UR5eAssemblyAtomicSkillAdapter._contact_close_control(
        state=state, spec=spec, metrics=observation, gripper_q=q,
        open_q=0., closed_q=.8, requested_openness=NOMINAL_OPENNESS, min_openness=.08)


def test_zero_contact_keeps_closing_past_old_aperture_at_slow_rate():
    result = control({'contact_command_closure': NOMINAL_Q}, metrics())
    assert result['command_q'] == pytest.approx(NOMINAL_Q + .1/240)
    assert not result['clamp_force_ready']
    assert not result['contact_control_active']


def test_unconfigured_legacy_control_keeps_its_requested_aperture():
    legacy = {k: v for k, v in SPEC.items() if k != 'close_force_seek_until_contact'}
    result = control({'contact_command_closure': NOMINAL_Q}, metrics(), spec=legacy)
    assert result['command_q'] == pytest.approx(NOMINAL_Q)


def test_no_contact_at_safe_travel_limit_never_counts_as_clamped():
    state = {'contact_command_closure': .736}
    result = control(state, metrics(), q=.736)
    assert result['command_q'] == pytest.approx(.736)
    assert not result['clamp_force_ready']


def test_invalid_probe_holds_and_overload_relieves_instead_of_squeezing():
    state = {'contact_command_closure': NOMINAL_Q}
    assert control(state, metrics(valid=False))['command_q'] <= NOMINAL_Q
    result = control(state, metrics(50., 50.))
    assert result['command_q'] < NOMINAL_Q
    assert not result['clamp_force_ready']
    assert not result['force_within_limit']


@pytest.fixture
def gate():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = NS(step_counter=0, robots={'right': NS(config=NS(gripper_close_openness=.08))})
    adapter._current_gripper_q = lambda **kw: NOMINAL_Q
    adapter._gripper_open_closed_q = lambda **kw: (0., .8)
    observation = metrics(6., 6.)
    strict = {'ready': True}
    adapter._grasp_contact_ready = lambda **kw: (
        strict['ready'], {'contact_metrics': observation, 'strict_contact_ready': strict['ready']})
    state = {}
    spec = SPEC.copy()
    def tick():
        task.step_counter += 1
        return adapter._close_until_contact_ready(
            state=state, task=task, robot_name='right', spec=spec,
            tracked_objects={}, close_elapsed_steps=task.step_counter + 24,
            gripper_openness=NOMINAL_OPENNESS)
    return NS(adapter=adapter, task=task, observation=observation, strict=strict, state=state, spec=spec, tick=tick)


@pytest.mark.parametrize('left,right', [(0., 0.), (6., 0.), (.2, .2), (3.9, 6.), (6., 13.), (50., 50.)])
def test_touch_single_finger_or_excess_load_cannot_pass_clamp_gate(gate, left, right):
    gate.observation.update(metrics(left, right))
    for _ in range(100):
        ready, detail = gate.tick()
        assert not ready
    assert detail['stable_steps'] == 0


def test_positive_forces_without_strict_opposing_target_contact_cannot_pass(gate):
    gate.strict['ready'] = False
    for _ in range(100):
        assert not gate.tick()[0]


def test_both_fingers_must_stay_clamped_for_96_samples_and_load_loss_restarts(gate):
    for _ in range(95):
        assert not gate.tick()[0]
    assert gate.tick()[0]
    gate.observation['right_finger']['force'] = 0.
    assert not gate.tick()[0]
    assert gate.state['close_contact_stable_steps'] == 0
    gate.observation['right_finger']['force'] = 6.
    for _ in range(95):
        assert not gate.tick()[0]
    assert gate.tick()[0]


def test_part_motion_resets_the_clamped_stability_window(gate):
    for _ in range(20):
        assert not gate.tick()[0]
    gate.observation['contact_box_center'][0] = .002
    ready, detail = gate.tick()
    assert not ready
    assert detail['stable_steps'] == 0


def enable_smoothing(gate):
    gate.spec['close_force_stability'] = {'enabled': True, 'window_steps': 5, 'hysteresis_margin': .5}


def test_smoothing_still_requires_full_force_window_and_96_distinct_stable_steps(gate):
    enable_smoothing(gate)
    for _ in range(4):
        ready, detail = gate.tick()
        assert not ready and detail['stable_steps'] == 0
    for _ in range(95):
        assert not gate.tick()[0]
    assert gate.tick()[0]
    assert gate.state['close_contact_stable_steps'] == 96


def test_small_force_jitter_passes_without_modifying_raw_force_controller(gate):
    enable_smoothing(gate)
    for _ in range(5):
        gate.tick()
    for i in range(96):
        gate.observation['left_finger']['force'] = 3.7 if i % 2 else 4.3
        ready, detail = gate.tick()
        assert detail['force_stability']['qualified']
        if i % 2:
            assert not detail['control']['clamp_force_ready']  # Raw diagnostic is unchanged.
    assert ready


@pytest.mark.parametrize('left,right', [(3.49, 6.), (0., 6.), (6., 12.01), (6., 50.)])
def test_smoothed_gate_clears_count_and_filter_on_raw_fault(gate, left, right):
    enable_smoothing(gate)
    for _ in range(30):
        gate.tick()
    gate.observation.update(metrics(left, right))
    ready, detail = gate.tick()
    assert not ready and detail['stable_steps'] == 0
    assert detail['force_stability']['window_samples'] == 0
    if right > 12.:
        assert detail['control']['mode'] in ('unload', 'force_limit_unload')


def test_smoothed_gate_keeps_strict_contact_and_pose_guards(gate):
    enable_smoothing(gate)
    for _ in range(30):
        gate.tick()
    gate.observation['contact_box_center'][0] = .002
    ready, detail = gate.tick()
    assert not ready and detail['stable_steps'] == 0
    assert detail['motion_detail']['position_error'] == pytest.approx(.002)
    gate.strict['ready'] = False
    ready, detail = gate.tick()
    assert not ready and detail['force_stability']['reason'] == 'target_contact_not_ready'


def test_smoothed_counter_cannot_advance_twice_in_same_physics_step(gate):
    enable_smoothing(gate)
    for _ in range(30):
        gate.tick()
    before = gate.state['close_contact_stable_steps']
    gate.task.step_counter -= 1
    assert not gate.tick()[0]
    assert gate.state['close_contact_stable_steps'] == before


def test_smoothed_history_is_cleared_on_grasp_loss_generation_change(gate):
    enable_smoothing(gate)
    for _ in range(30):
        gate.tick()
    gate.task._continuous_grasp_loss_generation = {'part': 1}
    ready, detail = gate.tick()
    assert not ready and detail['stable_steps'] == 0
    assert detail['force_stability']['window_samples'] == 1
